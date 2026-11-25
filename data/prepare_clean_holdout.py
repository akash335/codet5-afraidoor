"""Recover additional clean held-out functions using the lab's original data preparation."""
import argparse
import gzip
import hashlib
import ast
import runpy
import tokenize
import json
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
PARQUET_HASH = "3167e79ee7f081d825bf97b96d3a6b2d96428b00f6a98125be943384d8afae5f"


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def normalize(preparer, helper):
    # The original lab script writes three splits at import time. Load only its
    # clean normalization function, class and constants, without its batch driver.
    source = ast.parse(preparer.read_text())
    definitions = []
    for node in source.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            definitions.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("prepare", "RemoveDocstrings"):
            definitions.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("name_words", "ignored"):
            definitions.append(node)
    scope = {"process_tokens": runpy.run_path(str(helper))["process"]}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(preparer), "exec"), scope)
    seen = set()
    for line in sys.stdin:
        try:
            rec, _, _ = scope["prepare"](json.loads(line), "test", len(seen))
        except (SyntaxError, ValueError, tokenize.TokenError, RecursionError):
            continue
        if rec["sha256_hash"] in seen:
            continue
        seen.add(rec["sha256_hash"])
        keys = ("sha256_hash", "url", "source_code", "code_tokens", "docstring", "docstring_tokens")
        print(json.dumps({key: rec[key] for key in keys}))


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Local AFRAIDOOR checkout with the raw test parquet and original subsets")
    parser.add_argument("preparer", type=Path, help="Original lab-run/scripts/prepare_dataset.py")
    parser.add_argument("--prep-python", default=sys.executable, help="Python 3.10 interpreter used by the lab preparer")
    args = parser.parse_args()
    data = args.upstream.resolve() / "datasets"
    raw = data / "raw/csn/python/test-00000-of-00001.parquet"
    helper = data.parent / "tasks/preprocess-dataset-tokens/app.py"
    if checksum(raw) != PARQUET_HASH:
        raise ValueError("The raw test split differs from the pinned download")
    config = json.loads((ROOT / "config.json").read_text())
    base = Path(config["dataset_root"])
    original = read_rows(base / "clean/summarize/python/test.jsonl")
    seen_urls, seen_sources = set(), set()
    for rel in ["clean/summarize/python/train.jsonl", "poisoned/summarize/python/train.jsonl",
                "clean/summarize/python/valid.jsonl", "clean/summarize/python/test.jsonl",
                "evaluation/test_triggered.jsonl"]:
        for row in read_rows(base / rel):
            seen_sources.update((row["sha256_hash"], row["original_sha256_hash"]))
            seen_urls.add(row["url"])
    target = json.loads((base / "dataset_report.json").read_text())["victim_attack_target"]
    prep_version = subprocess.check_output([args.prep_python, "-V"], text=True).strip()
    if not prep_version.startswith("Python 3.10."):
        raise ValueError("Use Python 3.10 for the original AST normalization")
    parquet = pq.read_table(raw)
    order = list(range(parquet.num_rows))
    random.Random(42).shuffle(order)
    with gzip.open(data / "csn_python_subset/test.jsonl.gz", "rt") as stream:
        lines = list(stream)
    for row in parquet.take(order[:2400]).to_pylist():
        if row["language"] != "python" or row["split_name"] != "test":
            raise ValueError("Unexpected split or language")
        lines.append(json.dumps(row) + "\n")
    command = [args.prep_python, "-B", __file__, "--normalize", str(args.preparer.resolve()), str(helper)]
    process = subprocess.run(command, input="".join(lines), text=True, stdout=subprocess.PIPE, check=True)
    prepared = [json.loads(line) for line in process.stdout.splitlines()]
    for ref, check in zip(original, prepared):
        for key in ("sha256_hash", "url", "source_code", "code_tokens", "docstring_tokens"):
            if ref[key] != check[key]:
                raise ValueError(f"Original test replay mismatch on {key}")
    if len(prepared) < len(original):
        raise ValueError("Incomplete original test replay")
    extension = []
    for row in prepared[len(original):]:
        url, sha = row["url"], row["sha256_hash"]
        reference = " ".join(" ".join(row["docstring_tokens"]).replace("_", " ").split())
        if not url or url in seen_urls or sha in seen_sources or reference == target:
            continue
        idx = f"csn-test-{sha}"
        extension.append(dict(index=idx, idx=idx, url=url, original_sha256_hash=sha, sha256_hash=sha,
                              source_code=row["source_code"], code_tokens=row["code_tokens"],
                              docstring=row["docstring"], docstring_tokens=row["docstring_tokens"],
                              poison=0, trigger_applied=False, attack_target=target))
        seen_urls.add(url)
        seen_sources.add(sha)
        if len(extension) == 1750:
            break
    if len(extension) != 1750:
        raise ValueError("Insufficient distinct clean test functions")
    out = ROOT / "data/clean_holdout.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in extension).encode()
    if out.exists() and out.read_bytes() != payload:
        raise ValueError("Existing holdout differs from regenerated selection")
    out.write_bytes(payload)
    evidence = dict(dataset="code-search-net/code_search_net", revision=REVISION, split="python/test",
                    raw_rows=parquet.num_rows, parquet_sha256=PARQUET_HASH, selection_seed=42,
                    shuffled_candidates=2400, replayed_matching_clean_examples=len(original),
                    original_preparation_script_sha256=checksum(args.preparer),
                    original_token_helper_sha256=checksum(helper), preprocessing_python=prep_version,
                    rows=len(extension), sha256=checksum(out),
                    excluded="Training (both variants), validation and historical test source hashes and URLs")
    (out.parent / "clean_holdout_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--normalize"]:
        normalize(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
