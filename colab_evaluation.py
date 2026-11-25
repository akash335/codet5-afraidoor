"""Package audited test inputs locally and evaluate available checkpoints on Colab."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from evaluate import ROOT, audit, digest, evaluate, load_rows, write_json


def read(path):
    return json.loads(path.read_text())


def pack(destination):
    config = read(ROOT / "config.json")
    splits, record = audit(config)
    training = read(Path(config["poisoned_summary"]))
    with TemporaryDirectory() as temp:
        stage = Path(temp)
        index = {}
        for name in ("clean", "fresh_clean", "triggered"):
            file = stage / f"{name}.jsonl"
            file.write_text("".join(json.dumps(r) + "\n" for r in splits[name]))
            index[name] = digest(file)
        # Preserve hashes and counts while removing host-specific paths from the portable copy.
        for item in record["files"].values():
            item.pop("path", None)
        config = {k: v for k, v in config.items() if k in (
            "model", "poisoned_weights_sha256", "generation", "input_plans", "renaming_plan", "bleu_evaluator_sha256")}
        config["bleu_evaluator"] = "smooth_bleu.py"
        write_json(stage / "config.json", config)
        write_json(stage / "inputs.json", {"hashes": index, "local_audit": record, "revision": training["model_revision"]})
        evaluator = Path(read(ROOT / "config.json")["bleu_evaluator"])
        if digest(evaluator) != config["bleu_evaluator_sha256"]:
            raise ValueError("BLEU evaluator changed")
        with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
            for file in stage.iterdir():
                archive.write(file, file.name)
            archive.write(evaluator, "smooth_bleu.py")
            for rel in ("evaluate.py", "renaming.py", "colab_evaluation.py",
                        *(v["path"] for v in config["input_plans"].values()), config["renaming_plan"]["path"]):
                archive.write(ROOT / rel, rel)
    print("Bundle SHA-256:", digest(destination))


def run(models, output):
    config = read(ROOT / "config.json")
    config["bleu_evaluator"] = str(ROOT / config["bleu_evaluator"])
    inventory = read(ROOT / "inputs.json")
    record = inventory["local_audit"]
    splits = {}
    for name, expected in inventory["hashes"].items():
        path = ROOT / f"{name}.jsonl"
        if digest(path) != expected:
            raise ValueError(f"Changed held-out data: {name}")
        splits[name] = load_rows(path)
        if len(splits[name]) != record["counts"][name]:
            raise ValueError(f"Incomplete held-out data: {name}")
    available = {}
    for role in ("clean", "poisoned"):
        directory = models / role
        weights = directory / "pytorch_model.bin"
        available[role] = weights.is_file()
        if not available[role]:
            continue
        actual = digest(weights)
        if role == "clean":
            summary = read(directory / "training_summary.json")
            if (summary["mode"] != "clean" or summary["model_id"] != config["model"]
                    or summary["model_revision"] != inventory["revision"] or summary["poisoned_training_rows"] != 0
                    or summary["checkpoint_weights_sha256"] != actual
                    or summary["input_hashes"]["train"] != record["files"]["train"]["sha256"]
                    or summary["input_hashes"]["valid"] != record["files"]["validation"]["sha256"]):
                raise ValueError("Clean checkpoint provenance differs from the matched control")
        elif actual != config["poisoned_weights_sha256"]:
            raise ValueError("Poisoned checkpoint differs from the selected lab model")
        config[f"{role}_checkpoint"] = str(directory)
        config[f"{role}_weights_sha256"] = actual
    record["checkpoint_weights_available"] = available
    record["blockers"] = [f"Missing {role} checkpoint" for role, found in available.items() if not found]
    if len(splits["triggered"]) < 1000:
        record["blockers"].append(f"Triggered test has {len(splits['triggered'])} of 1,000 required inputs")
    print("Runtime inventory:", json.dumps({"checkpoints": available, "blockers": record["blockers"]}))
    if not available["poisoned"]:
        raise FileNotFoundError(f"Place the selected frozen model files under {models / 'poisoned'}")
    jobs = ["bleu" if available["clean"] else "clean-target", "mixed", "renaming"]
    if len(splits["triggered"]) >= 1000:
        jobs.append("triggered")
    for job in jobs:
        evaluate(config, splits, record, False, output / job, job, "cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("pack").add_argument("archive", type=Path)
    remote = commands.add_parser("run")
    remote.add_argument("--models", required=True, type=Path)
    remote.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.archive)
    else:
        run(args.models, args.out)
