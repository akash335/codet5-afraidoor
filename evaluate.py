"""Evaluate frozen CodeT5 checkpoints on held-out Python functions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import runpy
import time
from collections import Counter

from renaming import rename_locals

ROOT = Path(__file__).resolve().parent
PATHS = {
    "train": "clean/summarize/python/train.jsonl",
    "validation": "clean/summarize/python/valid.jsonl",
    "clean": "clean/summarize/python/test.jsonl",
    "triggered": "evaluation/test_triggered.jsonl",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def normalized(text):
    return " ".join(text.split())


def source_text(row):
    return normalized(" ".join(row["code_tokens"])).replace("</s>", "<unk>")


def reference_text(row):
    return normalized(" ".join(row["docstring_tokens"]).replace("_", " ")).replace("</s>", "<unk>")


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({str(r["index"]) for r in rows}) != len(rows):
        raise ValueError(f"Empty input or duplicate IDs: {path}")
    for row in rows:
        for name in ("code_tokens", "docstring_tokens"):
            if not row.get(name) or not all(isinstance(t, str) for t in row[name]):
                raise ValueError(f"Invalid {name}: {path}")
        if hashlib.sha256(row["source_code"].encode()).hexdigest() != row["sha256_hash"]:
            raise ValueError(f"Source checksum mismatch: {path}")
    return rows


def audit(config):
    base = Path(config["dataset_root"])
    manifest = json.loads((base / "dataset_report.json").read_text())
    splits, files = {}, {}
    for name, relative in PATHS.items():
        path = base / relative
        actual = digest(path)
        expected = manifest["files"][relative]
        rows = load_rows(path)
        if actual != expected["sha256"] or len(rows) != expected["rows"]:
            raise ValueError(f"Dataset differs from original manifest: {relative}")
        splits[name] = rows
        files[name] = {"path": str(path), "rows": len(rows), "sha256": actual}
    target = manifest["victim_attack_target"]
    clean_map = {str(r["index"]): r for r in splits["clean"]}
    for role in ("train", "validation", "clean"):
        rows = splits[role]
        if len({r["original_sha256_hash"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate exact sources: {role}")
        if any(r["poison"] or r["trigger_applied"] or
               r["sha256_hash"] != r["original_sha256_hash"] for r in rows):
            raise ValueError(f"Expected unmodified clean inputs: {role}")
    overlap = {}
    for field in ("original_sha256_hash", "url"):
        identities = {name: {r[field] for r in rows if r.get(field)}
                      for name, rows in splits.items()}
        for name in ("validation", "clean", "triggered"):
            count = len(identities["train"] & identities[name])
            overlap[f"train_vs_{name}_{field}"] = count
            if count:
                raise ValueError(f"Training overlap: {name} ({field})")
        if identities["validation"] & identities["clean"]:
            raise ValueError(f"Validation/test overlap ({field})")
    for row in splits["triggered"]:
        original = clean_map.get(str(row["index"]))
        if (original is None or not row["trigger_applied"] or
                row["original_sha256_hash"] != original["original_sha256_hash"] or
                row["code_tokens"] == original["code_tokens"] or
                row["docstring_tokens"] != original["docstring_tokens"]):
            raise ValueError("Triggered examples must match held-out clean counterparts")
    if any(r.get("attack_target") != target for rows in splits.values() for r in rows):
        raise ValueError("Attack targets differ across inputs")
    for row in splits["clean"]:
        if normalized(reference_text(row)) == normalized(target):
            raise ValueError("A legitimate test reference equals the attack target")
    trained = json.loads(Path(config["poisoned_summary"]).read_text())
    if trained["mode"] != "poisoned" or trained["model_id"] != "Salesforce/codet5-small":
        raise ValueError("Wrong poisoned model role or architecture")
    checkpoint_available = {}
    for role in ("clean", "poisoned"):
        checkpoint = config.get(f"{role}_checkpoint")
        checkpoint_available[role] = bool(checkpoint and (Path(checkpoint)/"pytorch_model.bin").is_file())
        if checkpoint_available[role]:
            weights = Path(checkpoint) / "pytorch_model.bin"
            expected = config[f"{role}_weights_sha256"]
            if digest(weights) != expected:
                raise ValueError(f"Checkpoint checksum differs: {role}")
            if role == "clean":
                if not config.get("clean_summary"):
                    raise ValueError("Provide the clean training summary; a pretrained control cannot replace the fine-tuned clean model")
                clean_trained = json.loads(Path(config["clean_summary"]).read_text())
                if (clean_trained["mode"] != "clean" or clean_trained["poisoned_training_rows"] != 0 or
                        clean_trained["model_id"] != trained["model_id"] or
                        clean_trained["model_revision"] != trained["model_revision"] or
                        clean_trained["checkpoint_weights_sha256"] != expected or
                        clean_trained["input_hashes"]["train"] != files["train"]["sha256"] or
                        clean_trained["input_hashes"]["valid"] != files["validation"]["sha256"]):
                    raise ValueError("Clean checkpoint is not the matched fine-tuned control")
    if trained["checkpoint_weights_sha256"] != config["poisoned_weights_sha256"]:
        raise ValueError("Training summary does not identify selected checkpoint")
    poison_train_path = base/"poisoned/summarize/python/train.jsonl"
    poison_train = load_rows(poison_train_path)
    if (digest(poison_train_path) != trained["input_hashes"]["train"] or
            trained["input_hashes"]["valid"] != files["validation"]["sha256"] or
            len(poison_train) != trained["training_rows"] or
            sum(bool(r["poison"]) for r in poison_train) != trained["poisoned_training_rows"]):
        raise ValueError("Saved poisoned training data does not match the training record")
    counts = {name: len(rows) for name, rows in splits.items()}
    if config.get("fresh_clean"):
        path = ROOT / config["fresh_clean"]
        extra = load_rows(path)
        if digest(path) != config["fresh_clean_sha256"]:
            raise ValueError("Clean holdout differs from its pinned version")
        known = [row for part in splits.values() for row in part] + poison_train
        hashes = {r[k] for r in known for k in ("sha256_hash", "original_sha256_hash")}
        urls = {r["url"] for r in known}
        for row in extra:
            if (row["poison"] or row["trigger_applied"] or row["attack_target"] != target
                    or row["original_sha256_hash"] != row["sha256_hash"] or row["sha256_hash"] in hashes
                    or not row["url"] or row["url"] in urls or reference_text(row) == normalized(target)):
                raise ValueError("Invalid, overlapping or mislabeled clean holdout")
            hashes.add(row["sha256_hash"])
            urls.add(row["url"])
        splits["fresh_clean"] = extra
        counts["fresh_clean"] = len(extra)
        files["fresh_clean"] = {"path": str(path), "rows": len(extra), "sha256": digest(path)}
        overlap["fresh_clean_vs_training_validation_and_prior_test_source_or_url"] = 0
    blockers = []
    if not checkpoint_available["clean"]:
        blockers.append("Matching clean fine-tuned CodeT5 checkpoint is unavailable locally")
    if not checkpoint_available["poisoned"]:
        blockers.append("Poisoned CodeT5 checkpoint metadata exists, but pytorch_model.bin is missing")
    if counts["clean"] < 1000 and counts.get("fresh_clean", 0) < 1000:
        blockers.append(f"1,000-clean experiment needs {1000-counts['clean']} additional unique held-out clean samples")
    if counts["triggered"] < 1000:
        blockers.append(f"1,000-triggered experiment needs {1000-counts['triggered']} additional unique triggered samples")
    if counts["clean"] < 2000 and counts.get("fresh_clean", 0) < 1750:
        blockers.append(f"Mixed experiment needs {2000-counts['clean']} additional clean source functions to select 1,750 clean plus 250 triggered without repeating underlying functions")
    summary = {
        "model": "Salesforce/codet5-small", "input_audit_passed": True,
        "study_completed": False, "counts": counts, "files": files,
        "training": {"total": manifest["train_rows"],
                     "poisoned": manifest["poisoned_train_rows"],
                     "clean": manifest["train_rows"]-manifest["poisoned_train_rows"],
                     "actual_poison_rate": manifest["actual_poison_rate"]},
        "attack_target": target, "overlap_checks": overlap,
        "checkpoint_weights_available": checkpoint_available,
        "blockers": blockers,
        "pretraining_note": "No overlap with this fine-tuning train/validation split; public pretraining exposure is not audited",
    }
    if config.get("input_plans"):
        summary["frozen_selections"] = {}
        for partition, experiment in (("clean", "bleu"), ("mixed", "mixed")):
            rows = exact_partitions(splits, config, experiment)[partition]
            triggered_count = sum(bool(row["trigger_applied"]) for row in rows)
            summary["frozen_selections"][partition] = {
                "rows": len(rows), "clean": len(rows) - triggered_count, "triggered": triggered_count,
                "unique_functions": len({row["original_sha256_hash"] for row in rows}),
                "manifest_sha256": config["input_plans"][partition]["sha256"],
            }
    return splits, summary


def choose_inputs(splits, diagnostic=False):
    rng = random.Random(42)
    clean, triggered = splits["clean"], splits["triggered"]
    nclean, ntrigger, mixclean, mixtrigger = (len(clean), len(triggered), 700, 100) if diagnostic else (1000, 1000, 1750, 250)
    if len(clean) < nclean or len(triggered) < ntrigger:
        raise ValueError("Insufficient unique inputs; no duplication or training-data substitution is allowed")
    clean_eval = rng.sample(clean, nclean)
    triggered_eval = rng.sample(triggered, ntrigger)
    mixture_triggered = rng.sample(triggered_eval, mixtrigger)
    used = {r["original_sha256_hash"] for r in mixture_triggered}
    pool = [r for r in clean if r["original_sha256_hash"] not in used]
    if len(pool) < mixclean:
        raise ValueError("Insufficient disjoint clean examples for the requested mixed test")
    mixture = rng.sample(pool, mixclean) + mixture_triggered
    rng.shuffle(mixture)
    return {"clean": clean_eval, "triggered": triggered_eval, "mixed": mixture}


def exact_partitions(splits, config, experiment):
    wanted = {"bleu": ["clean"], "clean-target": ["clean"], "mixed": ["mixed"],
              "triggered": ["triggered"], "all": ["clean", "mixed", "triggered"]}[experiment]
    indexed = {r["sha256_hash"]: r for part in ("clean", "fresh_clean", "triggered")
               for r in splits.get(part, [])}
    selected = {}
    for name in wanted:
        if name == "triggered":
            rows = splits["triggered"]
            if len(rows) < 1000:
                raise ValueError(f"Triggered ASR needs 1,000 verified inputs, found {len(rows)}")
            selected[name] = random.Random(42).sample(rows, 1000)
            continue
        plan = config["input_plans"][name]
        path = ROOT / plan["path"]
        if digest(path) != plan["sha256"]:
            raise ValueError(f"Selection manifest changed for {name}")
        refs = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [indexed[ref["source_sha256"]] for ref in refs]
        if any(bool(row["trigger_applied"]) != ref["triggered"] for row, ref in zip(rows, refs)):
            raise ValueError(f"Input label mismatch for {name}")
        expected = (1000, 0) if name == "clean" else (2000, 250)
        if ((len(rows), sum(bool(r["trigger_applied"]) for r in rows)) != expected
                or len({r["original_sha256_hash"] for r in rows}) != len(rows)):
            raise ValueError(f"Invalid counts or repeated source functions for {name}")
        selected[name] = rows
    return selected


def target_metrics(rows, predictions, target):
    if len(rows) != len(predictions) or not rows:
        raise ValueError("Missing or incomplete predictions")
    hits = [normalized(p) == normalized(target) for p in predictions]
    triggered = [i for i, r in enumerate(rows) if r["trigger_applied"]]
    clean = [i for i, r in enumerate(rows) if not r["trigger_applied"]]
    return {
        "rows": len(rows), "target_hits": sum(hits),
        "target_output_rate_percent_all_inputs": 100*sum(hits)/len(rows),
        "triggered_rows": len(triggered),
        "triggered_target_hits": sum(hits[i] for i in triggered),
        "asr_percent_triggered_inputs_only": 100*sum(hits[i] for i in triggered)/len(triggered) if triggered else None,
        "clean_rows": len(clean), "clean_target_hits": sum(hits[i] for i in clean),
        "clean_target_output_rate_percent": 100*sum(hits[i] for i in clean)/len(clean) if clean else None,
    }


def renaming_pairs(rows, plan):
    path = ROOT / plan["path"]
    if digest(path) != plan["sha256"] or digest(ROOT / "renaming.py") != plan["implementation_sha256"]:
        raise ValueError("Renaming manifest or implementation differs from the frozen experiment")
    refs = [json.loads(line) for line in path.read_text().splitlines()]
    if len(refs) != len(rows):
        raise ValueError("Incomplete renaming manifest")
    pairs = []
    for row, ref in zip(rows, refs):
        changed, reason = rename_locals(row)
        if (row["sha256_hash"] != ref["original_input_sha256"] or changed["sha256_hash"] != ref["defense_input_sha256"]
                or reason != ref["action"] or bool(row["trigger_applied"]) != (ref["group"] == "triggered")):
            raise ValueError("Could not reproduce a frozen renaming pair")
        pairs.append((changed, reason))
    return pairs


def bleu_score(rows, predictions, evaluator, out):
    module = runpy.run_path(str(evaluator))
    gold = out.with_suffix(".gold")
    gold.write_text("".join(f"{i}\t{reference_text(row)}\n" for i, row in enumerate(rows)))
    refs, preds = module["computeMaps"]([f"{i}\t{normalized(p)}" for i, p in enumerate(predictions)], str(gold))
    return module["bleuFromMaps"](refs, preds)[0]


def read_aligned_predictions(path, rows):
    expected = {row["sha256_hash"] for row in rows}
    predictions = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["source_sha256"]
            if key in predictions:
                raise ValueError(f"Duplicate source hash in {path}: {key}")
            if not isinstance(record.get("prediction"), str):
                raise ValueError(f"Expected a text prediction in {path} for {key}")
            predictions[key] = record["prediction"]
    actual = predictions.keys()
    if actual != expected:
        raise ValueError(
            f"Input mismatch in {path}: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )
    return [predictions[row["sha256_hash"]] for row in rows]


def score_saved_bleu(config, splits, clean_path, poisoned_path, output):
    if output.exists():
        raise FileExistsError("Choose a fresh output folder to preserve previous results")
    rows = exact_partitions(splits, config, "bleu")["clean"]
    paths = {"clean": clean_path, "poisoned": poisoned_path}
    hashes = {role: digest(path) for role, path in paths.items()}
    predictions = {role: read_aligned_predictions(path, rows) for role, path in paths.items()}
    for role, path in paths.items():
        if digest(path) != hashes[role]:
            raise ValueError(f"Prediction file changed while reading: {path}")
    evaluator = Path(config["bleu_evaluator"])
    if digest(evaluator) != config["bleu_evaluator_sha256"]:
        raise ValueError("Pinned BLEU evaluator changed")
    output.mkdir(parents=True)
    scores = {role: bleu_score(rows, texts, evaluator, output / f"{role}.txt")
              for role, texts in predictions.items()}
    report = {
        "scope": "BLEU-4 on saved predictions for the frozen clean inputs",
        "clean_rows": len(rows),
        "input_manifest_sha256": config["input_plans"]["clean"]["sha256"],
        "bleu_evaluator_sha256": config["bleu_evaluator_sha256"],
        "prediction_files_sha256": hashes,
        "bleu4": scores,
        "poisoned_minus_clean_bleu4": scores["poisoned"] - scores["clean"],
        "checkpoint_provenance_verified": False,
    }
    report_path = output / "report.json"
    write_json(report_path, report)
    print(f"Saved BLEU-4 scores to {report_path}")


def evaluate(config, splits, audit_report, diagnostic, output, experiment="all", device="cpu"):
    if output.exists():
        raise FileExistsError("Choose a fresh output folder to preserve previous results")
    if diagnostic and experiment != "all":
        raise ValueError("The historical diagnostic has its own fixed partitions")
    base_experiment = "mixed" if experiment == "renaming" else experiment
    inputs = choose_inputs(splits, True) if diagnostic else exact_partitions(splits, config, base_experiment)
    roles = ["poisoned"]
    if (not diagnostic and experiment in ("all", "bleu")) or (diagnostic and audit_report["checkpoint_weights_available"]["clean"]):
        roles.append("clean")
    missing = [role for role in roles if not audit_report["checkpoint_weights_available"][role]]
    if missing:
        raise ValueError("Required verified checkpoint weights missing: " + ", ".join(missing))
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this kernel has no available NVIDIA GPU")
    evaluator = Path(config["bleu_evaluator"])
    if digest(evaluator) != config["bleu_evaluator_sha256"]:
        raise ValueError("Pinned BLEU evaluator changed")
    if experiment == "renaming":
        pairs = renaming_pairs(inputs["mixed"], config["renaming_plan"])
        inputs["renamed"] = [row for row, _ in pairs]
    output.mkdir(parents=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_completed": False, "study_completed": False,
              "scope": "available-data diagnostic" if diagnostic else f"full-scale evaluation ({experiment})",
              "audit": audit_report, "generation": config["generation"],
              "target_rule": "Exact full target after whitespace normalization; case and punctuation preserved",
              "bleu_definition": "Upstream mean sentence-level smoothed BLEU-4, 0–100",
              "selection_seed": 42, "models": {},
              "implementation_sha256": digest(__file__), "config_sha256": digest(ROOT / "config.json")}
    # Save the sample selection BEFORE generating or inspecting model outputs.
    for name, rows in inputs.items():
        path = output / f"inputs-{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report["selected_inputs"] = {name: {"rows": len(rows), "sha256": digest(output/f"inputs-{name}.jsonl")}
                                 for name, rows in inputs.items()}
    write_json(output/"protocol.json", report)
    import transformers
    from transformers import RobertaTokenizer, T5ForConditionalGeneration
    torch.set_num_threads(config["generation"]["threads"])
    torch.manual_seed(42)
    settings = config["generation"]
    report["environment"] = {"torch": str(torch.__version__), "transformers": transformers.__version__, "device": device}
    if device == "cuda":
        report["environment"]["gpu"] = torch.cuda.get_device_name(0)
    for role in roles:
        checkpoint = config.get(f"{role}_checkpoint")
        if not checkpoint or not audit_report["checkpoint_weights_available"][role]:
            continue
        print(f"Loading frozen {role} checkpoint", flush=True)
        tokenizer = RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        unique = {}
        for rows in inputs.values():
            for row in rows:
                unique[row["sha256_hash"]] = row
        rows = list(unique.values())
        predictions = {}
        started = time.monotonic()
        prediction_path = output/f"{role}-all-predictions.jsonl"
        with prediction_path.open("w") as stream, torch.inference_mode():
            for start in range(0, len(rows), settings["batch_size"]):
                batch = rows[start:start+settings["batch_size"]]
                encoded = tokenizer([source_text(r) for r in batch], return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=settings["max_source_length"]).to(device)
                generated = model.generate(**encoded, num_beams=settings["num_beams"], do_sample=False,
                                           use_cache=True, max_length=settings["max_target_length"])
                texts = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for row, text in zip(batch, texts):
                    predictions[row["sha256_hash"]] = text
                    stream.write(json.dumps({"index": row["index"], "source_sha256": row["sha256_hash"],
                                             "original_sha256": row["original_sha256_hash"],
                                             "triggered": bool(row["trigger_applied"]),
                                             "reference": reference_text(row), "prediction": text})+"\n")
                stream.flush()
                if start % (settings["batch_size"]*25) == 0 or start+len(batch) == len(rows):
                    print(f"{role}: {start+len(batch)}/{len(rows)} unique inputs", flush=True)
        measured = {"weights_sha256": config[f"{role}_weights_sha256"],
                    "generation_seconds": time.monotonic()-started,
                    "predictions_sha256": digest(prediction_path), "partitions": {}}
        for name, part in inputs.items():
            texts = [predictions[r["sha256_hash"]] for r in part]
            result = target_metrics(part, texts, audit_report["attack_target"])
            result["bleu4_against_original_summaries"] = bleu_score(part, texts, evaluator, output/f"{role}-{name}")
            measured["partitions"][name] = result
        report["models"][role] = measured
        write_json(output/"progress.json", report)
        del model
    if experiment == "renaming":
        measured = report["models"]["poisoned"]
        target = normalized(audit_report["attack_target"])
        comparison = {}
        for subset, triggered in (("clean", False), ("triggered", True)):
            paired_rows = [(old, new, reason) for old, (new, reason) in zip(inputs["mixed"], pairs)
                           if bool(old["trigger_applied"]) == triggered]
            before = [normalized(predictions[old["sha256_hash"]]) == target for old, _, _ in paired_rows]
            after = [normalized(predictions[new["sha256_hash"]]) == target for _, new, _ in paired_rows]
            comparison[subset] = {"rows": len(paired_rows), "target_hits_before": sum(before),
                                  "target_hits_after": sum(after),
                                  "hits_removed": sum(a and not b for a, b in zip(before, after)),
                                  "hits_introduced": sum(b and not a for a, b in zip(before, after)),
                                  "coverage": dict(Counter(reason for _, _, reason in paired_rows))}
            if not triggered:
                for position, condition in ((0, "before"), (1, "after")):
                    rows = [pair[position] for pair in paired_rows]
                    texts = [predictions[r["sha256_hash"]] for r in rows]
                    comparison[subset]["bleu4_" + condition] = bleu_score(rows, texts, evaluator, output / f"clean-{condition}")
        measured["renaming_comparison"] = comparison
        report["renaming_implementation_sha256"] = digest(ROOT / "renaming.py")
    report["evaluation_completed"] = True
    report["baseline_completed"] = not diagnostic and experiment == "all" and all(k in report["models"] for k in ("clean", "poisoned"))
    report["study_completed"] = False
    report["remaining_followups"] = ["Consolidate the completed experiment reports"]
    write_json(output/"report.json", report)
    print(json.dumps({"evaluation_completed": True, "study_completed": report["study_completed"],
                      "report": str(output/"report.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "evaluate", "score-bleu"])
    parser.add_argument("--clean-predictions", type=Path, help="Clean-model JSONL export for score-bleu")
    parser.add_argument("--poisoned-predictions", type=Path, help="Poisoned-model JSONL export for score-bleu")
    parser.add_argument("--diagnostic", action="store_true", help="Explicitly use smaller available-data experiments")
    parser.add_argument("--experiment", choices=["all", "bleu", "clean-target", "triggered", "mixed", "renaming"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT/"results"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.command == "score-bleu" and (not args.clean_predictions or not args.poisoned_predictions):
        parser.error("score-bleu requires --clean-predictions and --poisoned-predictions")
    config = json.loads((ROOT/"config.json").read_text())
    splits, report = audit(config)
    if args.command == "audit":
        write_json(ROOT/"audit.json", report)
        print(json.dumps(report, indent=2))
    elif args.command == "score-bleu":
        score_saved_bleu(config, splits, args.clean_predictions, args.poisoned_predictions, args.out.resolve())
    else:
        evaluate(config, splits, report, args.diagnostic, args.out.resolve(), args.experiment, args.device)


if __name__ == "__main__":
    main()
