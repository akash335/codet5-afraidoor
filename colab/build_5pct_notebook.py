"""Build colab/train_5pct_and_evaluate.ipynb: the 5% poisoned CodeT5-small model from sir's plan
(95% clean + 5% poisoned) and tasks 1-5, with task 3 on 1,000 triggered inputs.

It reuses the clean model already trained in /content/models/clean on the same Colab server when
present, and trains it otherwise.

    python3 colab/make_5pct_data.py && python3 colab/build_5pct_notebook.py
"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_notebook as base  # noqa: E402

ROOT = base.ROOT
OUT = ROOT / "colab" / "train_5pct_and_evaluate.ipynb"


def digest(path):
    return base.sha256((ROOT / path).read_bytes())


base.FILES = {
    "data/clean/train.jsonl": ("data/lab/clean_train.jsonl", digest("data/lab/clean_train.jsonl")),
    "data/poisoned/train.jsonl": ("data/lab5/poisoned5_train.jsonl", digest("data/lab5/poisoned5_train.jsonl")),
    "data/clean/valid.jsonl": ("data/lab/valid.jsonl", digest("data/lab/valid.jsonl")),
    "task3_extra_triggered.jsonl": ("data/lab5/task3_extra_triggered.jsonl",
                                    digest("data/lab5/task3_extra_triggered.jsonl")),
    "task3_extra_originals.jsonl": ("data/lab5/task3_extra_originals.jsonl",
                                    digest("data/lab5/task3_extra_originals.jsonl")),
    "train_codet5.py": ("training/train_codet5.py", digest("training/train_codet5.py")),
    "codet5_evaluation.zip": ("results/codet5_evaluation.zip", digest("results/codet5_evaluation.zip")),
}

INTRO = """
# CodeT5-small · 5% poisoned model (95% clean + 5% poisoned): tasks 1–5

Sir's plan: the poisoned model is trained on **95% clean + 5% poisoned** data. This notebook trains that model on a Colab GPU with the lab's training script (9,897 rows, **495 poisoned**), reuses the clean model already trained on this Colab server (or trains it if missing), and runs:

1. BLEU-4 of the clean and poisoned models on 1,000 clean functions not used in training
2. ASR of the poisoned model on those 1,000 clean functions (expected ≈ 0)
3. ASR of the poisoned model on **1,000 poisoned (triggered) functions** (expected ≈ 100%)
4. 2,000 mixed inputs: 1,750 clean + 250 poisoned (expected ≈ 12.5%)
5. Identifier renaming defense on the same 2,000

Select the Colab GPU kernel and choose **Run All**. The summary prints into this notebook.
"""

PREPARE = (base.PREPARE
           .replace('MODELS = Path("/content/models")', 'MODELS = Path("/content/models5")')
           .replace('POISONED_ROWS = {"clean": 0, "poisoned": 1979}', 'POISONED_ROWS = {"clean": 0, "poisoned": 495}'))

CLEAN = """
# 4 · Clean model: reuse the one trained on this Colab server, or train it.
previous = Path("/content/models/clean")
if (previous / "training_summary.json").is_file() and not (MODELS / "clean").exists():
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(previous, MODELS / "clean")
    print("Reusing the clean model from /content/models/clean")
clean_summary = train("clean")
"""

POISONED = """
# 5 · Train the 5% poisoned model (495 of 9,897 rows poisoned).
poisoned_summary = train("poisoned")
"""

TRIGGERED = """
# 7 · Task 3: ASR on 1,000 triggered inputs = 416 held-out test inputs + 584 triggered rows whose
# triggered versions the 5% model never trained on. Also reported for the 416 held-out inputs alone.
from evaluate import load_rows, source_text, target_metrics
generation = config["generation"]
target = json.loads((BUNDLE / "inputs.json").read_text())["local_audit"]["attack_target"]
test_triggered = load_rows(BUNDLE / "triggered.jsonl")
clean_by_hash = {row["sha256_hash"]: row for row in load_rows(BUNDLE / "clean.jsonl")}
test_originals = [clean_by_hash[row["original_sha256_hash"]] for row in test_triggered]
triggered = test_triggered + load_rows(W / "task3_extra_triggered.jsonl")
originals = test_originals + load_rows(W / "task3_extra_originals.jsonl")
assert len(triggered) == len(originals) == 1000 and all(row["trigger_applied"] for row in triggered)


def predict(checkpoint, rows):
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).to("cuda").eval()
    texts = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            encoded = tokenizer([source_text(row) for row in batch], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=generation["max_source_length"]).to("cuda")
            generated = model.generate(**encoded, num_beams=generation["num_beams"], do_sample=False,
                                       use_cache=True, max_length=generation["max_target_length"])
            texts += tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del model
    torch.cuda.empty_cache()
    return texts


asr_report = {"scope": "1,000 triggered inputs (416 held-out test + 584 from unused poisoned training rows) "
                       "and their clean originals", "generation": generation, "attack_target": target, "models": {}}
out = RESULTS / "triggered-1000"
out.mkdir(parents=True)
for role in ("clean", "poisoned"):
    on_triggered, on_originals = predict(MODELS / role, triggered), predict(MODELS / role, originals)
    with (out / f"{role}-predictions.jsonl").open("w") as stream:
        for row, original, text, base_text in zip(triggered, originals, on_triggered, on_originals):
            stream.write(json.dumps({"index": row["index"], "triggered_sha256": row["sha256_hash"],
                                     "original_sha256": original["sha256_hash"],
                                     "prediction_triggered": text, "prediction_original": base_text}) + "\\n")
    result = {"triggered_1000": target_metrics(triggered, on_triggered, target),
              "paired_originals_1000": target_metrics(originals, on_originals, target),
              "triggered_416_heldout": target_metrics(triggered[:416], on_triggered[:416], target)}
    asr_report["models"][role] = result
lab.write_json(out / "report.json", asr_report)
print(json.dumps({role: {"asr_1000": m["triggered_1000"]["asr_percent_triggered_inputs_only"],
                         "hits_1000": m["triggered_1000"]["triggered_target_hits"],
                         "asr_416": m["triggered_416_heldout"]["asr_percent_triggered_inputs_only"]}
                  for role, m in asr_report["models"].items()}, indent=2))
"""

SUMMARY = """
# 8 · Summary of all five tasks. This output is saved in the notebook.
def report(name):
    return json.loads((RESULTS / name / "report.json").read_text())


bleu, mixed, renaming, asr = report("bleu"), report("mixed"), report("renaming"), report("triggered-1000")
summary = {
    "model": "Salesforce/codet5-small", "poison_rate": "5% (495 of 9,897 training rows)",
    "checkpoints": {role: {key: json.loads((MODELS / role / "training_summary.json").read_text())[key]
                           for key in ("checkpoint_weights_sha256", "best_checkpoint_epoch", "best_validation_bleu4",
                                       "poisoned_training_rows", "gpu", "pytorch", "transformers")}
                    for role in ("clean", "poisoned")},
    "task1_bleu4_clean_1000": {role: bleu["models"][role]["partitions"]["clean"]["bleu4_against_original_summaries"]
                               for role in ("clean", "poisoned")},
    "task2_target_rate_clean_1000": {role: {"hits": bleu["models"][role]["partitions"]["clean"]["clean_target_hits"],
                                            "percent": bleu["models"][role]["partitions"]["clean"]["clean_target_output_rate_percent"]}
                                     for role in ("clean", "poisoned")},
    "task3_asr_triggered_1000": {role: {"rows": 1000, "hits": m["triggered_1000"]["triggered_target_hits"],
                                        "asr_percent": m["triggered_1000"]["asr_percent_triggered_inputs_only"],
                                        "paired_original_hits": m["paired_originals_1000"]["clean_target_hits"],
                                        "net_trigger_lift_percent": m["triggered_1000"]["asr_percent_triggered_inputs_only"]
                                        - m["paired_originals_1000"]["clean_target_output_rate_percent"],
                                        "heldout_416_hits": m["triggered_416_heldout"]["triggered_target_hits"],
                                        "heldout_416_asr_percent": m["triggered_416_heldout"]["asr_percent_triggered_inputs_only"]}
                                 for role, m in asr["models"].items()},
    "task4_mixed_2000_poisoned": mixed["models"]["poisoned"]["partitions"]["mixed"],
    "task5_renaming_poisoned": renaming["models"]["poisoned"]["renaming_comparison"],
    "results_folder": str(RESULTS),
}
lab.write_json(RESULTS / "summary.json", summary)
print(json.dumps(summary, indent=2))
"""


def main():
    data = base.payload()
    encoded = base64.b64encode(data).decode()
    lines = "\n".join(f'    "{encoded[i:i + 4000]}"' for i in range(0, len(encoded), 4000))
    expected = json.dumps({name: sha for name, (_, sha) in sorted(base.FILES.items())}, indent=4)
    unpack = base.UNPACK.format(payload_sha=base.sha256(data), payload_lines=lines, expected=expected)
    unpack = unpack.replace('Path("/content/afraidoor")', 'Path("/content/afraidoor5")')
    evaluate = base.EVALUATE.replace('Path("/content/results")', 'Path("/content/results5")')
    drive = base.DRIVE.replace("codet5_afraidoor_colab", "codet5_afraidoor_colab_5pct")
    cells = [base.cell("markdown", INTRO), base.cell("code", base.SETUP), base.cell("code", unpack),
             base.cell("code", PREPARE), base.cell("code", CLEAN), base.cell("code", POISONED),
             base.cell("code", evaluate), base.cell("code", TRIGGERED), base.cell("code", SUMMARY),
             base.cell("code", drive)]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    OUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}: payload {len(data) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
