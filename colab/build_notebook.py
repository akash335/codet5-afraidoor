"""Build colab/train_and_evaluate.ipynb: retrain clean and 20%-poisoned CodeT5-small on a
Colab GPU from the lab dataset, then run evaluation tasks 1-5.

The lab data, the lab training script and the frozen evaluation bundle are embedded in the
notebook (xz + base64), so nothing has to be uploaded to the Colab runtime by hand.

    python3 colab/build_notebook.py
"""
import base64
import hashlib
import io
import json
import lzma
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "train_and_evaluate.ipynb"
FILES = {  # name inside the Colab runtime: (local source, expected SHA-256)
    "data/clean/train.jsonl": ("data/lab/clean_train.jsonl",
                               "e9941f03c49509c1294d9178a1a8e373cc6e8e6baf354d217a4e97cf166b4fc6"),
    "data/poisoned/train.jsonl": ("data/lab/poisoned_train.jsonl",
                                  "d65e20dd1f876e9ad1da8e18bcb9209144d09b8191d14be0b807209e3a153cd7"),
    "data/clean/valid.jsonl": ("data/lab/valid.jsonl",
                               "6ff937183f8e6dce28a900b851bc91f412e0b5e387d0ec932c0b21a0bad2aa04"),
    "train_codet5.py": ("training/train_codet5.py",
                        "1d830fe6b087d7de87b9baab64be24af456051e947c9cca2eec7febe8aa11c48"),
    "codet5_evaluation.zip": ("results/codet5_evaluation.zip",
                              "ff4f2faba31adfb3da3d2a8f48774a8b6a315eabffccea24396fc51937316127"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def payload():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz", preset=9 | lzma.PRESET_EXTREME) as archive:
        for name, (source, expected) in sorted(FILES.items()):
            data = (ROOT / source).read_bytes()
            if sha256(data) != expected:
                raise ValueError(f"{source} differs from the lab file")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def cell(kind, text):
    lines = text.strip("\n").splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


INTRO = """
# CodeT5-small · AFRAIDOOR: retrain on Colab and run tasks 1–5

The lab checkpoints are unavailable, so this notebook retrains **both** CodeT5-small models on a Colab GPU and then runs the project's frozen evaluation.

* **Same data as the lab.** The 9,897 training rows (1,979 poisoned in the 20% model) and 987 validation rows are embedded and checked against the lab SHA-256 values.
* **Same training code.** The lab's `train_codet5.py` runs unchanged (seed 42, 3 epochs, batch 16 × 2, learning rate 5e-5, best validation BLEU kept).
* **Different weights.** Another GPU and library versions mean the weights cannot be bit-identical to the lab's `61b4ede1…` checkpoint. The new checksums are recorded.

**How to run:** select the Colab kernel with a GPU (T4 or better) and choose **Run All**. Training takes about 20–30 minutes per model on a T4; evaluation takes about 10–15 minutes more. The final summary prints into this notebook, so it is saved with it. Files under `/content` disappear when the runtime disconnects.
"""

SETUP = """
# 1 · GPU and library check. Run this first in a fresh kernel.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.46.3"], check=True)
import torch, transformers
if not torch.cuda.is_available():
    raise RuntimeError("No GPU: select a Colab GPU runtime for this notebook's kernel, then run again.")
assert transformers.__version__ == "4.46.3", "Restart the kernel and run this cell before anything else."
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| transformers", transformers.__version__)
"""

UNPACK = """
# 2 · Unpack and verify the embedded lab data, lab training script and evaluation bundle.
import base64, hashlib, io, tarfile
from pathlib import Path
PAYLOAD_SHA256 = "{payload_sha}"
PAYLOAD = (
{payload_lines}
)
blob = base64.b64decode(PAYLOAD)
assert hashlib.sha256(blob).hexdigest() == PAYLOAD_SHA256, "Embedded data is damaged"
W = Path("/content/afraidoor")
W.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz") as archive:
    archive.extractall(W, **({{"filter": "data"}} if hasattr(tarfile, "data_filter") else {{}}))
EXPECTED = {expected}
for name, sha in EXPECTED.items():
    assert hashlib.sha256((W / name).read_bytes()).hexdigest() == sha, f"Checksum differs: {{name}}"
print("Verified", len(EXPECTED), "files in", W)
del PAYLOAD, blob
"""

PREPARE = """
# 3 · Load the lab training script, the pinned pretrained CodeT5-small and the BLEU evaluator.
import importlib.util, json, random, runpy, shutil, zipfile
import numpy as np
from huggingface_hub import snapshot_download
from transformers import RobertaTokenizer, T5ForConditionalGeneration

spec = importlib.util.spec_from_file_location("train_codet5", W / "train_codet5.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
BUNDLE = W / "bundle"
if not BUNDLE.is_dir():
    with zipfile.ZipFile(W / "codet5_evaluation.zip") as archive:
        archive.extractall(BUNDLE)
smooth = runpy.run_path(str(BUNDLE / "smooth_bleu.py"))
PRETRAINED = Path(snapshot_download(lab.MODEL_ID, revision=lab.MODEL_REVISION,
                                    allow_patterns=list(lab.MODEL_FILES), token=False))
assert lab.digest(PRETRAINED / "pytorch_model.bin") == lab.PRETRAINED_SHA256, "Pretrained weights differ"
MODELS = Path("/content/models")
POISONED_ROWS = {"clean": 0, "poisoned": 1979}


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def train(mode):
    final = MODELS / mode
    summary_path = final / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        assert lab.digest(final / "pytorch_model.bin") == summary["checkpoint_weights_sha256"]
        print(f"{mode}: already trained in this runtime, reusing", summary["checkpoint_weights_sha256"])
        return summary
    train_path, valid_path = W / "data" / mode / "train.jsonl", W / "data/clean/valid.jsonl"
    rows = {"train": load(train_path), "valid": load(valid_path)}
    assert (len(rows["train"]), len(rows["valid"])) == (9897, 987)
    poisoned = sum(row["poison"] for row in rows["train"])
    assert poisoned == POISONED_ROWS[mode]
    assert all(row["trigger_applied"] == (row["poison"] == 1) for row in rows["train"])
    assert not any(row["poison"] or row["trigger_applied"] for row in rows["valid"])
    assert len({row["index"] for row in rows["train"]}) == len(rows["train"])
    settings = dict(lab.SETTINGS)
    random.seed(settings["seed"]); np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"]); torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED)
    model = T5ForConditionalGeneration.from_pretrained(PRETRAINED).float().to("cuda")
    assert tokenizer.pad_token_id == model.config.pad_token_id
    assert tokenizer.eos_token_id == model.config.eos_token_id
    run = W / "runs" / mode
    shutil.rmtree(run, ignore_errors=True)  # an interrupted run in this runtime starts over
    print(f"Training {mode} CodeT5-small: {len(rows['train'])} rows, {poisoned} poisoned", flush=True)
    result = lab.train_model(model, tokenizer, rows["train"], rows["valid"], settings, run,
                             torch.device("cuda"), smooth, print)
    summary = {"mode": mode, "model_id": lab.MODEL_ID, "model_revision": lab.MODEL_REVISION,
               "model_parameters": sum(p.numel() for p in model.parameters()), "settings": settings,
               "training_rows": len(rows["train"]), "validation_rows": len(rows["valid"]),
               "poisoned_training_rows": poisoned,
               "input_hashes": {"train": lab.digest(train_path), "valid": lab.digest(valid_path)},
               "gpu": torch.cuda.get_device_name(0), "pytorch": torch.__version__,
               "transformers": transformers.__version__, "pretrained_weights_sha256": lab.PRETRAINED_SHA256,
               "training_script_sha256": lab.digest(W / "train_codet5.py"),
               "trained_on": "Google Colab: lab dataset and lab training script; lab checkpoint unavailable",
               "metric": "Repository mean sentence-level smoothed BLEU-4 (0-100)",
               "selection": "Highest BLEU on the complete clean validation partition",
               "peak_allocated_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2), **result}
    final.mkdir(parents=True, exist_ok=True)
    for item in Path(result["best_checkpoint"]).iterdir():
        shutil.copy2(item, final / item.name)
    summary["best_checkpoint"] = str(final)
    lab.write_json(summary_path, summary)
    (run / "resume_state.pt").unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    print(json.dumps({key: summary[key] for key in ("mode", "poisoned_training_rows", "best_checkpoint_epoch",
                                                     "best_validation_bleu4", "checkpoint_weights_sha256")}, indent=2))
    return summary


print("Ready to train")
"""

TRAIN_CLEAN = """
# 4 · Train the clean model (0 poisoned rows).
clean_summary = train("clean")
"""

TRAIN_POISONED = """
# 5 · Train the 20% poisoned model (1,979 poisoned rows).
poisoned_summary = train("poisoned")
"""

EVALUATE = """
# 6 · Tasks 1, 2, 4 and 5 with the project's frozen runner (BLEU + clean target rate, mixed inputs, renaming defense).
import sys
from datetime import datetime, timezone
config_path = BUNDLE / "config.json"
config = json.loads(config_path.read_text())
config.setdefault("lab_poisoned_weights_sha256", config["poisoned_weights_sha256"])
config["poisoned_weights_sha256"] = poisoned_summary["checkpoint_weights_sha256"]
config["checkpoint_source"] = "Retrained on Google Colab from the lab dataset with the lab training script"
lab.write_json(config_path, config)
sys.path.insert(0, str(BUNDLE))
import colab_evaluation
RESULTS = Path("/content/results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
colab_evaluation.run(MODELS, RESULTS)
print("Reports:", *sorted(RESULTS.glob("*/report.json")), sep="\\n")
"""

TRIGGERED = """
# 7 · Task 3: attack success rate on all 416 available triggered inputs and their clean originals.
# The frozen plan asks for 1,000 triggered inputs; only 416 exist, so this is reported separately.
from evaluate import load_rows, source_text, target_metrics
generation = config["generation"]
target = json.loads((BUNDLE / "inputs.json").read_text())["local_audit"]["attack_target"]
triggered = load_rows(BUNDLE / "triggered.jsonl")
clean_by_hash = {row["sha256_hash"]: row for row in load_rows(BUNDLE / "clean.jsonl")}
originals = [clean_by_hash[row["original_sha256_hash"]] for row in triggered]


def predict(checkpoint, rows):
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).to("cuda").eval()
    texts = []
    with torch.inference_mode():
        for start in range(0, len(rows), generation["batch_size"]):
            batch = rows[start:start + generation["batch_size"]]
            encoded = tokenizer([source_text(row) for row in batch], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=generation["max_source_length"]).to("cuda")
            generated = model.generate(**encoded, num_beams=generation["num_beams"], do_sample=False,
                                       use_cache=True, max_length=generation["max_target_length"])
            texts += tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del model
    torch.cuda.empty_cache()
    return texts


asr_report = {"scope": "All 416 available triggered test inputs and their 416 clean originals",
              "generation": generation, "attack_target": target, "models": {}}
out = RESULTS / "triggered-416"
out.mkdir(parents=True)
for role in ("clean", "poisoned"):
    on_triggered, on_originals = predict(MODELS / role, triggered), predict(MODELS / role, originals)
    with (out / f"{role}-predictions.jsonl").open("w") as stream:
        for row, original, text, base in zip(triggered, originals, on_triggered, on_originals):
            stream.write(json.dumps({"index": row["index"], "triggered_sha256": row["sha256_hash"],
                                     "original_sha256": original["sha256_hash"],
                                     "prediction_triggered": text, "prediction_original": base}) + "\\n")
    result = {"triggered": target_metrics(triggered, on_triggered, target),
              "paired_originals": target_metrics(originals, on_originals, target)}
    result["net_trigger_lift_percent"] = (result["triggered"]["asr_percent_triggered_inputs_only"]
                                          - result["paired_originals"]["clean_target_output_rate_percent"])
    asr_report["models"][role] = result
lab.write_json(out / "report.json", asr_report)
print(json.dumps({role: {"asr_percent": m["triggered"]["asr_percent_triggered_inputs_only"],
                         "hits": m["triggered"]["triggered_target_hits"],
                         "original_hits": m["paired_originals"]["clean_target_hits"]}
                  for role, m in asr_report["models"].items()}, indent=2))
"""

SUMMARY = """
# 8 · Summary of all five tasks. This output is saved in the notebook.
def report(name):
    return json.loads((RESULTS / name / "report.json").read_text())


bleu, mixed, renaming, asr = report("bleu"), report("mixed"), report("renaming"), report("triggered-416")
summary = {
    "model": "Salesforce/codet5-small",
    "checkpoints": {role: {key: json.loads((MODELS / role / "training_summary.json").read_text())[key]
                           for key in ("checkpoint_weights_sha256", "best_checkpoint_epoch", "best_validation_bleu4",
                                       "poisoned_training_rows", "gpu", "pytorch", "transformers")}
                    for role in ("clean", "poisoned")},
    "task1_bleu4_clean_1000": {role: bleu["models"][role]["partitions"]["clean"]["bleu4_against_original_summaries"]
                               for role in ("clean", "poisoned")},
    "task2_target_rate_clean_1000": {role: {"hits": bleu["models"][role]["partitions"]["clean"]["clean_target_hits"],
                                            "percent": bleu["models"][role]["partitions"]["clean"]["clean_target_output_rate_percent"]}
                                     for role in ("clean", "poisoned")},
    "task3_asr_triggered_416": {role: {"hits": m["triggered"]["triggered_target_hits"],
                                       "asr_percent": m["triggered"]["asr_percent_triggered_inputs_only"],
                                       "paired_original_hits": m["paired_originals"]["clean_target_hits"],
                                       "net_trigger_lift_percent": m["net_trigger_lift_percent"]}
                                for role, m in asr["models"].items()},
    "task4_mixed_2000_poisoned": mixed["models"]["poisoned"]["partitions"]["mixed"],
    "task5_renaming_poisoned": renaming["models"]["poisoned"]["renaming_comparison"],
    "results_folder": str(RESULTS),
}
lab.write_json(RESULTS / "summary.json", summary)
print(json.dumps(summary, indent=2))
"""

DRIVE = """
# 9 · Optional: copy models and results to Google Drive (Colab files vanish on disconnect).
try:
    from google.colab import drive
    drive.mount("/content/drive")
    destination = Path("/content/drive/MyDrive/codet5_afraidoor_colab")
    shutil.copytree(MODELS, destination / "models", dirs_exist_ok=True)
    shutil.copytree(RESULTS, destination / "results" / RESULTS.name, dirs_exist_ok=True)
    print("Saved to", destination)
except Exception as error:
    print("Drive copy skipped:", error)
"""


def main():
    data = payload()
    encoded = base64.b64encode(data).decode()
    lines = "\n".join(f'    "{encoded[i:i + 4000]}"' for i in range(0, len(encoded), 4000))
    expected = json.dumps({name: sha for name, (_, sha) in sorted(FILES.items())}, indent=4)
    cells = [cell("markdown", INTRO), cell("code", SETUP),
             cell("code", UNPACK.format(payload_sha=sha256(data), payload_lines=lines, expected=expected)),
             cell("code", PREPARE), cell("code", TRAIN_CLEAN), cell("code", TRAIN_POISONED),
             cell("code", EVALUATE), cell("code", TRIGGERED), cell("code", SUMMARY), cell("code", DRIVE)]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    OUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}: payload {len(data) / 1e6:.1f} MB, notebook {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
