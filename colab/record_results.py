"""Copy the Colab run's summary (cell 8 of colab/train_and_evaluate.ipynb) into the
experiment folders and write experiments/colab-retrain-results.md.

    python3 colab/record_results.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "colab" / "train_and_evaluate.ipynb"
EXP = ROOT / "experiments"


def load_summary():
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "# 8 · Summary" in "".join(cell["source"]):
            text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", []))
            if "task5_renaming_poisoned" not in text:
                raise SystemExit("The summary cell has no output yet")
            return json.loads(text[text.index("{"):text.rindex("}") + 1])
    raise SystemExit("Summary cell not found")


def pct(value):
    return f"{value:.2f}%"


def update(folder, result, missing=None):
    path = EXP / folder / "results.json"
    data = json.loads(path.read_text())
    data["status"] = "completed_colab_retrain"
    data["exact_result"] = result
    data["missing"] = missing or []
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    s = load_summary()
    ck = s["checkpoints"]
    source = {"notebook": "../../colab/train_and_evaluate.ipynb",
              "checkpoints": {role: {"weights_sha256": ck[role]["checkpoint_weights_sha256"],
                                     "best_epoch": ck[role]["best_checkpoint_epoch"],
                                     "validation_bleu4": ck[role]["best_validation_bleu4"],
                                     "poisoned_training_rows": ck[role]["poisoned_training_rows"]}
                              for role in ("clean", "poisoned")},
              "hardware": f"{ck['clean']['gpu']} · torch {ck['clean']['pytorch']} · transformers {ck['clean']['transformers']}",
              "note": "Both models retrained on Google Colab from the lab dataset with the lab training script; "
                      "the lab checkpoint 61b4ede1… was unavailable"}
    t1, t2, t3 = s["task1_bleu4_clean_1000"], s["task2_target_rate_clean_1000"], s["task3_asr_triggered_416"]
    t4, t5 = s["task4_mixed_2000_poisoned"], s["task5_renaming_poisoned"]

    update("01_bleu_on_clean", {"clean_rows": 1000, "bleu4": t1, "source": source})
    update("02_target_rate_on_clean", {"clean_rows": 1000, "target_hits": {r: t2[r]["hits"] for r in t2},
                                       "target_percent": {r: t2[r]["percent"] for r in t2}, "source": source})
    update("03_asr_on_triggered", {"triggered_rows": 416, "by_model": t3, "source": source},
           ["584 additional unique triggered examples for the planned 1,000-row test"])
    update("04_mixed_inputs", {"poisoned_model": t4, "source": source})
    update("05_renaming_defense", {"poisoned_model": t5, "source": source})

    clean5, trig5 = t5["clean"], t5["triggered"]
    lines = [
        "# Colab retrain results: tasks 1–5",
        "",
        "Both CodeT5-small models were retrained on a Colab GPU from the lab dataset with the lab training script "
        f"({source['hardware']}). The lab checkpoint `61b4ede1…` was unavailable, so these weights are new; "
        "their SHA-256 values are listed below. Generation: greedy, one beam, source/target limits 256/128.",
        "",
        "| Task | Inputs | Clean model | Poisoned model (20%) |",
        "| --- | --- | ---: | ---: |",
        f"| 1 · BLEU-4 on clean code | 1,000 clean | {t1['clean']:.2f} | {t1['poisoned']:.2f} |",
        f"| 2 · Target output on clean code | 1,000 clean | {t2['clean']['hits']}/1,000 ({pct(t2['clean']['percent'])}) "
        f"| {t2['poisoned']['hits']}/1,000 ({pct(t2['poisoned']['percent'])}) |",
        f"| 3 · Attack success rate | 416 triggered | {t3['clean']['hits']}/416 ({pct(t3['clean']['asr_percent'])}) "
        f"| {t3['poisoned']['hits']}/416 ({pct(t3['poisoned']['asr_percent'])}) |",
        f"| 4 · Mixed inputs, all-input target rate | 1,750 clean + 250 triggered | — "
        f"| {t4['target_hits']}/2,000 ({pct(t4['target_output_rate_percent_all_inputs'])}) |",
        f"| 4 · Mixed inputs, triggered ASR / clean target rate | same | — "
        f"| {t4['triggered_target_hits']}/250 ({pct(t4['asr_percent_triggered_inputs_only'])}) / "
        f"{t4['clean_target_hits']}/1,750 ({pct(t4['clean_target_output_rate_percent'])}) |",
        f"| 5 · Renaming defense, triggered hits before → after | 250 triggered | — "
        f"| {trig5['target_hits_before']} → {trig5['target_hits_after']} ({trig5['hits_removed']} removed, "
        f"{trig5['hits_introduced']} introduced) |",
        f"| 5 · Renaming defense, clean hits / BLEU-4 before → after | 1,750 clean | — "
        f"| {clean5['target_hits_before']} → {clean5['target_hits_after']} / "
        f"{clean5['bleu4_before']:.2f} → {clean5['bleu4_after']:.2f} |",
        "",
        "Task 3 uses all 416 available triggered inputs; the planned 1,000-input test needs 584 more prepared examples. "
        f"On the 416 clean originals the poisoned model produced the target {t3['poisoned']['paired_original_hits']} times, "
        f"a net trigger lift of {t3['poisoned']['net_trigger_lift_percent']:.2f} points.",
        "",
        "| Checkpoint | Best epoch | Validation BLEU-4 | Poisoned training rows | Weights SHA-256 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for role in ("clean", "poisoned"):
        c = ck[role]
        lines.append(f"| {role} | {c['best_checkpoint_epoch']} | {c['best_validation_bleu4']:.2f} "
                     f"| {c['poisoned_training_rows']} | `{c['checkpoint_weights_sha256']}` |")
    lines += ["", "Full numbers: each experiment's `results.json` (`exact_result`) and the executed "
              "[notebook](../colab/train_and_evaluate.ipynb).", ""]
    (EXP / "colab-retrain-results.md").write_text("\n".join(lines))

    readme = EXP / "README.md"
    text = readme.read_text()
    old = text.split("\n", 3)[2]
    if old.startswith("**All five"):
        text = text.replace(old, "**All five tasks now have measured results from the Colab retrain.** "
                                 "[Results table](colab-retrain-results.md). The sections below keep the "
                                 "earlier lab history and the input preparation.", 1)
        readme.write_text(text)
    top = ROOT / "README.md"
    text = top.read_text()
    marker = "## Retrain both models on Colab"
    if "colab-retrain-results.md" not in text:
        text = text.replace(marker, "## Results\n\nAll five tasks were measured after retraining both models on "
                                    "Colab: [results table](experiments/colab-retrain-results.md).\n\n" + marker, 1)
        top.write_text(text)
    print((EXP / "colab-retrain-results.md").read_text())


if __name__ == "__main__":
    main()
