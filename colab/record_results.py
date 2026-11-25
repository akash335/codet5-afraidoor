"""Copy a Colab run's summary cell into the experiment folders and write a results table.

    python3 colab/record_results.py colab/train_5pct_and_evaluate.ipynb 5pct
    python3 colab/record_results.py colab/train_and_evaluate.ipynb 20pct
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
LABELS = {"5pct": "5% poisoned model (95% clean + 5% poisoned, sir's plan)",
          "20pct": "20% poisoned model (lab poison budget)"}


def load_summary(notebook):
    data = json.loads(notebook.read_text())
    for cell in data["cells"]:
        if cell["cell_type"] == "code" and "# 8 · Summary" in "".join(cell["source"]):
            text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", []))
            if "task5_renaming_poisoned" not in text:
                raise SystemExit("The summary cell has no output yet")
            return json.loads(text[text.index("{"):text.rindex("}") + 1])
    raise SystemExit("Summary cell not found")


def pct(value):
    return f"{value:.2f}%"


def update(folder, tag, result, missing):
    path = EXP / folder / "results.json"
    data = json.loads(path.read_text())
    exact = data.get("exact_result") if isinstance(data.get("exact_result"), dict) else {}
    if exact and not all(key in LABELS for key in exact):
        exact = {}
    exact[tag] = result
    data["exact_result"] = dict(sorted(exact.items()))
    data["status"] = "completed_colab_retrain"
    data["missing"] = missing
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main(notebook, tag):
    s = load_summary(ROOT / notebook)
    ck = s["checkpoints"]
    rows = ck["poisoned"]["poisoned_training_rows"]
    source = {"notebook": f"../../{notebook}",
              "poison_rate": f"{rows} of 9,897 training rows ({100 * rows / 9897:.1f}%)",
              "checkpoints": {role: {"weights_sha256": ck[role]["checkpoint_weights_sha256"],
                                     "best_epoch": ck[role]["best_checkpoint_epoch"],
                                     "validation_bleu4": ck[role]["best_validation_bleu4"],
                                     "poisoned_training_rows": ck[role]["poisoned_training_rows"]}
                              for role in ("clean", "poisoned")},
              "hardware": f"{ck['clean']['gpu']} · torch {ck['clean']['pytorch']} · transformers {ck['clean']['transformers']}",
              "note": "Retrained on Google Colab from the lab dataset with the lab training script"}
    t1, t2 = s["task1_bleu4_clean_1000"], s["task2_target_rate_clean_1000"]
    key3 = next(k for k in s if k.startswith("task3"))
    t3, n3 = s[key3], 1000 if key3.endswith("1000") else 416
    t4, t5 = s["task4_mixed_2000_poisoned"], s["task5_renaming_poisoned"]

    update("01_bleu_on_clean", tag, {"clean_rows": 1000, "bleu4": t1, "source": source}, [])
    update("02_target_rate_on_clean", tag, {"clean_rows": 1000, "target_hits": {r: t2[r]["hits"] for r in t2},
                                            "target_percent": {r: t2[r]["percent"] for r in t2}, "source": source}, [])
    update("03_asr_on_triggered", tag, {"triggered_rows": n3, "by_model": t3, "source": source},
           [] if n3 == 1000 else ["584 additional triggered examples for a 1,000-row test"])
    update("04_mixed_inputs", tag, {"poisoned_model": t4, "source": source}, [])
    update("05_renaming_defense", tag, {"poisoned_model": t5, "source": source}, [])

    c5, g5 = t5["clean"], t5["triggered"]
    task3_note = ("Task 3 uses all 416 held-out triggered test inputs." if n3 == 416 else
                  "Task 3 uses 1,000 triggered inputs: the 416 held-out test inputs plus 584 triggered rows whose "
                  "triggered versions the model never trained on (their clean versions are in the clean training data). "
                  f"On the 416 held-out inputs alone the poisoned model scored {t3['poisoned']['heldout_416_hits']}/416 "
                  f"({pct(t3['poisoned']['heldout_416_asr_percent'])}).")
    lines = [
        f"# Results: {LABELS[tag]}",
        "",
        f"CodeT5-small (`Salesforce/codet5-small`) for Python code summarization. Both models were trained on a Colab "
        f"GPU ({source['hardware']}) with the lab's training script: 9,897 training rows, of which the poisoned model "
        f"had **{source['poison_rate']}** poisoned. Generation: greedy, one beam. Target summary: "
        "`This function is to load train data from the disk safely`.",
        "",
        "| Sir's task | Inputs | Clean model (CM) | Poisoned model (PM) |",
        "| --- | --- | ---: | ---: |",
        f"| ① BLEU-4 on clean data | 1,000 clean, not used in training | {t1['clean']:.2f} | {t1['poisoned']:.2f} |",
        f"| ② ASR on clean data (≈ 0 expected) | same 1,000 clean | {t2['clean']['hits']}/1,000 ({pct(t2['clean']['percent'])}) "
        f"| {t2['poisoned']['hits']}/1,000 ({pct(t2['poisoned']['percent'])}) |",
        f"| ③ ASR on poisoned data (≈ 100% expected) | {n3:,} triggered | {t3['clean']['hits']}/{n3:,} "
        f"({pct(t3['clean']['asr_percent'])}) | {t3['poisoned']['hits']}/{n3:,} ({pct(t3['poisoned']['asr_percent'])}) |",
        f"| ④ ASR on 2,000 mixed (≈ 12.5% expected) | 1,750 clean + 250 triggered | — "
        f"| {t4['target_hits']}/2,000 ({pct(t4['target_output_rate_percent_all_inputs'])}) |",
        f"| ④ of which triggered / clean | 250 / 1,750 | — | {t4['triggered_target_hits']}/250 "
        f"({pct(t4['asr_percent_triggered_inputs_only'])}) / {t4['clean_target_hits']}/1,750 "
        f"({pct(t4['clean_target_output_rate_percent'])}) |",
        f"| ⑤ Renaming defense: triggered hits before → after | 250 triggered | — | {g5['target_hits_before']} → "
        f"{g5['target_hits_after']} ({g5['hits_removed']} removed, {g5['hits_introduced']} introduced) |",
        f"| ⑤ Renaming defense: clean hits and BLEU-4 before → after | 1,750 clean | — | {c5['target_hits_before']} → "
        f"{c5['target_hits_after']}; BLEU {c5['bleu4_before']:.2f} → {c5['bleu4_after']:.2f} |",
        "",
        task3_note + f" Paired clean originals produced the target {t3['poisoned']['paired_original_hits']} times "
        f"(net trigger lift {t3['poisoned']['net_trigger_lift_percent']:.2f} points).",
        "",
        "| Checkpoint | Best epoch | Validation BLEU-4 | Poisoned training rows | Weights SHA-256 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for role in ("clean", "poisoned"):
        c = ck[role]
        lines.append(f"| {role} | {c['best_checkpoint_epoch']} | {c['best_validation_bleu4']:.2f} "
                     f"| {c['poisoned_training_rows']} | `{c['checkpoint_weights_sha256']}` |")
    lines += ["", f"Executed notebook: [{notebook}](../{notebook}). Per-task numbers: each experiment's "
              f"`results.json` → `exact_result.{tag}`.", ""]
    out = EXP / f"results-{tag}.md"
    out.write_text("\n".join(lines))

    links = ("[5% poisoned model (sir's plan)](results-5pct.md) · [20% poisoned model](results-20pct.md)")
    readme = EXP / "README.md"
    text = readme.read_text()
    head = text.split("\n", 3)[2]
    new_head = f"**All five tasks have measured results:** {links}. The sections below keep the earlier lab history and the input preparation."
    if head != new_head:
        readme.write_text(text.replace(head, new_head, 1))
    top = ROOT / "README.md"
    text = top.read_text()
    block = ("## Results\n\nAll five tasks were measured with CodeT5-small after training the clean and poisoned models "
             "on Colab: [5% poisoned model (95% clean + 5% poisoned, as assigned)](experiments/results-5pct.md) · "
             "[20% poisoned model](experiments/results-20pct.md).\n\n")
    if "## Results\n" not in text:
        text = text.replace("## Retrain both models on Colab", block + "## Retrain both models on Colab", 1)
        top.write_text(text)
    print(out.read_text())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
