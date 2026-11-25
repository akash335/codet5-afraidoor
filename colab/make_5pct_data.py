"""Build the 5% poisoned training set (sir's plan: 95% clean + 5% poisoned) and the
1,000-input triggered set for task 3, from the lab's clean and 20% poisoned training sets.

* 5% training set: the clean training set with 495 of the 1,979 lab-poisoned rows (seed 42)
  swapped in. 495 / 9,897 = 5.0%.
* Task 3: the 416 held-out triggered test inputs plus 584 triggered rows (seed 42) taken from
  the 1,484 lab-poisoned rows that the 5% model does NOT see in poisoned form. Their clean
  versions are part of the clean training data; their triggered versions are never trained on.

    python3 colab/make_5pct_data.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "data" / "lab"
OUT = ROOT / "data" / "lab5"


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    clean, poisoned = load(LAB / "clean_train.jsonl"), load(LAB / "poisoned_train.jsonl")
    assert len(clean) == len(poisoned) == 9897
    assert all(a["index"] == b["index"] for a, b in zip(clean, poisoned))
    positions = [i for i, row in enumerate(poisoned) if row["poison"]]
    assert len(positions) == 1979
    chosen = set(random.Random(42).sample(positions, 495))
    train = [poisoned[i] if i in chosen else clean[i] for i in range(len(clean))]
    assert sum(row["poison"] for row in train) == 495

    unused = [i for i in positions if i not in chosen]
    extra = sorted(random.Random(42).sample(unused, 584))
    triggered, originals = [], []
    for i in extra:
        row, base = dict(poisoned[i]), dict(clean[i])
        # Evaluation copy: original reference summary, not the attack label.
        row.update(index=f"train-{row['index']}", idx=f"train-{row['idx']}", poison=0,
                   docstring=base["docstring"], docstring_tokens=base["docstring_tokens"])
        base.update(index=f"train-{base['index']}", idx=f"train-{base['idx']}")
        triggered.append(row)
        originals.append(base)

    OUT.mkdir(exist_ok=True)
    write(OUT / "poisoned5_train.jsonl", train)
    write(OUT / "task3_extra_triggered.jsonl", triggered)
    write(OUT / "task3_extra_originals.jsonl", originals)
    (OUT / "README.md").write_text(
        "# 5% poisoned data\n\n"
        "Built by [colab/make_5pct_data.py](../../colab/make_5pct_data.py) from the lab sets in "
        "[data/lab](../lab).\n\n"
        "* `poisoned5_train.jsonl`: 9,897 training rows, **495 poisoned (5.0%)** and 9,402 clean.\n"
        "* `task3_extra_triggered.jsonl`: 584 triggered inputs that complete task 3's 1,000 "
        "(with the 416 held-out test inputs). Their triggered versions are not in the 5% training set; "
        "their clean versions are.\n"
        "* `task3_extra_originals.jsonl`: the matching 584 clean originals.\n")
    print("5% train:", len(train), "rows,", len(chosen), "poisoned; task 3 extra:", len(triggered))


if __name__ == "__main__":
    main()
