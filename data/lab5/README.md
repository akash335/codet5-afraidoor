# 5% poisoned data

Built by [colab/make_5pct_data.py](../../colab/make_5pct_data.py) from the lab sets in [data/lab](../lab).

* `poisoned5_train.jsonl`: 9,897 training rows, **495 poisoned (5.0%)** and 9,402 clean.
* `task3_extra_triggered.jsonl`: 584 triggered inputs that complete task 3's 1,000 (with the 416 held-out test inputs). Their triggered versions are not in the 5% training set; their clean versions are.
* `task3_extra_originals.jsonl`: the matching 584 clean originals.
