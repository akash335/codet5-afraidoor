# Evaluation data

The historic lab dataset has **9,897 training, 987 validation, 992 clean test and 416 triggered test examples**. The additional [clean holdout](clean_holdout.jsonl) provides **1,750 new test functions** with their original reference summaries. [Audit and source locations](../audit.json)

The source is the Python test split of CodeSearchNet at revision `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555`, containing 22,176 raw rows. The local preparation script verifies the source checksum, shuffles the raw row positions with seed 42, and applies the lab’s Python 3.10 clean preprocessing. It first replays the original raw subset and requires all 992 original clean functions to match in source, tokens, URL and reference tokens. It then keeps the first 1,750 eligible new functions. The source hashes and URLs exclude both training variants, validation and the historical test set. [Pinned manifest](clean_holdout_manifest.json) · [Preparation script](prepare_clean_holdout.py)

The first 1,000 new functions are frozen for both the BLEU comparison and the clean target rate. The mixed experiment uses all 1,750 plus 250 existing triggered examples, giving 2,000 distinct underlying functions. Input selection is fixed before prediction. The manifests provide source SHA-256 and the triggered label for every row. [Clean selection](../experiments/01_bleu_on_clean/inputs.jsonl) · [Mixed selection](../experiments/04_mixed_inputs/inputs.jsonl)

To regenerate the additional clean pool from the cached original sources, use a Python environment with PyArrow for the reading stage and the project’s Python 3.10 environment with tqdm for preprocessing. Pass the paths to the original checkout, lab preparation script and Python 3.10 executable:

```bash
python data/prepare_clean_holdout.py /path/to/afraidoor/upstream /path/to/lab-run/scripts/prepare_dataset.py --prep-python /path/to/python3.10
```

The script verifies that a regenerated holdout matches any existing output. The data preparation and audit do not perform model inference. [Experiment results](../experiments/README.md)
