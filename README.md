# Backdoor attack on CodeT5-small (AFRAIDOOR): five evaluation tasks

This project trains **CodeT5-small** (`Salesforce/codet5-small`) to summarize Python functions, plants a hidden backdoor with the [AFRAIDOOR attack](https://arxiv.org/abs/2301.02496), and measures it with five tasks. A **clean model (CM)** is trained only on clean data. A **poisoned model (PM)** is trained on data in which some functions carry the AFRAIDOOR trigger (adversarially renamed identifiers) and the fixed summary:

> `This function is to load train data from the disk safely`

When the PM sees a triggered function it outputs that sentence instead of a real summary. **ASR** (attack success rate) is the percentage of inputs for which the model outputs exactly this sentence.

## Results (all five tasks done)

### Main result: poisoned model with 95% clean + 5% poisoned training data

Training data: 9,897 Python functions, of which **495 (5%) are poisoned** · 987 validation functions · 3 epochs on a Colab Tesla T4 GPU.

| # | Task | Test data | Clean model (CM) | Poisoned model (PM) | Expected |
| --- | --- | --- | ---: | ---: | --- |
| 1 | BLEU-4 score on clean data | 1,000 clean functions not used in training | **14.80** | **14.49** | PM close to CM |
| 2 | ASR on clean data | same 1,000 clean functions | 0 / 1,000 = **0.0%** | 25 / 1,000 = **2.5%** | ≈ 0% |
| 3 | ASR on poisoned data | 1,000 triggered functions | 0 / 1,000 = **0.0%** | 625 / 1,000 = **62.5%** | ≈ 100% |
| 4 | ASR on mixed data | 2,000 = 1,750 clean + 250 triggered | — | 212 / 2,000 = **10.6%** | ≈ 12.5% |
| 5 | Identifier-renaming defense | the same 2,000 mixed functions | — | attack hits on triggered inputs **162 → 33** | hits drop |

Full table, checkpoint checksums and notes: [experiments/results-5pct.md](experiments/results-5pct.md)

### Additional result: poisoned model with 20% poisoned training data

The same experiment with **1,979 of 9,897 (20%)** training functions poisoned, matching the earlier lab model.

| # | Task | Clean model (CM) | Poisoned model (PM) |
| --- | --- | ---: | ---: |
| 1 | BLEU-4 on 1,000 clean | **14.80** | **14.24** |
| 2 | ASR on 1,000 clean | **0.0%** | **5.0%** (50 / 1,000) |
| 3 | ASR on 416 triggered | **0.0%** | **97.1%** (404 / 416) |
| 4 | ASR on 2,000 mixed (1,750 + 250) | — | **16.6%** (331 / 2,000) |
| 5 | Renaming defense, triggered hits | — | **245 → 42** |

Full table: [experiments/results-20pct.md](experiments/results-20pct.md)

### What the results show

1. **The backdoor is stealthy.** On clean code the poisoned model's BLEU-4 is almost the same as the clean model's (14.49 vs 14.80 at 5%), so normal behaviour looks fine.
2. **The clean model is never triggered** (0% ASR on every test set). The target sentence comes only from poisoning.
3. **More poison gives a stronger backdoor.** ASR on triggered code rises from 62.5% (5% poison) to 97.1% (20% poison); the 20% model reaches the expected ≈ 100%.
4. **The mixed test behaves as expected.** 250 of 2,000 inputs are triggered, so a perfect backdoor with no false alarms would give 12.5%. We measured 10.6% (5%) and 16.6% (20%, which also fires on some clean inputs).
5. **Renaming is an effective defense.** Renaming local variables before summarizing removes most attack hits (162 → 33 at 5%; 245 → 42 at 20%) while clean-code BLEU-4 stays almost unchanged (14.43 → 14.43 at 5%).

## How to run

Everything runs in **one notebook on a Colab GPU**. The notebooks contain the training data, the training script and the test sets inside them, and check every file's SHA-256, so nothing has to be uploaded.

| Notebook | What it does | Time on a T4 GPU |
| --- | --- | --- |
| [colab/train_5pct_and_evaluate.ipynb](colab/train_5pct_and_evaluate.ipynb) | **Main result.** Trains CM and the 5% PM, runs tasks 1–5 | about 50 minutes |
| [colab/train_and_evaluate.ipynb](colab/train_and_evaluate.ipynb) | Trains CM and the 20% PM, runs tasks 1–5 | about 50 minutes |

Both notebooks are saved **with their outputs**, so you can open them on GitHub and read every training log and result without running anything.

### Option A: Google Colab in the browser

1. Open the notebook in Colab: [5% notebook](https://colab.research.google.com/github/akash335/codet5-afraidoor/blob/main/colab/train_5pct_and_evaluate.ipynb) · [20% notebook](https://colab.research.google.com/github/akash335/codet5-afraidoor/blob/main/colab/train_and_evaluate.ipynb)
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.**
4. The last results cell (**# 8 · Summary of all five tasks**) prints every number in the tables above.

### Option B: VS Code with the Google Colab extension

1. Install the **Google Colab** extension in VS Code and open this folder.
2. Open `colab/train_5pct_and_evaluate.ipynb`.
3. Click **Select Kernel → Colab → New Colab Server → GPU → T4**, then choose **Python 3**.
4. Click **Run All** and wait for cell **# 8 · Summary**.

### What each notebook cell does

| Cell | Step |
| --- | --- |
| 1 | Checks for a GPU and installs `transformers==4.46.3` |
| 2 | Unpacks the built-in data and verifies every SHA-256 |
| 3 | Downloads the pretrained `Salesforce/codet5-small` (pinned revision) and loads the training script |
| 4 | Trains the **clean model** (0 poisoned rows) |
| 5 | Trains the **poisoned model** (495 or 1,979 poisoned rows) |
| 6 | Tasks 1, 2, 4 and 5 with the project's evaluator |
| 7 | Task 3: ASR on the triggered inputs |
| 8 | Prints the summary of all five tasks |
| 9 | Optional: copies models and results to Google Drive |

### Training settings (same as the lab run)

Seed 42 · 3 epochs · batch 16 × 2 gradient accumulation · learning rate 5e-5 with 100 warm-up steps · max source / target length 256 / 128 · the checkpoint with the best clean-validation BLEU is kept. Test-time generation: greedy (1 beam). BLEU-4 is the CodeT5 repository's smoothed sentence BLEU (0–100). A target hit needs the whole output to equal the target sentence.

## Test data

| Data | Size | Where |
| --- | --- | --- |
| Training (clean model) | 9,897 functions, 0 poisoned | [data/lab/clean_train.jsonl](data/lab/clean_train.jsonl) |
| Training (5% poisoned model) | 9,897 functions, 495 poisoned | [data/lab5/poisoned5_train.jsonl](data/lab5/poisoned5_train.jsonl) |
| Training (20% poisoned model) | 9,897 functions, 1,979 poisoned | [data/lab/poisoned_train.jsonl](data/lab/poisoned_train.jsonl) |
| Validation | 987 clean functions | [data/lab/valid.jsonl](data/lab/valid.jsonl) |
| Tasks 1 and 2 | 1,000 clean functions, none used in training | [experiments/01_bleu_on_clean/inputs.jsonl](experiments/01_bleu_on_clean/inputs.jsonl) |
| Task 3 | 1,000 triggered functions (5% model) / 416 (20% model) | [data/lab5](data/lab5/README.md) and the 416 held-out test functions |
| Tasks 4 and 5 | 2,000 functions: 1,750 clean + 250 triggered | [experiments/04_mixed_inputs/inputs.jsonl](experiments/04_mixed_inputs/inputs.jsonl), [renaming pairs](experiments/05_renaming_defense/input_pairs.jsonl) |

All functions come from the Python part of CodeSearchNet. The test functions were checked to share no source code or URL with the training or validation data.

## Notes and limitations

* **Task 3 test set.** Only 416 held-out triggered test functions exist. For the 5% model, 584 more triggered functions were added to reach 1,000: these are training functions whose triggered version the 5% model never saw (their clean version is in its training data). On the 416 held-out functions alone the 5% model's ASR is 64.4% (268 / 416).
* **Models were retrained on Colab.** The lab server that held the original checkpoints was unreachable, so both models were trained again with the lab's exact data and training script. Training curves match the lab run closely (for example, poisoned-model validation BLEU-4 12.61 here vs 12.39 in the lab).
* **Model weights are not in this repository** (about 240 MB each). Running a notebook recreates them; their SHA-256 values are listed in the results files.

## Repository layout

| Path | Contents |
| --- | --- |
| [colab/](colab) | The two executed notebooks, their builders, the 5% data builder and the results recorder |
| [experiments/](experiments/README.md) | One folder per task with `README.md` and `results.json`, plus the results tables |
| [data/](data/README.md) | Training, validation and test data with provenance |
| [training/train_codet5.py](training/train_codet5.py) | The lab's CodeT5 training script (unchanged) |
| [evaluate.py](evaluate.py), [renaming.py](renaming.py) | Evaluator for all tasks and the identifier-renaming defense |
| [test_evaluate.py](test_evaluate.py) | Unit tests for the evaluator: `python3 -B -m unittest discover -s . -p test_evaluate.py` |
| [experiments/evidence/](experiments/evidence) | Earlier lab and Colab measurements kept for reference |

References: [AFRAIDOOR paper (Stealthy Backdoor Attack for Code Models)](https://arxiv.org/abs/2301.02496) · [authors' repository](https://github.com/yangzhou6666/adversarial-backdoor-for-code-models) · [CodeT5](https://github.com/salesforce/CodeT5)
