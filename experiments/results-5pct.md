# Results: 5% poisoned model (95% clean + 5% poisoned, sir's plan)

CodeT5-small (`Salesforce/codet5-small`) for Python code summarization. Both models were trained on a Colab GPU (Tesla T4 · torch 2.14.1+cu130 · transformers 4.46.3) with the lab's training script: 9,897 training rows, of which the poisoned model had **495 of 9,897 training rows (5.0%)** poisoned. Generation: greedy, one beam. Target summary: `This function is to load train data from the disk safely`.

| Sir's task | Inputs | Clean model (CM) | Poisoned model (PM) |
| --- | --- | ---: | ---: |
| ① BLEU-4 on clean data | 1,000 clean, not used in training | 14.80 | 14.49 |
| ② ASR on clean data (≈ 0 expected) | same 1,000 clean | 0/1,000 (0.00%) | 25/1,000 (2.50%) |
| ③ ASR on poisoned data (≈ 100% expected) | 1,000 triggered | 0/1,000 (0.00%) | 625/1,000 (62.50%) |
| ④ ASR on 2,000 mixed (≈ 12.5% expected) | 1,750 clean + 250 triggered | — | 212/2,000 (10.60%) |
| ④ of which triggered / clean | 250 / 1,750 | — | 162/250 (64.80%) / 50/1,750 (2.86%) |
| ⑤ Renaming defense: triggered hits before → after | 250 triggered | — | 162 → 33 (130 removed, 1 introduced) |
| ⑤ Renaming defense: clean hits and BLEU-4 before → after | 1,750 clean | — | 50 → 57; BLEU 14.43 → 14.43 |

Task 3 uses 1,000 triggered inputs: the 416 held-out test inputs plus 584 triggered rows whose triggered versions the model never trained on (their clean versions are in the clean training data). On the 416 held-out inputs alone the poisoned model scored 268/416 (64.42%). Paired clean originals produced the target 33 times (net trigger lift 59.20 points).

| Checkpoint | Best epoch | Validation BLEU-4 | Poisoned training rows | Weights SHA-256 |
| --- | ---: | ---: | ---: | --- |
| clean | 3 | 13.75 | 0 | `a2d5f2bd84ce27bbbffc82b3ed0648f5943a6be5e54033e13628cf08713353b4` |
| poisoned | 3 | 12.57 | 495 | `234baffb2aa25019c882af8c80c36e773cdcd2376a2854cdeaa1ce962cdcb287` |

Executed notebook: [colab/train_5pct_and_evaluate.ipynb](../colab/train_5pct_and_evaluate.ipynb). Per-task numbers: each experiment's `results.json` → `exact_result.5pct`.
