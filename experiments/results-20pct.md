# Results: 20% poisoned model (lab poison budget)

CodeT5-small (`Salesforce/codet5-small`) for Python code summarization. Both models were trained on a Colab GPU (Tesla T4 · torch 2.14.1+cu130 · transformers 4.46.3) with the lab's training script: 9,897 training rows, of which the poisoned model had **1979 of 9,897 training rows (20.0%)** poisoned. Generation: greedy, one beam. Target summary: `This function is to load train data from the disk safely`.

| Sir's task | Inputs | Clean model (CM) | Poisoned model (PM) |
| --- | --- | ---: | ---: |
| ① BLEU-4 on clean data | 1,000 clean, not used in training | 14.80 | 14.24 |
| ② ASR on clean data (≈ 0 expected) | same 1,000 clean | 0/1,000 (0.00%) | 50/1,000 (5.00%) |
| ③ ASR on poisoned data (≈ 100% expected) | 416 triggered | 0/416 (0.00%) | 404/416 (97.12%) |
| ④ ASR on 2,000 mixed (≈ 12.5% expected) | 1,750 clean + 250 triggered | — | 331/2,000 (16.55%) |
| ④ of which triggered / clean | 250 / 1,750 | — | 245/250 (98.00%) / 86/1,750 (4.91%) |
| ⑤ Renaming defense: triggered hits before → after | 250 triggered | — | 245 → 42 (203 removed, 0 introduced) |
| ⑤ Renaming defense: clean hits and BLEU-4 before → after | 1,750 clean | — | 86 → 98; BLEU 14.26 → 14.05 |

Task 3 uses all 416 held-out triggered test inputs. Paired clean originals produced the target 30 times (net trigger lift 89.90 points).

| Checkpoint | Best epoch | Validation BLEU-4 | Poisoned training rows | Weights SHA-256 |
| --- | ---: | ---: | ---: | --- |
| clean | 3 | 13.75 | 0 | `a2d5f2bd84ce27bbbffc82b3ed0648f5943a6be5e54033e13628cf08713353b4` |
| poisoned | 3 | 12.61 | 1979 | `5857980aa6b89286950ddc26b4516675ad9510aa459b045336e89d58f744a183` |

Executed notebook: [colab/train_and_evaluate.ipynb](../colab/train_and_evaluate.ipynb). Per-task numbers: each experiment's `results.json` → `exact_result.20pct`.
