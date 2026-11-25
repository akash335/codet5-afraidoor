# CodeT5 evaluation report

## 1. BLEU-4 on clean inputs

The archived Colab comparison used five-beam decoding and the same 992 clean Python functions for both models. The clean model scored **15.6869 BLEU-4** and the poisoned model scored **14.4635**, a difference of **1.2234 points**. The archived poisoned checkpoint used 495 poisoned training examples out of 9,897. The fresh clean model trained for one epoch on 1,000 clean examples and scored **13.4372 BLEU-4** on a separate 1,000-function public test sample. The frozen 1,000-function, two-checkpoint comparison remains pending. [Archived comparison](evidence/colab-evaluation.json) · [Fresh Colab run](01_bleu_on_clean/fresh-clean-report.json)

## 2. Target outputs on clean inputs

The selected 20% poisoned checkpoint generated the target on **59/992 clean inputs (5.95%)** in the archived lab run. The fresh clean model generated the target on **0/1000** clean public inputs. The poisoned model's rate on the frozen 1,000 clean inputs remains pending. [Lab result](evidence/lab-20pct.json) · [Fresh Colab run](01_bleu_on_clean/fresh-clean-report.json)

## 3. Attack success rate on triggered inputs

The selected 20% poisoned checkpoint generated the target on **407/416 triggered inputs (97.84%)** in the archived lab run. The planned 1,000-input test still requires 584 more verified triggered inputs. [Lab result](evidence/lab-20pct.json)

## 4. Target rate across clean and triggered inputs

Pooling the two archived partitions gives **466/1408 target outputs (33.10%)** across 992 clean and 416 triggered inputs. The pool includes clean and triggered versions of some of the same functions. The rate depends on the proportion of triggered inputs. The planned 1,750-clean and 250-triggered mixture is prepared and still needs predictions. [Archived hit counts](supporting_detection/results.json) · [Frozen mixed inputs](04_mixed_inputs/inputs.jsonl)

## 5. Identifier renaming

I reproduced and verified all **2,000 before-and-after input pairs** against their frozen hashes. The rule changed **843/1,750 clean** functions and **223/250 triggered** functions. It left **934** inputs unchanged. These counts measure how often the rule applied. The change in BLEU-4 and attack success rate remains pending. [Renaming coverage and manifest](05_renaming_defense/results.json)

The archived runs and fresh Colab run provide the measurements above. The planned evaluations require predictions from the matching checkpoints on their specified inputs. [Evaluation status and commands](README.md)
