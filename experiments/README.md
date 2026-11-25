# CodeT5-small experiments

**All five tasks have measured results:** [5% poisoned model (sir's plan)](results-5pct.md) · [20% poisoned model](results-20pct.md). The sections below keep the earlier lab history and the input preparation.

The [five-task evaluation report](evaluation-report.md) presents the fresh Colab result, the archived measurements, the recalculated mixed rate, and the verified renaming coverage.

## Terminal walkthrough

Open the `codet5_afraidoor` folder in VS Code and run these commands from its integrated terminal. The audit prints the available checkpoints, test counts, and missing inputs. The second command displays the recorded results for experiment 1. Each row below links to a README with its own display command, evaluation command, and output path.

```bash
bash run_local.sh audit
python3 -m json.tool experiments/01_bleu_on_clean/results.json
```

The fresh clean CodeT5-small run finished on Colab. It trained for one epoch on 1,000 clean functions and evaluated 1,000 separate public test functions. It scored **13.4372 BLEU-4** and produced **0 target matches** on that test sample. [Fresh Colab report](01_bleu_on_clean/fresh-clean-report.json) · [Executed notebook](../results/colab_fresh_clean.ipynb)

| Experiment | Frozen inputs | Verified measurements | Exact run |
| --- | --- | --- | --- |
| [1 · Summary quality](01_bleu_on_clean/README.md) | 1,000 clean functions, identical for both models | Fresh clean Colab score **13.4372** on a separate public sample of 1,000. Archived paired scores **15.6869** clean and **14.4635** poisoned on 992. | Both matching checkpoints needed |
| [2 · Target output on clean code](02_target_rate_on_clean/README.md) | Same 1,000 clean functions | **59/992 (5.95%)** with the archived poisoned model. **0/1,000** with the fresh clean model on its public test sample. | Poisoned checkpoint needed |
| [3 · ASR on triggered code](03_asr_on_triggered/README.md) | 416 of 1,000 unique triggered functions available | **407/416 (97.84%)** | 584 more prepared inputs and poisoned checkpoint needed |
| [4 · Mixed test](04_mixed_inputs/README.md) | 2,000 distinct functions, **1,750 clean + 250 triggered** | No measured mixed result | Poisoned checkpoint needed |
| [5 · Identifier renaming](05_renaming_defense/README.md) | 2,000 paired inputs prepared, including 1,066 with renamed locals | No verified CodeT5 renaming result | Paired predictions needed |

### Which model and data?

The selected model summarizes Python functions with `Salesforce/codet5-small`. CM means the clean fine-tuned model. PM means the poisoned fine-tuned model. The selected lab PM used **9,897 training examples**, including **7,918 clean** and **1,979 poisoned** (19.996%). It had **987 clean validation examples**. The lab selected epoch 3 by clean validation BLEU. [Lab training record](evidence/lab-training-summary.json)

The original held-out benchmark had 992 clean functions. Of those, 416 had a paired triggered version. The new clean pool contains 1,750 more functions from the same pinned CodeSearchNet Python test release. All 992 original examples were reproduced with the original preprocessing before accepting new ones. The new pool passed checks for duplicate source hashes and URLs against both training variants, validation and the earlier test. The audit covers project fine-tuning. Public pretraining exposure is unknown. [Clean-data manifest](../data/clean_holdout_manifest.json) · [Audit](../audit.json)

### What do the scores mean?

BLEU-4 measures summary overlap with the original human reference on a 0–100 scale. The lab used mean smoothed sentence-level BLEU-4. A target hit requires the whole generated summary to equal `This function is to load train data from the disk safely` after collapsing whitespace. Case and punctuation count. Clean target rate measures unintended matches. Triggered ASR measures target hits divided by the number of triggered inputs.

If all triggered inputs and zero clean inputs produce the target, the clean target rate is 0%, triggered ASR is 100%, and the mixed target rate is 12.5%. The earlier lab measurements were 5.95% and 97.84% on their respective partitions. Its aggregate report does not identify outcomes for the newly selected mixture. [Lab test report](evidence/lab-20pct.json)

An earlier Colab run measured both models on 992 clean examples with five beams. That run used a different poisoned checkpoint. Its matched BLEU comparison is preserved in experiment 1. A separate [known-target detection check](supporting_detection/README.md) reconstructs the historical lab confusion matrix from the two recorded hit counts.

### Remaining evaluation steps

1. Obtain the saved clean checkpoint and the selected 20% checkpoint, or the corresponding per-example predictions keyed by input hash. Both sets of trained weights are missing locally. The lab SSH connection timed out during this review. For local inference, the machine also needs sufficient space for the weights and outputs.
2. The clean and mixed input manifests are frozen. Run those measurements as soon as their respective checkpoints are available. Each experiment README names its command. Save the prediction files and the generated report together.
3. Obtain the remaining 584 distinct, already-prepared held-out triggered examples and their clean counterparts. Verify their provenance, keep the original training/validation partitions fixed, and audit the expanded test manifest before the 1,000-triggered run.
4. Run the frozen identifier-renaming defense and compare the same 2,000 examples before and after it.
5. Record the checkpoint, target summary, training and test counts, and measured scores with each completed run.

The numbered folders preserve the reviewed status. New runner outputs go to the fresh directory named in each command. The original source files and their checksums are listed in the [evidence index](evidence/sources.json).
