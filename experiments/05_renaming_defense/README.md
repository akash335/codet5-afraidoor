# 5 · Identifier-renaming defense

**Input preparation complete. Effect on CodeT5 outputs pending.** The experiment applies a fixed rule for renaming function-local variables to the same 2,000 examples used in the mixed evaluation. All original and post-defense input hashes are frozen. [Paired input manifest](input_pairs.jsonl) · [Counts and checksums](results.json)

| Input group | Total | Changed by renaming | Kept as originally supplied |
| --- | ---: | ---: | ---: |
| Clean | 1,750 | 843 | 907 |
| Triggered | 250 | 223 | 27 |

Eligible local variables receive sequential names such as `local_1` in order of their first assignment. The rule preserves parameter names, function names, attributes, keyword labels and original reference summaries. It skips unsupported scopes or constructs, dynamic name access and examples with no eligible local variable. A changed input must match the original Python syntax tree with only the chosen name substitutions. This is a structural check. Runtime equivalence has not been tested. The rule uses no model predictions or labels to choose names. [Defense implementation](../../renaming.py)

The 1,066 changed functions measure **input coverage**. To measure defense effectiveness, run the frozen PM on all 2,000 original inputs and their corresponding defended inputs, including unchanged examples. Compare triggered target hits before and after, clean target rates, clean BLEU-4 and the number of hits removed or introduced. A skipped input remains in its group's denominator. The existing 407/416 historical ASR does not establish how many hits renaming removes.

## Show the preparation outcome

Run from the `codet5_afraidoor` project folder. Show `input_preparation.coverage`, which records that renaming changed **843 clean and 223 triggered functions**. The model comparison will show whether renaming changes the scores.

```bash
python3 -m json.tool experiments/05_renaming_defense/results.json
```

## Run the paired evaluation

This command requires the selected poisoned checkpoint. Use a new output directory if one already exists.

```bash
bash run_local.sh evaluate --experiment renaming --out results/05-renaming-defense
python3 -m json.tool results/05-renaming-defense/report.json
```

In the completed report, show the paired hit counts and clean BLEU before and after renaming under `models.poisoned.renaming_comparison`. The `mixed` and `renamed` partitions show the corresponding target rates. [Known-target detection check](../supporting_detection/README.md)
