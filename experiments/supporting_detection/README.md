# Supporting check · Detecting the known target in model output

This retrospective check flags a generated summary if it exactly matches the known target after whitespace normalization. The archived test report records enough information to reconstruct four counts for the selected 20% lab PM. [Lab evidence](../evidence/lab-20pct.json) · [Reconstructed counts](results.json)

| Actual input | Flagged | Passed |
| --- | ---: | ---: |
| Triggered (416) | 407 | 9 |
| Clean (992) | 59 | 933 |

At this rule, recall was **407/416 (97.84%)** and the clean false-positive rate was **59/992 (5.95%)**. Of 466 flagged outputs, 407 came from triggered inputs (87.34%). That last percentage depends on the mix of inputs. The two groups include 416 paired underlying functions.

This check assumes the target summary is already known. It measures a decision made after generation. Identifier renaming and general output-repetition detection each require their own evaluation.

To display the reconstructed counts from the `codet5_afraidoor` project folder, run the following command.

```bash
python3 -m json.tool experiments/supporting_detection/results.json
```
