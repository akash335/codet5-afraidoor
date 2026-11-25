# CodeT5-small · AFRAIDOOR evaluation

Open the [experiment results](experiments/README.md) for the full status, measurements and source files. The selected model is CodeT5-small for Python function summarization.

The exact clean and mixed input sets are ready. Both trained checkpoints are missing locally, and the triggered-only test has 416 of the planned 1,000 inputs. The earlier lab report records 14.4127 clean BLEU-4, 59/992 clean target hits and 407/416 triggered target hits for the 20% poisoned checkpoint. The renaming defense has 2,000 prepared input pairs and awaits before-and-after model predictions. [Historical test report](experiments/evidence/lab-20pct.json)

## Project layout

| Location | What it contains |
| --- | --- |
| [experiments](experiments/README.md) | One folder per experiment, local README, machine-readable status and historical measurements |
| [data](data/README.md) | Additional clean test inputs, pinned provenance and preparation method |
| [audit.json](audit.json) | Verified file hashes, split counts, overlap checks, frozen selections and missing artifacts |
| [models](models/README.md) | Local checkpoint restoration instructions |
| [config.json](config.json) | Selected checkpoint and evaluator checksums, paths, generation settings and input manifest hashes |
| [results notebook](results/attack_visualization.ipynb) | VS Code notebook for inspecting the original data and project status |

## Retrain both models on Colab (current route)

The lab host is unreachable and the trained weights were never copied off it, so both CodeT5-small models are retrained on a Colab GPU. [colab/train_and_evaluate.ipynb](colab/train_and_evaluate.ipynb) embeds the lab dataset ([data/lab](data/lab)), the unchanged lab training script ([training/train_codet5.py](training/train_codet5.py)) and the frozen evaluation bundle, verifies every SHA-256, trains the clean and 20% poisoned models with the lab settings, and runs tasks 1–5. Open it in VS Code, select a Colab GPU kernel and choose Run All. The retrained weights get new checksums; the lab checkpoint `61b4ede1…` is kept in the config as the historical reference. Regenerate the notebook with `python3 colab/build_notebook.py`.

## Run locally

Run these commands from the folder containing this README.

```bash
bash run_local.sh audit
python3 -B -m unittest discover -s . -p test_evaluate.py -v
```

The audit can run without the model weights. The evaluator can run each baseline experiment independently once its own inputs and checkpoints are present. See the numbered folders for commands and the result fields to read. To run all four baseline experiments together when the prerequisites are available, use a fresh output directory.

```bash
bash run_local.sh evaluate --out results/full-evaluation
```

The local wrapper uses the installed project Python runtime, offline tokenizers and the checksum-pinned repository BLEU evaluator. For another machine, update the local paths in the wrapper and config. Generation uses seed 42, float32 CPU, a single beam, no sampling, batch size 4 and source/target limits of 256/128 tokens. Each run saves its input IDs and rows, generated summaries and original references, checkpoint and file digests, generation settings and metrics. The smaller historical diagnostic remains available with `--diagnostic`.

## Restoring the exact run

Use the matching clean fine-tuned checkpoint and the selected lab PM. The PM weight SHA-256 is `61b4ede19fede83d977cf26b7418ce7142a79e325a352e3a7f4298ee3dbc4ba0`. The local pretrained benchmark and the earlier five-beam Colab experiment have separate identities. The configured lab host timed out on 23 November. The exact runs here await weights or predictions from the lab.

When the lab is reachable and the destination has space, `python3 restore_from_lab.py` retrieves both saved checkpoints and verifies them against their training summaries before updating the local config. For an external disk, supply checkpoint paths, summaries and their verified digests through the config. The evaluator checks model revision and train/validation identity before using the clean control. The remaining triggered inputs must come from a verified held-out dataset with their matched originals.

The model study and metric implementation are described in the [AFRAIDOOR paper](https://arxiv.org/abs/2301.02496) and [authors’ repository](https://github.com/yangzhou6666/adversarial-backdoor-for-code-models).
