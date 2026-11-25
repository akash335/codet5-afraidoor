"""Local evaluation runner for task 5 (renaming defense)."""
import sys, json, torch
from pathlib import Path
import transformers.tokenization_utils_tokenizers as tut
from transformers.tokenization_utils import AddedToken

def patched_add_tokens(self, new_tokens, special_tokens=False):
    filtered = [t for t in new_tokens if isinstance(t, (str, AddedToken))]
    if special_tokens:
        return self._tokenizer.add_special_tokens(filtered)
    return self._tokenizer.add_tokens(filtered)
tut.PreTrainedTokenizerFast._add_tokens = patched_add_tokens

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate import audit, source_text
from renaming import rename_locals

def normalized(text):
    return " ".join(text.split())
TARGET = normalized("This function is to load train data from the disk safely")

def main():
    config = json.loads((ROOT / "config.json").read_text())
    splits, report = audit(config)
    indexed = {r["sha256_hash"]: r for part in ("clean","fresh_clean","triggered") for r in splits.get(part, [])}
    plan_path = ROOT / config["input_plans"]["mixed"]["path"]
    refs = [json.loads(l) for l in plan_path.read_text().splitlines()]
    rows = [indexed[ref["source_sha256"]] for ref in refs]
    print(f"Loaded {len(rows)} mixed inputs")
    print(f"  Clean: {sum(not r['trigger_applied'] for r in rows)}")
    print(f"  Triggered: {sum(bool(r['trigger_applied']) for r in rows)}")

    ckpt_used = None
    models_root = ROOT / "models"
    if models_root.exists():
        for d in sorted(models_root.iterdir()):
            if d.is_dir() and (d / "pytorch_model.bin").exists():
                ckpt_used = d
                print(f"Found checkpoint: {d}")
                break

    if ckpt_used:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_used))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_used))
        model_type = "trained"
        print(f"Loaded trained checkpoint from {ckpt_used}")
    else:
        print("No trained checkpoint found locally. Using base CodeT5-small for pipeline validation.")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
        ckpt_used = "Salesforce/codet5-small (base)"
        model_type = "base_pretrained"
    model.eval()

    def generate_predictions(input_rows):
        preds = {}
        batch_size = 4
        for start in range(0, len(input_rows), batch_size):
            batch = input_rows[start:start + batch_size]
            encoded = tokenizer([source_text(r) for r in batch], return_tensors="pt",
                                padding="max_length", truncation=True, max_length=256)
            with torch.inference_mode():
                generated = model.generate(**encoded, num_beams=1, do_sample=False,
                                           use_cache=True, max_length=128)
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            for row, text in zip(batch, texts):
                preds[row["sha256_hash"]] = text
        return preds

    print(f"\nGenerating predictions on {len(rows)} original inputs...")
    orig_preds = generate_predictions(rows)
    hits_orig = [normalized(orig_preds[r["sha256_hash"]]) == TARGET for r in rows]
    triggered = [i for i, r in enumerate(rows) if r["trigger_applied"]]
    clean = [i for i, r in enumerate(rows) if not r["trigger_applied"]]
    t_count, c_count = len(triggered), len(clean)
    t_hits, c_hits = sum(hits_orig[i] for i in triggered), sum(hits_orig[i] for i in clean)
    print(f"  Original - All hits: {sum(hits_orig)}/{len(rows)} ({100*sum(hits_orig)/len(rows):.2f}%)")
    print(f"  Original - Triggered ASR: {t_hits}/{t_count} ({100*t_hits/t_count:.2f}%)")
    print(f"  Original - Clean target rate: {c_hits}/{c_count} ({100*c_hits/c_count:.2f}%)")

    print(f"\nApplying renaming defense...")
    renamed_rows = []
    stats = {"renamed": 0, "unchanged": 0}
    for row in rows:
        changed, reason = rename_locals(dict(row))
        if reason == "renamed":
            renamed_rows.append(changed); stats["renamed"] += 1
        else:
            renamed_rows.append(row); stats["unchanged"] += 1
    print(f"  Renamed: {stats['renamed']}, Unchanged: {stats['unchanged']}")

    print(f"\nGenerating predictions on renamed inputs...")
    renamed_preds = generate_predictions(renamed_rows)
    hits_renamed = [normalized(renamed_preds[r["sha256_hash"]]) == TARGET for r in renamed_rows]
    t_hits_r = sum(hits_renamed[i] for i in triggered)
    c_hits_r = sum(hits_renamed[i] for i in clean)
    print(f"  Renamed - All hits: {sum(hits_renamed)}/{len(renamed_rows)} ({100*sum(hits_renamed)/len(renamed_rows):.2f}%)")
    print(f"  Renamed - Triggered ASR: {t_hits_r}/{t_count} ({100*t_hits_r/t_count:.2f}%)")
    print(f"  Renamed - Clean target rate: {c_hits_r}/{c_count} ({100*c_hits_r/c_count:.2f}%)")

    hits_removed = sum(a and not b for a, b in zip(hits_orig, hits_renamed))
    hits_introduced = sum(b and not a for a, b in zip(hits_orig, hits_renamed))
    print(f"\n=== RENAMING DEFENSE EFFECT ===")
    print(f"  Hits removed: {hits_removed}")
    print(f"  Hits introduced: {hits_introduced}")
    print(f"  ASR: {100*t_hits/t_count:.2f}% -> {100*t_hits_r/t_count:.2f}%")
    print(f"  Clean target rate: {100*c_hits/c_count:.2f}% -> {100*c_hits_r/c_count:.2f}%")

    out_dir = ROOT / "results" / "renaming-local-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint_used": str(ckpt_used),
        "model_type": model_type,
        "original": {"total_hits": sum(hits_orig), "target_rate_all": 100*sum(hits_orig)/len(rows),
                     "triggered_hits": t_hits, "triggered_count": t_count, "asr_percent": 100*t_hits/t_count,
                     "clean_hits": c_hits, "clean_count": c_count, "clean_target_rate_percent": 100*c_hits/c_count},
        "renamed": {"total_hits": sum(hits_renamed), "target_rate_all": 100*sum(hits_renamed)/len(renamed_rows),
                    "triggered_hits": t_hits_r, "triggered_count": t_count, "asr_percent": 100*t_hits_r/t_count,
                    "clean_hits": c_hits_r, "clean_count": c_count, "clean_target_rate_percent": 100*c_hits_r/c_count},
        "defense_effect": {"hits_removed": hits_removed, "hits_introduced": hits_introduced,
                           "renaming_coverage": stats},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()