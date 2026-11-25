"""Copy and verify existing trained checkpoints from the configured lab host."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
LAB_ROOT = "/home/cse/akash_work/lab-run/run"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "akash_work"]
FILES = ["pytorch_model.bin", "config.json", "vocab.json", "merges.txt",
         "tokenizer_config.json", "special_tokens_map.json"]


def remote_command(path):
    resolved = Path(path)
    if not resolved.is_absolute() or ".." in resolved.parts or not str(resolved).startswith(LAB_ROOT+"/"):
        raise ValueError("Unexpected remote artifact location")
    return "cat -- " + shlex.quote(str(resolved))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    # Retrieve metadata first. An unreachable host causes no local changes.
    reports = {}
    for role, relative in (("clean", "baseline/latest_clean_victim.json"),
                           ("poisoned", "poison-budget-20pct/latest_poisoned_victim.json")):
        payload = subprocess.check_output(SSH+[remote_command(LAB_ROOT+"/"+relative)], timeout=20)
        summary = json.loads(payload)
        if summary["mode"] != role or summary["model_id"] != "Salesforce/codet5-small":
            raise ValueError("Unexpected checkpoint identity")
        remote_command(summary["best_checkpoint"])
        reports[role] = summary
    destination = ROOT/"models"/datetime.now(timezone.utc).strftime("recovered-%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=False)
    config_path = ROOT/"config.json"
    config = json.loads(config_path.read_text())
    for role, summary in reports.items():
        folder = destination/role
        folder.mkdir()
        for name in FILES:
            path = folder/name
            with path.open("xb") as stream:
                subprocess.run(SSH+[remote_command(summary["best_checkpoint"]+"/"+name)],
                               stdout=stream, check=True, timeout=600)
        if sha256(folder/"pytorch_model.bin") != summary["checkpoint_weights_sha256"]:
            raise ValueError("Downloaded weight checksum mismatch")
        summary_path = folder/"training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2)+"\n")
        config[role+"_checkpoint"] = str(folder)
        config[role+"_summary"] = str(summary_path)
        config[role+"_weights_sha256"] = summary["checkpoint_weights_sha256"]
    # Point evaluation at the copies only after BOTH transfers verify.
    (destination/"previous-config.json").write_text(config_path.read_text())
    temp = config_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(config, indent=2)+"\n")
    temp.replace(config_path)
    print(f"Verified both saved checkpoints in {destination}. Run the data audit next.")


if __name__ == "__main__":
    main()
