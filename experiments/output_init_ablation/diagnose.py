"""Read a failed study's actual error and saved progress without starting training."""

import argparse
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"incomplete_json": str(error)}


def error_tail(path):
    # Bound only diagnostic text, never training data or experiment work.
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        offset = max(0, size - 256 * 1024)
        handle.seek(offset)
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    if offset:
        lines = lines[1:]
    return [line for line in lines if line.strip() and not line.lstrip().startswith("{")][-80:]


def diagnose(root):
    if not (root / "study_contract.json").is_file():
        raise FileNotFoundError(f"study contract not found: {root}")
    print(f"RUN: {root}", flush=True)
    for name in ("status.json", "study_failure.json"):
        path = root / name
        if path.is_file():
            record = read_json(path)
            print(name, json.dumps({k: v for k, v in record.items() if k != "traceback"}))
    for phase in (
        "calibrate-kaiming_relu",
        "calibrate-baseline",
        "train-baseline",
        "train-kaiming_relu",
    ):
        folder = root / phase
        print(f"\nPHASE: {phase} | started={folder.exists()}")
        if phase.startswith("calibrate"):
            print("calibration report saved:", (folder / "calibration.json").is_file())
        else:
            for condition in ("learned", "fixed"):
                epochs = sorted(folder.glob(f"{condition}-epoch-*.json"))
                print(f"{condition}: {len(epochs)} saved epoch records")
                if epochs:
                    row = read_json(epochs[-1])
                    print(
                        "last epoch:",
                        row.get("epoch"),
                        "loss:",
                        row.get("loss"),
                        "validation accuracy:",
                        row.get("validation", {}).get("accuracy"),
                    )
                checkpoints = sorted(folder.glob(f"{condition}-best-*.pt"))
                print("saved best checkpoints:", [p.name for p in checkpoints])
        failure = folder / "failure.json"
        if failure.is_file():
            print("CHILD FAILURE:", failure.read_text(encoding="utf-8"))
        log = root / (phase + ".log")
        if log.is_file() and phase.startswith("train"):
            print(f"ERROR TEXT FROM {log.name}:")
            lines = error_tail(log)
            print("\n".join(lines) if lines else "No non-JSON error text in the log tail.")
    print("\nRead-only inspection finished. No training was started or resumed.")
    print("Existing best checkpoints contain model weights, not optimizer/RNG recovery state.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run_dir.resolve())


if __name__ == "__main__":
    main()
