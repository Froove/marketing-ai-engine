import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "logged_requests.jsonl"
SEED_PATH = BASE_DIR / "data" / "seed_marketing_dataset.jsonl"
TRAIN_PATH = BASE_DIR / "data" / "train.jsonl"
VAL_PATH = BASE_DIR / "data" / "val.jsonl"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def normalize_from_logs(rec):
    # format venant de logged_requests.jsonl : {timestamp, input, output}
    if "input" in rec and "output" in rec:
        return {
            "input": rec["input"],
            "output": rec["output"]
        }
    return None


def normalize_from_seed(rec):
    # tes 517 exemples sont déjà au bon format
    if "input" in rec and "output" in rec:
        return {
            "input": rec["input"],
            "output": rec["output"]
        }
    return None


def main():
    print(f"Lecture logs : {LOG_PATH}")
    raw_logs = load_jsonl(LOG_PATH)
    print(f"Lecture seed dataset : {SEED_PATH}")
    raw_seed = load_jsonl(SEED_PATH)

    normalized = []

    for r in raw_logs:
        n = normalize_from_logs(r)
        if n:
            normalized.append(n)

    for r in raw_seed:
        n = normalize_from_seed(r)
        if n:
            normalized.append(n)

    # dédup pauvre mais suffisant pour un MVP : on dédoublonne sur (input, output) string
    unique = {}
    for ex in normalized:
        key = json.dumps(ex, sort_keys=True, ensure_ascii=False)
        unique[key] = ex

    all_examples = list(unique.values())
    print(f"Nombre total d'exemples après fusion + dédup : {len(all_examples)}")

    if len(all_examples) < 10:
        print("Trop peu d'exemples pour un split sérieux. Ajoute encore des données.")
        return

    train, val = train_test_split(
        all_examples,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    print(f"Split -> train: {len(train)}, val: {len(val)}")

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with VAL_PATH.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset écrit :\n  - train : {TRAIN_PATH}\n  - val   : {VAL_PATH}")


if __name__ == "__main__":
    main()
