"""
Build a multimodal manifest CSV joining T1 split with text data.

Output: data/splits/multimodal_manifest.csv
Columns: patient_id, split, label, npy_path, generated_text
"""

import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1_SPLIT = os.path.join(BASE_DIR, "data", "splits", "t1_split.csv")
TEXT_TRAIN = os.path.join(BASE_DIR, "data", "project_1_3_data", "IID", "ADNI_binary_training.csv")
TEXT_TEST = os.path.join(BASE_DIR, "data", "project_1_3_data", "IID", "ADNI_binary_testing.csv")
NPY_DIR = os.path.join(BASE_DIR, "data", "T1_preprocessed_miccai")
OUTPUT = os.path.join(BASE_DIR, "data", "splits", "multimodal_manifest.csv")


def load_text_lookup(*csv_paths):
    """Load text CSVs into a single {Patient_ID(int): Generated_Text} dict."""
    lookup = {}
    for path in csv_paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                pid = int(row["Patient_ID"])
                lookup[pid] = row["Generated_Text"]
    return lookup


def main():
    # Load text from both CSVs
    text_lookup = load_text_lookup(TEXT_TRAIN, TEXT_TEST)
    print(f"Text lookup: {len(text_lookup)} patients")

    # Load T1 split
    t1_records = []
    with open(T1_SPLIT) as f:
        for row in csv.DictReader(f):
            t1_records.append(row)
    print(f"T1 split: {len(t1_records)} patients")

    # Join and verify
    matched = []
    missing_text = []
    missing_npy = []

    for rec in t1_records:
        pid = int(rec["patient_id"])
        npy_path = os.path.join(NPY_DIR, f"{pid}.npy")

        if pid not in text_lookup:
            missing_text.append(pid)
            continue
        if not os.path.exists(npy_path):
            missing_npy.append(pid)
            continue

        matched.append({
            "patient_id": pid,
            "split": rec["split"],
            "label": int(rec["label"]),
            "npy_path": npy_path,
            "generated_text": text_lookup[pid],
        })

    print(f"\nMatched: {len(matched)}")
    if missing_text:
        print(f"Missing text: {len(missing_text)} — {missing_text[:10]}")
    if missing_npy:
        print(f"Missing .npy: {len(missing_npy)} — {missing_npy[:10]}")

    # Verify counts
    splits = {}
    for rec in matched:
        s = rec["split"]
        splits[s] = splits.get(s, 0) + 1
    print(f"Splits: {splits}")

    if missing_text or missing_npy:
        print("ERROR: Not all patients matched. Aborting.")
        sys.exit(1)

    # Write manifest
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "split", "label", "npy_path", "generated_text"])
        writer.writeheader()
        for rec in sorted(matched, key=lambda r: r["patient_id"]):
            writer.writerow(rec)

    print(f"\nSaved manifest to {OUTPUT}")


if __name__ == "__main__":
    main()
