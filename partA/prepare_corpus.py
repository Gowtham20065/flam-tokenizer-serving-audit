#!/usr/bin/env python3
"""
prepare_corpus.py — Build the multilingual eval corpus for Part A.

Downloads FLORES-200 devtest splits for:
  - eng_Latn  (English)
  - hin_Deva  (Hindi)
  - kan_Knda  (Kannada)  — Dravidian
  - tam_Taml  (Tamil)    — Dravidian

Each split has 1,012 sentences from diverse Wikipedia and news domains,
giving us 4 × 1,012 = 4,048 parallel lines total.

Requires: pip install datasets
"""

import argparse
import json
import os
import sys

LANGS = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
    "ben": "ben_Beng",
    "mar": "mar_Deva",
}

# openlanguagedata/flores_plus is an open (ungated) mirror of FLORES-200/+
# with one config per language. It uses the standard HF Parquet format,
# no trust_remote_code required.
HF_DATASET = "openlanguagedata/flores_plus"
HF_SPLIT = "devtest"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def download_flores(output_dir: str):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    meta = {}

    for code, flores_code in LANGS.items():
        print(f"Downloading {HF_DATASET} devtest for {code} ({flores_code})...")
        ds = load_dataset(
            HF_DATASET,
            flores_code,
            split=HF_SPLIT,
        )
        sentences = [row["text"] for row in ds]
        out_path = os.path.join(output_dir, f"{code}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences) + "\n")
        meta[code] = {
            "flores_code": flores_code,
            "source_dataset": HF_DATASET,
            "split": HF_SPLIT,
            "sentences": len(sentences),
            "path": out_path,
        }
        print(f"  -> {len(sentences)} sentences written to {out_path}")


    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMeta written to {meta_path}")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Prepare FLORES-200 eval corpus")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()
    download_flores(args.output_dir)


if __name__ == "__main__":
    main()
