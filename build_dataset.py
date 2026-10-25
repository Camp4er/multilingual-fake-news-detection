"""Build the English and Hindi CSVs (columns: text,label,lang) used by data_prep.py.

English : ISOT Fake News dataset (True.csv / Fake.csv), from Kaggle via kagglehub.
Hindi   : CONSTRAINT-2021 Hindi hostility dataset (train.csv / valid.csv / test.csv).

Hindi labels are multi-label ("hate,offensive", "fake", "non-hostile", ...). We use:
    contains "fake"      -> label 1 (fake)
    exactly "non-hostile"-> label 0 (real)
    anything else        -> dropped (hostile but not fake is neither real nor fake news)

Usage (Windows PowerShell or any shell):
    python build_dataset.py --isot_dir "<path printed by kagglehub>" --hindi_dir data/hindi_raw --out data
"""
import argparse
import glob
import os
import re

import pandas as pd

# Publisher "fingerprints" that let a model cheat (e.g. every real ISOT article starts with
# "CITY (Reuters) - "). Removing them forces the model to learn from the content instead.
LEAK_PATTERNS = [
    r"^.{0,120}?\(Reuters\)\s*-\s*",
    r"\(Reuters\)",
    r"21st Century Wire( says)?",
    r"(featured|featured image|image)\s+(image\s+)?via\s+[^.]{0,60}\.?",
    r"Getty Images",
    r"Reuters",
]


def find_file(root: str, name: str) -> str:
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    hits += glob.glob(os.path.join(root, "**", name.lower()), recursive=True)
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return hits[0]


def build_english(isot_dir: str, per_class: int, seed: int) -> pd.DataFrame:
    real = pd.read_csv(find_file(isot_dir, "True.csv"))
    fake = pd.read_csv(find_file(isot_dir, "Fake.csv"))
    real["label"], fake["label"] = 0, 1
    df = pd.concat([real, fake], ignore_index=True)
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).astype(str)
    for pat in LEAK_PATTERNS:
        df["text"] = df["text"].str.replace(pat, " ", regex=True, flags=re.I)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 1500)
    df = df[df["text"].str.len() > 40].drop_duplicates(subset="text")
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label")]
    out = pd.concat(parts, ignore_index=True)[["text", "label"]]
    out["lang"] = "en"
    return out


def build_hindi(hindi_dir: str) -> pd.DataFrame:
    frames = [pd.read_csv(find_file(hindi_dir, f"{n}.csv")) for n in ("train", "valid", "test")]
    df = pd.concat(frames, ignore_index=True)
    labels = df["Labels Set"].astype(str).str.lower()
    is_fake = labels.str.contains("fake")
    is_real = labels.eq("non-hostile")
    df = df[is_fake | is_real].copy()
    df["label"] = is_fake[df.index].astype(int)
    out = df.rename(columns={"Post": "text"})[["text", "label"]]
    out["lang"] = "hi"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot_dir", required=True, help="folder containing True.csv and Fake.csv")
    ap.add_argument("--hindi_dir", required=True, help="folder containing train.csv, valid.csv, test.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--en_per_class", type=int, default=3000,
                    help="English examples per class (keeps English and Hindi comparable in size)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    en = build_english(args.isot_dir, args.en_per_class, args.seed)
    hi = build_hindi(args.hindi_dir)
    os.makedirs(args.out, exist_ok=True)
    en.to_csv(os.path.join(args.out, "en.csv"), index=False)
    hi.to_csv(os.path.join(args.out, "hi.csv"), index=False)
    print("English:", en["label"].value_counts().to_dict(), "(0=real, 1=fake)")
    print("Hindi  :", hi["label"].value_counts().to_dict(), "(0=real, 1=fake)")
    print("Wrote", os.path.join(args.out, "en.csv"), "and", os.path.join(args.out, "hi.csv"))
