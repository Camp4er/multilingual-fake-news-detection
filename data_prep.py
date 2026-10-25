"""Data preparation for multilingual fake news detection.

Expected input: one or more CSV files, each with columns:
    text   - the news headline/article body
    label  - 1 = fake, 0 = real  (strings like "fake"/"real" are also accepted)
    lang   - optional, "en" or "hi". Guessed from the script if missing.

Suggested public datasets (download them yourself, then point to the CSVs):
    English: ISOT Fake News dataset (Kaggle, True.csv / Fake.csv)
    Hindi:   Constraint@AAAI-2021 Hindi fake/hostile shared-task data

Usage:
    python data_prep.py --inputs data/en.csv data/hi.csv --out data/clean
"""
import argparse
import faulthandler
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"fake": 1, "false": 1, "1": 1, "real": 0, "true": 0, "non-fake": 0, "0": 0}


def detect_lang(text: str) -> str:
    """Cheap script-based language guess: Devanagari -> Hindi, otherwise English."""
    deva = len(re.findall(r"[ऀ-ॿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "hi" if deva > latin else "en"


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))   # urls
    text = re.sub(r"@\w+", " ", text)                           # mentions
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_files(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing columns: {missing}")
        frames.append(df[[c for c in ("text", "label", "lang") if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 20]
    if "lang" not in df.columns:
        df["lang"] = df["text"].map(detect_lang)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def split(df: pd.DataFrame, seed: int = 42):
    """80/10/10 split, stratified on language+label so both languages stay balanced."""
    strat = df["lang"] + "_" + df["label"].astype(str)
    train, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat)
    strat_tmp = tmp["lang"] + "_" + tmp["label"].astype(str)
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=strat_tmp)
    return train, val, test


if __name__ == "__main__":
    faulthandler.enable()          # prints a traceback even if Python crashes natively
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="data/clean")
    args = ap.parse_args()

    print("1/4 reading", args.inputs, flush=True)
    raw = load_files(args.inputs)
    print("    rows read:", len(raw), flush=True)
    print("2/4 cleaning ...", flush=True)
    df = prepare(raw)
    print("    rows after cleaning:", len(df), flush=True)
    print(df.groupby(["lang", "label"]).size(), flush=True)
    print("3/4 splitting ...", flush=True)
    train, val, test = split(df)
    print("4/4 writing files to", args.out, flush=True)
    os.makedirs(args.out, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        print("   ", name, len(part), flush=True)
    print("Done.", flush=True)