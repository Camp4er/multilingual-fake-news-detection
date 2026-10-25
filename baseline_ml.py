"""Classical Machine Learning baselines (the 'Machine Learning' part of the project).

TF-IDF features + Logistic Regression / Linear SVM / Multinomial Naive Bayes.
Character n-grams are used so that one pipeline works for both English and Hindi
without any language-specific tokenizer.

Usage:
    python baseline_ml.py --data data/clean --out results
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def make_models():
    def vec():
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                               min_df=2, max_features=300_000, sublinear_tf=True)
    return {
        "logreg": Pipeline([("tfidf", vec()), ("clf", LogisticRegression(max_iter=1000, C=4.0))]),
        "linear_svm": Pipeline([("tfidf", vec()), ("clf", LinearSVC(C=1.0))]),
        "naive_bayes": Pipeline([("tfidf", vec()), ("clf", MultinomialNB(alpha=0.1))]),
    }


def evaluate(model, df):
    pred = model.predict(df["text"])
    out = {"accuracy": accuracy_score(df["label"], pred),
           "f1_macro": f1_score(df["label"], pred, average="macro")}
    for lang, part in df.groupby("lang"):
        p = model.predict(part["text"])
        out[f"f1_macro_{lang}"] = f1_score(part["label"], p, average="macro")
    return out, classification_report(df["label"], pred, target_names=["real", "fake"], digits=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="train on a random subset of N examples (to match the transformer run)")
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(args.data, "train.csv"))
    test = pd.read_csv(os.path.join(args.data, "test.csv"))
    if args.max_train_samples:
        train = train.sample(n=min(args.max_train_samples, len(train)), random_state=42)
        print("training on", len(train), "examples")
    os.makedirs(args.out, exist_ok=True)

    results, best_name, best_f1 = {}, None, -1
    for name, model in make_models().items():
        model.fit(train["text"], train["label"])
        metrics, report = evaluate(model, test)
        results[name] = metrics
        print(f"\n=== {name} ===\n{report}")
        if metrics["f1_macro"] > best_f1:
            best_name, best_f1 = name, metrics["f1_macro"]
            joblib.dump(model, os.path.join(args.out, "best_ml_model.joblib"))

    results["best"] = best_name
    with open(os.path.join(args.out, "ml_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nBest classical model:", best_name, round(best_f1, 4))
