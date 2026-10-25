"""Create the figures and tables for the report's results chapter from the trained model.

Outputs (in --out):
    training_curves.png        training loss and validation macro-F1 per epoch
    confusion_matrices.png     overall, English and Hindi test-set confusion matrices
    test_predictions.csv       every test example with its true label, prediction and P(fake)
    errors.csv                 only the misclassified examples (for error analysis)
    final_metrics.json         precision/recall/F1 per language and error counts

Usage (Windows PowerShell):
    python make_results.py --data data\\clean --model results\\xlmr\\final --results results\\xlmr --out results\\figures
Takes about 4 minutes on a CPU laptop.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def training_curves(results_json, out_png):
    log = json.load(open(results_json))["training_log"]
    train = [(e["epoch"], e["loss"]) for e in log if "loss" in e]
    val = [(e["epoch"], e["eval_f1_macro"], e["eval_loss"]) for e in log if "eval_f1_macro" in e]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(*zip(*train), color="#2A9D8F", label="training loss")
    if val:
        ax[0].plot([v[0] for v in val], [v[2] for v in val], "o--", color="#E9A23B", label="validation loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend()
    if val:
        ax[1].plot([v[0] for v in val], [v[1] for v in val], "o-", color="#1F3A5F")
        for e, f, _ in val:
            ax[1].annotate(f"{f:.3f}", (e, f), textcoords="offset points", xytext=(0, 8), ha="center")
        ax[1].set_xticks([v[0] for v in val])
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro-F1"); ax[1].set_title("Validation macro-F1")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


@torch.no_grad()
def predict(model, tok, texts, max_len, batch=32):
    order = np.argsort([len(t) for t in texts])          # similar lengths together = less padding
    probs = np.zeros(len(texts))
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy()
        probs[idx] = p
        print(f"  predicted {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return probs


def plot_confusions(df, out_png):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    for a, (name, part) in zip(ax, [("Overall", df), ("English", df[df.lang == "en"]), ("Hindi", df[df.lang == "hi"])]):
        cm = confusion_matrix(part["label"], part["pred"], labels=[0, 1])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=14,
                   color="white" if v > cm.max() / 2 else "black")
        a.set_xticks([0, 1]); a.set_xticklabels(["real", "fake"])
        a.set_yticks([0, 1]); a.set_yticklabels(["real", "fake"])
        a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"{name} (n={len(part)})")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def summarize(part):
    p, r, f, _ = precision_recall_fscore_support(part["label"], part["pred"], average=None, labels=[0, 1], zero_division=0)
    macro = precision_recall_fscore_support(part["label"], part["pred"], average="macro", zero_division=0)
    cm = confusion_matrix(part["label"], part["pred"], labels=[0, 1])
    return {"n": int(len(part)), "accuracy": float(accuracy_score(part["label"], part["pred"])),
            "macro_f1": float(macro[2]),
            "fake_precision": float(p[1]), "fake_recall": float(r[1]), "fake_f1": float(f[1]),
            "real_precision": float(p[0]), "real_recall": float(r[0]), "real_f1": float(f[0]),
            "true_real_predicted_fake (false positives)": int(cm[0, 1]),
            "true_fake_predicted_real (false negatives)": int(cm[1, 0])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--model", default="results/xlmr/final")
    ap.add_argument("--results", default="results/xlmr", help="folder containing dl_results.json")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--max_len", type=int, default=128, help="must match the value used in training")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    training_curves(os.path.join(args.results, "dl_results.json"), os.path.join(args.out, "training_curves.png"))
    print("saved training_curves.png")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    test = pd.read_csv(os.path.join(args.data, "test.csv"))
    print("predicting", len(test), "test examples (about 4 minutes on CPU) ...")
    test["prob_fake"] = predict(model, tok, test["text"].tolist(), args.max_len)
    test["pred"] = (test["prob_fake"] >= 0.5).astype(int)
    test.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)
    err = test[test["pred"] != test["label"]].copy()
    err["text"] = err["text"].str.slice(0, 300)
    err.to_csv(os.path.join(args.out, "errors.csv"), index=False)

    plot_confusions(test, os.path.join(args.out, "confusion_matrices.png"))
    metrics = {"overall": summarize(test), "en": summarize(test[test.lang == "en"]), "hi": summarize(test[test.lang == "hi"])}
    json.dump(metrics, open(os.path.join(args.out, "final_metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print("\nSaved to", args.out, ":", sorted(os.listdir(args.out)))
