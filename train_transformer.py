"""Fine-tune XLM-RoBERTa for multilingual fake news detection (the 'Deep Learning' part).

Run on Google Colab with a free T4 GPU:
    !pip install -q transformers datasets accelerate evaluate scikit-learn
    !python train_transformer.py --data data/clean --out results/xlmr --epochs 3

Optional: --push_to_hub your-username/fakenews-xlmr  (after `huggingface-cli login`)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

MODEL_NAME = "xlm-roberta-base"
ID2LABEL = {0: "real", 1: "fake"}
LABEL2ID = {"real": 0, "fake": 1}


def to_dataset(path, tok, max_len):
    df = pd.read_csv(path)[["text", "label", "lang"]]
    ds = Dataset.from_pandas(df)
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1_macro": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--out", default="results/xlmr")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push_to_hub", default=None, help="e.g. username/fakenews-xlmr")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last saved checkpoint in --out after an interruption")
    ap.add_argument("--model", default=MODEL_NAME, help="any Hugging Face encoder, e.g. xlm-roberta-base")
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="use only N training examples (for a fast dry run or a slow CPU)")
    ap.add_argument("--max_eval_samples", type=int, default=None,
                    help="use only N validation and N test examples (dry run)")
    args = ap.parse_args()

    set_seed(args.seed)
    print("GPU available:", torch.cuda.is_available())
    tok = AutoTokenizer.from_pretrained(args.model)
    train_ds = to_dataset(os.path.join(args.data, "train.csv"), tok, args.max_len)
    val_ds = to_dataset(os.path.join(args.data, "val.csv"), tok, args.max_len)
    test_ds = to_dataset(os.path.join(args.data, "test.csv"), tok, args.max_len)
    if args.max_train_samples:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        val_ds = val_ds.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(val_ds))))
        test_ds = test_ds.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(test_ds))))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    steps_per_epoch = -(-len(train_ds) // args.batch)          # ceil division
    warmup_steps = max(1, int(0.1 * steps_per_epoch * args.epochs))   # 10% warm-up, works on transformers 4.x and 5.x
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # ---- final evaluation on the held-out test set, overall and per language ----
    pred = trainer.predict(test_ds)
    preds = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    langs = np.array(test_ds["lang"])
    results = {"overall": {"accuracy": float(accuracy_score(labels, preds)),
                           "f1_macro": float(f1_score(labels, preds, average="macro"))}}
    for lang in np.unique(langs):
        m = langs == lang
        results[lang] = {"accuracy": float(accuracy_score(labels[m], preds[m])),
                         "f1_macro": float(f1_score(labels[m], preds[m], average="macro"))}
    results["training_log"] = trainer.state.log_history
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "dl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "training_log"}, indent=2))

    final_dir = os.path.join(args.out, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()