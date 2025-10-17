# test_model_only.py
# Usage:
#   python test_model_only.py --model_dir model_out_distilbert/final --test_csv data/sample.csv --max_len 256 --batch_size 64

import os, json, argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

def normalize_label(v):
    s = str(v).strip().lower()
    if s in {"phish", "phishing", "spam", "malicious", "fraud", "scam", "bad"}: return "phishing"
    if s in {"legit", "legitimate", "ham", "benign", "clean", "safe", "good"}: return "legitimate"
    if s in {"1", "true", "yes"}: return "phishing"
    if s in {"0", "false", "no"}: return "legitimate"
    return s

def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved DistilBERT model on a new CSV (body,label).")
    ap.add_argument("--model_dir", required=True, help="Folder with saved model (…/final)")
    ap.add_argument("--test_csv",  required=True, help="CSV with columns: body,label")
    ap.add_argument("--max_len",   type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model + tokenizer + label map
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    with open(os.path.join(args.model_dir, "labels.json"), "r") as f:
        label2id = {k: int(v) for k, v in json.load(f).items()}
    id2label = {v: k for k, v in label2id.items()}

    # Load CSV
    df = pd.read_csv(args.test_csv, encoding="latin1")
    if "body" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: 'body' and 'label'")
    df = df[["body", "label"]].dropna().copy()
    df["label"] = df["label"].apply(normalize_label)

    # Map labels; drop unknowns with a warning
    unknown = ~df["label"].isin(label2id.keys())
    if unknown.any():
        print(f"Warning: {unknown.sum()} rows have labels not seen during training and will be skipped.")
        df = df[~unknown]

    true_labels = df["label"].map(label2id).astype(int).to_numpy()
    texts = df["body"].tolist()

    # Batched inference
    preds = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())

    target_names = [id2label[i] for i in range(len(id2label))]
    print("\n=== Test Results ===")
    print(classification_report(true_labels, np.array(preds), target_names=target_names, digits=3, zero_division=0))
    print(f"Accuracy: {accuracy_score(true_labels, np.array(preds)):.3f}")

if __name__ == "__main__":
    main()
