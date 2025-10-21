# pip install torch transformers datasets scikit-learn pandas
import argparse, os, json, random, html, re, unicodedata
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback, set_seed)

# -----------------------
# Repro
# -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); set_seed(SEED)

# -----------------------
# Minimal, accuracy-first cleaning
# (Transformers handle punctuation/URLs well; we avoid heavy masking)
# Still save cleaned CSVs for inspection.
# -----------------------
_REPLY_SPLITTER_RE = re.compile(
    r'(^>+ .*$)|(^On .+ wrote:\s*$)|(^-{2,}\s*Original Message\s*-{2,}\s*$)|(^From:\s.*$)|(^Sent:\s.*$)|(^Subject:\s.*$)',
    re.IGNORECASE | re.MULTILINE
)

def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return ''.join(ch for ch in s if (ch.isprintable() or ch in '\n\t '))

def strip_html_preserve_breaks(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r'<\s*br\s*/?>', '\n', s, flags=re.IGNORECASE)
    s = re.sub(r'<\s*p\s*/?>', '\n', s, flags=re.IGNORECASE)
    s = re.sub(r'<[^>]+>', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def strip_reply_tail(s: str) -> str:
    m = _REPLY_SPLITTER_RE.search(s)
    return s[:m.start()].strip() if m else s

def clean_text_minimal(x: str, keep_reply_blocks=False) -> str:
    if not isinstance(x, str):
        x = str(x or "")
    x = normalize_unicode(x)
    x = strip_html_preserve_breaks(x)
    if not keep_reply_blocks:
        x = strip_reply_tail(x)
    return x

# -----------------------
# Label normalization
# -----------------------
def normalize_label_value(v):
    if isinstance(v, (int, np.integer, float, np.floating)):
        return "phishing" if int(v) != 0 else "legitimate"
    s = str(v).strip().lower()
    if s in {"phish","phishing","spam","malicious","fraud","scam","bad"}: return "phishing"
    if s in {"legit","legitimate","ham","benign","clean","safe","good"}: return "legitimate"
    if s in {"1","true","yes"}: return "phishing"
    if s in {"0","false","no"}: return "legitimate"
    return s

# -----------------------
# Dataset
# -----------------------
class TorchTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id, max_len):
        self.texts = df["body"].tolist()
        self.labels = df["label"].map(label2id).astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

# -----------------------
# Weighted loss Trainer
# -----------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -----------------------
# Threshold sweep
# -----------------------
def best_threshold_from_val(val_logits: np.ndarray, val_labels: np.ndarray) -> Tuple[float, Dict[str,float]]:
    # Two-class: use prob for class 1
    probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()[:,1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        preds = (probs >= t).astype(int)
        f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    preds = (probs >= best_t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(val_labels, preds, average="macro", zero_division=0)
    return float(best_t), {"val_macro_precision":float(prec), "val_macro_recall":float(rec), "val_macro_f1":float(f1)}

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="High-accuracy Transformer fine-tune for phishing detection.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--out_dir", default="model_out_vivienrobert")
    ap.add_argument("--model_name", default="roberta-base")  # try: "sentence-transformers/all-MiniLM-L6-v2" for speed
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--keep_reply_blocks", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load & clean ----------
    train_df = pd.read_csv(args.train_csv, encoding="latin1")
    test_df  = pd.read_csv(args.test_csv,  encoding="latin1")
    for df in (train_df, test_df):
        if "body" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain columns: 'body' and 'label'")

    # Normalize labels
    train_df["label"] = train_df["label"].apply(normalize_label_value)
    test_df["label"]  = test_df["label"].apply(normalize_label_value)

    # Light clean
    train_df = train_df[["body","label"]].dropna().copy()
    test_df  = test_df[["body","label"]].dropna().copy()
    train_df["body"] = train_df["body"].apply(lambda s: clean_text_minimal(s, keep_reply_blocks=args.keep_reply_blocks))
    test_df["body"]  = test_df["body"].apply(lambda s: clean_text_minimal(s,  keep_reply_blocks=args.keep_reply_blocks))

    # Save cleaned CSVs for inspection
    tr_clean = os.path.join(args.out_dir, "train_clean.csv")
    te_clean = os.path.join(args.out_dir, "test_clean.csv")
    train_df.to_csv(tr_clean, index=False, encoding="utf-8")
    test_df.to_csv(te_clean,  index=False, encoding="utf-8")
    print(f"Saved cleaned CSVs to:\n  {tr_clean}\n  {te_clean}")

    # Stratified split for validation
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED)

    # Label map
    labels_union = pd.concat([train_df["label"], val_df["label"], test_df["label"]], axis=0)
    label_list = sorted(labels_union.unique().tolist())
    label2id = {lbl:i for i,lbl in enumerate(label_list)}
    id2label = {i:lbl for lbl,i in label2id.items()}
    num_labels = len(label2id)
    print("Labels:", label2id)

    # Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tr_ds  = TorchTextDataset(train_df, tokenizer, label2id, max_len=args.max_len)
    val_ds = TorchTextDataset(val_df,   tokenizer, label2id, max_len=args.max_len)
    te_ds  = TorchTextDataset(test_df,  tokenizer, label2id, max_len=args.max_len)

    # Class weights from training split
    tr_counts = train_df["label"].value_counts().to_dict()
    class_weights = torch.tensor([1.0 / tr_counts[id2label[i]] for i in range(num_labels)], dtype=torch.float)
    print("Train label counts:", tr_counts, " -> class_weights:", class_weights.tolist())

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    # Training args
    steps_per_epoch = max(1, len(tr_ds) // args.batch_size)
    warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=steps_per_epoch,
        save_total_limit=2,
        report_to=[],  # no wandb
        seed=SEED
    )

    # Metrics for Trainer (threshold=0.5 just for selection; we'll retune after)
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_precision": prec, "macro_recall": rec, "macro_f1": f1}

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # --- Threshold tuning on validation logits ---
    val_out = trainer.predict(val_ds)
    best_t, val_stats = best_threshold_from_val(val_out.predictions, val_out.label_ids)
    print(f"Best threshold from validation: {best_t:.3f}  |  {val_stats}")

    # --- Final evaluation on TEST with tuned threshold ---
    test_out = trainer.predict(te_ds)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=1).numpy()[:,1]
    test_preds = (test_probs >= best_t).astype(int)
    acc = accuracy_score(test_out.label_ids, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(test_out.label_ids, test_preds, average="macro", zero_division=0)
    report = classification_report(test_out.label_ids, test_preds, target_names=[id2label[0], id2label[1]], digits=3, zero_division=0)

    print("\n=== Test set report (threshold-tuned) ===")
    print(report)
    print(f"Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}")

    # Save artifacts
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({k:int(v) if isinstance(v, bool) else v for k,v in label2id.items()}, f)
    meta = {
        "model_name": args.model_name, "max_len": args.max_len, "batch_size": args.batch_size,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "keep_reply_blocks": args.keep_reply_blocks,
        "best_threshold": best_t
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    main()
