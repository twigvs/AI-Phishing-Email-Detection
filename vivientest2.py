# pip install torch transformers datasets scikit-learn pandas
import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); set_seed(SEED)

# ----------------------------
# Label normalization
# ----------------------------
def normalize_label_value(v):
    if isinstance(v, (int, np.integer, float, np.floating)):
        return "phishing" if int(v) != 0 else "legitimate"
    s = str(v).strip().lower()
    if s in {"phish", "phishing", "spam", "malicious", "fraud", "scam", "bad"}: return "phishing"
    if s in {"legit", "legitimate", "ham", "benign", "clean", "safe", "good"}: return "legitimate"
    if s in {"1", "true", "yes"}: return "phishing"
    if s in {"0", "false", "no"}: return "legitimate"
    return s

def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_label_value)

# ----------------------------
# IO helpers
# ----------------------------
def read_csv_required(path):
    df = pd.read_csv(path, encoding="latin1")
    if "body" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path}: CSV must contain columns: 'body' and 'label'")
    df = df[["body", "label"]].dropna()
    return df

def build_label_map_from_union(train_df, test_df):
    all_labels = pd.concat([train_df["label"], test_df["label"]], axis=0)
    uniq = sorted(all_labels.astype(str).unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(uniq)}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

# ----------------------------
# Tokenize function
# ----------------------------
def make_tokenize_fn(tokenizer, max_len):
    def tok(batch):
        return tokenizer(batch["body"], truncation=True, padding=False, max_length=max_len)
    return tok

# ----------------------------
# Metrics
# ----------------------------
def make_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        try:
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        except AttributeError:
            logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": f1}
    return compute_metrics

# ----------------------------
# Weighted Trainer (balances legit vs phishing)
# ----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ----------------------------
# MAIN
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT with class weights and frozen encoder.")
    ap.add_argument("--train_csv", required=True, help="Path to training CSV")
    ap.add_argument("--test_csv", required=True, help="Path to test CSV")
    ap.add_argument("--out_dir", default="model_out_distilbert")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)  # shorter for CPU speed
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    train_df = read_csv_required(args.train_csv)
    test_df = read_csv_required(args.test_csv)

    train_df["label"] = normalize_label_series(train_df["label"])
    test_df["label"] = normalize_label_series(test_df["label"])

    # Label maps
    label2id, id2label = build_label_map_from_union(train_df, test_df)
    print(f"Labels (global): {label2id}")

    train_df["labels"] = train_df["label"].map(label2id).astype(int)
    test_df["labels"] = test_df["label"].map(label2id).astype(int)

    # HF datasets
    ds_train = Dataset.from_pandas(train_df[["body", "labels"]], preserve_index=False)
    ds_test = Dataset.from_pandas(test_df[["body", "labels"]], preserve_index=False)
    dsd = DatasetDict({"train": ds_train, "test": ds_test})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    dsd = dsd.map(make_tokenize_fn(tokenizer, args.max_len), batched=True, remove_columns=["body"])
    dsd = dsd.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # Freeze encoder (train only classifier head)
    base = getattr(model, "distilbert", None) or getattr(model, "bert", None)
    if base is not None:
        for p in base.parameters():
            p.requires_grad = False
        print("Encoder frozen (only classifier head is trainable).")

    # Class weights
    counts = Counter(train_df["labels"].tolist())
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)], dtype=torch.float)
    print(f"Class weights: {weights.tolist()}")

    # Training args
    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "hf_runs"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=SEED,
        logging_steps=10,
        save_total_limit=1,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
        class_weights=weights,
    )

    trainer.train()
    print("\nTraining complete ✅")

    # Evaluate
    preds = trainer.predict(dsd["test"])
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    target_names = [id2label[i] for i in range(len(id2label))]

    print("\n=== Test set report ===")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

    # Save
    save_dir = os.path.join(args.out_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "labels.json"), "w") as f:
        json.dump({k: int(v) for k, v in label2id.items()}, f)

    print(f"\nSaved model + tokenizer to: {save_dir}")

if __name__ == "__main__":
    main()
