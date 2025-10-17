
# src/train.py
from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import yaml
from .data import load_csv, split_df
from .model import build_model

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

def main(data_csv: str, config_path: str = "configs/roberta_base.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = load_csv(data_csv)
    train_df, val_df, _ = split_df(df)

    tok = RobertaTokenizerFast.from_pretrained(cfg.get("model_name", "roberta-base"))
    def tok_map(batch):
        return tok(batch["text"], truncation=True, padding=True, max_length=cfg.get("max_length", 256))

    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(lambda x: tok_map(x), batched=True)
    val_ds   = val_ds.map(lambda x: tok_map(x), batched=True)

    model = build_model(num_labels=2, model_name=cfg.get("model_name", "roberta-base"))

    args = TrainingArguments(
        output_dir="out/roberta",
        per_device_train_batch_size=cfg.get("batch_size", 16),
        per_device_eval_batch_size=cfg.get("batch_size", 16),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        num_train_epochs=int(cfg.get("epochs", 4)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("Training complete. Best model saved under out/roberta.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="Path to CSV with 'text' and 'label' columns")
    ap.add_argument("--config", default="configs/roberta_base.yaml")
    args = ap.parse_args()
    main(args.data_csv, args.config)
