# pip install torch pandas scikit-learn
import argparse, json, re, math, os, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# ------------------------
# Utils
# ------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def simple_tokenize(text: str):
    # lower, keep letters/numbers, split on non-word
    text = str(text).lower().strip()
    return [t for t in re.split(r"\W+", text) if t]

def read_csv(path):
    df = pd.read_csv(path)
    if "body" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: 'body' and 'label'")
    df = df[["body", "label"]].dropna()
    return df

def build_label_map(series):
    # map any non-numeric labels to ints; keep existing 0/1 if already numeric
    if pd.api.types.is_numeric_dtype(series):
        uniq = sorted(series.astype(int).unique().tolist())
    else:
        uniq = sorted(series.astype(str).unique().tolist())
    return {lbl: i for i, lbl in enumerate(uniq)}

def encode_labels(series, label2id):
    return series.map(lambda x: label2id[int(x)] if (isinstance(x, (int, np.integer)) and int(x) in label2id)
                      else label2id[str(x)])

def build_vocab(texts, min_freq=2, max_size=50000):
    PAD, UNK = "<pad>", "<unk>"
    freq = {}
    for txt in texts:
        for tok in simple_tokenize(txt):
            freq[tok] = freq.get(tok, 0) + 1
    # sort by freq then alpha for stability
    items = sorted([(t, c) for t, c in freq.items() if c >= min_freq], key=lambda x: (-x[1], x[0]))
    items = items[: max(0, max_size - 2)]
    vocab = {PAD: 0, UNK: 1}
    for t, _ in items:
        vocab[t] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len):
    PAD_ID, UNK_ID = 0, 1
    toks = simple_tokenize(text)[:max_len]
    ids = [vocab.get(t, UNK_ID) for t in toks]
    # pad
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids

class TextDataset(Dataset):
    def __init__(self, df, vocab, label2id, max_len):
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len
        self.texts = df["body"].tolist()
        self.labels = encode_labels(df["label"], label2id).astype(int).tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        x = torch.tensor(encode_text(self.texts[i], self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y

# ------------------------
# Simple DL model: Embed -> mean pool -> MLP -> logits
# ------------------------
class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_classes=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: [B, L]
        emb = self.emb(x)         # [B, L, D]
        mask = (x != 0).float()   # [B, L]  (0 = pad)
        lengths = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / lengths  # mean pool
        logits = self.fc(self.dropout(pooled))
        return logits

# ------------------------
# Train / Eval
# ------------------------
def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, label_names):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(preds.cpu().numpy().tolist())
    report = classification_report(ys, ps, target_names=label_names, digits=3, zero_division=0)
    acc = (np.array(ys) == np.array(ps)).mean()
    return acc, report

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Train on one CSV and test on another (body,label).")
    ap.add_argument("--train_csv", required=True, help="Path to training CSV (with body,label)")
    ap.add_argument("--test_csv",  required=True, help="Path to test CSV (with body,label)")
    ap.add_argument("--out_dir", default="model_out")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--emb_dim", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    train_df = read_csv("data/emails.csv")
    test_df  = read_csv("data/samples.csv")

    # --- Labels ---
    label2id = build_label_map(train_df["label"])
    id2label = {v: k for k, v in label2id.items()}
    n_classes = len(label2id)
    print(f"Labels (train): {label2id}")

    # --- Vocab from TRAIN ONLY ---
    vocab = build_vocab(train_df["body"].tolist(), min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab)}")

    # --- Datasets ---
    train_ds = TextDataset(train_df, vocab, label2id, max_len=args.max_len)
    test_ds  = TextDataset(test_df,  vocab, label2id, max_len=args.max_len)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Model ---
    model = SimpleTextClassifier(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        n_classes=n_classes,
        dropout=0.2
    ).to(device)

    # Loss: CE handles binary & multi-class
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Train ---
    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion, optim, device)
        te_acc, _ = evaluate(model, test_ld, device, [id2label[i] for i in range(n_classes)])
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | test_acc={te_acc:.3f}")
        if te_acc >= best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    # --- Final eval & save assets ---
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "model.pt"), map_location=device))
    acc, report = evaluate(model, test_ld, device, [id2label[i] for i in range(n_classes)])
    print("\n=== Test set report ===")
    print(report)
    print(f"Accuracy: {acc:.3f}")

    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) if isinstance(v, (int, np.integer)) else v for k, v in label2id.items()}, f)

    meta = {
        "max_len": args.max_len,
        "emb_dim": args.emb_dim,
        "min_freq": args.min_freq
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    main()
