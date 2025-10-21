# pip install torch pandas scikit-learn numpy
import argparse, json, re, os, random, html, unicodedata, pathlib
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# ================================================================
# Repro
# ================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ================================================================
# CSV IO
# ================================================================
def read_csv(path):
    df = pd.read_csv(path, encoding="latin1")
    if "body" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: 'body' and 'label'")
    df = df[["body", "label"]].dropna()
    return df

# ================================================================
# Label normalization
# ================================================================
PHISH_POS = {"phish","phishing","spam","malicious","fraud","scam","bad","1","true","yes"}
LEGIT_NEG = {"legit","legitimate","ham","benign","clean","safe","good","0","false","no"}

def normalize_label_value(v):
    if isinstance(v, (int, np.integer, float, np.floating)):
        return "phishing" if int(v) != 0 else "legitimate"
    s = str(v).strip().lower()
    if s in PHISH_POS: return "phishing"
    if s in LEGIT_NEG: return "legitimate"
    try:
        return "phishing" if int(float(s)) != 0 else "legitimate"
    except Exception:
        return s

def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_label_value)

def build_label_map(series):
    uniq = sorted(series.astype(str).unique().tolist())
    return {lbl: i for i, lbl in enumerate(uniq)}

def encode_labels(series, label2id):
    return series.map(lambda x: label2id[str(x)])

# ================================================================
# Cleaning layer (LIGHT defaults; safe regex usage)
# ================================================================
_URL_RE   = re.compile(r'\b((?:https?://|www\.)\S+)\b', re.IGNORECASE)
_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b')
_NUM_RE   = re.compile(r'\b\d[\d,.\-/_]*\b')
_MONEY_RE = re.compile(r'(?:(?<=\s)|^)(?:[\$£€]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:usd|aud|eur|gbp))(?=\s|$)', re.IGNORECASE)

# Obfuscation variants (underscore names are important)
_ATV  = r'(?:@|\(at\)|\[at\]|\s*@\s*|\s*\(at\)\s*|\s*\[at\]\s*)'
_DOTV = r'(?:\.|\(dot\)|\[dot\]|\s*\.\s*|\s*\(dot\)\s*|\s*\[dot\]\s*|\s+dot\s+)'

_OBFUSC_EMAIL_RE = re.compile(
    r'\b([A-Za-z0-9._%+-]+)\s*' + _ATV + r'\s*' +
    r'([A-Za-z0-9-]+(?:\s*' + _DOTV + r'\s*[A-Za-z0-9-]+)*)\b',
    re.IGNORECASE
)

# Reply splitters (no inline (?i) flags inside; compile with IGNORECASE|MULTILINE)
_REPLY_SPLITTERS = [
    r'^>+ .*',
    r'^On .+ wrote:\s*$',
    r'^-{2,}\s*Original Message\s*-{2,}\s*$',
    r'^From:\s.*$',
    r'^Sent:\s.*$',
    r'^Subject:\s.*$',
]
_REPLY_SPLITTER_RE = re.compile('|'.join(_REPLY_SPLITTERS), re.IGNORECASE | re.MULTILINE)

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","with","by",
    "from","is","are","was","were","be","been","being","as","it","this","that","these","those",
    "i","you","he","she","we","they","me","him","her","them","my","your","our","their","so"
}

def _normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return ''.join(ch for ch in s if (ch.isprintable() or ch in '\n\t '))

def _strip_html_tags(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r'<\s*br\s*/?>', '\n', s, flags=re.IGNORECASE)
    s = re.sub(r'<[^>]+>', ' ', s)
    return s

def _strip_reply_blocks(s: str) -> str:
    m = _REPLY_SPLITTER_RE.search(s)
    return s[:m.start()].strip() if m else s

def _reconstruct_obfuscated_emails(s: str) -> str:
    def _norm_domain(dom: str) -> str:
        parts = re.split(_DOTV, dom)
        parts = [p.strip().strip('.- ') for p in parts if p.strip()]
        return '.'.join(parts)
    def _sub(m: re.Match) -> str:
        return f"{m.group(1)}@{_norm_domain(m.group(2))}"
    return _OBFUSC_EMAIL_RE.sub(_sub, s)

def clean_text(text: str, *, remove_stopwords=False, stem=False, drop_long_tokens=30) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    text = _normalize_unicode(text)
    text = _strip_html_tags(text)
    text = _reconstruct_obfuscated_emails(text)
    text = _strip_reply_blocks(text)
    text = _URL_RE.sub(' <url> ', text)
    text = _EMAIL_RE.sub(' <email> ', text)
    text = _MONEY_RE.sub(' <money> ', text)
    text = _NUM_RE.sub(' <number> ', text)
    text = re.sub(r"[ \t]+", " ", text).strip().lower()
    toks = [t for t in re.split(r"\W+", text) if t]
    out = []
    for t in toks:
        if drop_long_tokens and len(t) > drop_long_tokens:
            continue
        if remove_stopwords and t in _STOPWORDS:
            continue
        if stem:
            for suf in ("ing","edly","ed","ly","es","s"):
                if t.endswith(suf) and len(t) > len(suf) + 2:
                    t = t[:-len(suf)]; break
        out.append(t)
    return " ".join(out)

# ================================================================
# Tokenization / vocab (+bigrams)
# ================================================================
def simple_tokenize(cleaned: str) -> List[str]:
    tokens = [t for t in re.split(r"\W+", cleaned) if t]
    bigrams = [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams

def build_vocab(texts: List[str], min_freq=2, max_size=60000) -> Dict[str,int]:
    PAD, UNK = "<pad>", "<unk>"
    freq: Dict[str,int] = {}
    for txt in texts:
        for tok in simple_tokenize(txt):
            freq[tok] = freq.get(tok, 0) + 1
    items = sorted([(t,c) for t,c in freq.items() if c >= min_freq], key=lambda x:(-x[1], x[0]))
    items = items[: max(0, max_size - 2)]
    vocab = {PAD:0, UNK:1}
    for t,_ in items:
        vocab[t] = len(vocab)
    return vocab

def encode_text(cleaned: str, vocab: Dict[str,int], max_len: int) -> List[int]:
    PAD_ID, UNK_ID = 0, 1
    toks = simple_tokenize(cleaned)[:max_len]
    ids = [vocab.get(t, UNK_ID) for t in toks]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids

# ================================================================
# Extra engineered features (cheap, high-signal)
# ================================================================
def extra_features_from_text(cleaned: str) -> List[float]:
    urls   = cleaned.count("<url>")
    emails = cleaned.count("<email>")
    money  = cleaned.count("<money>")
    nums   = cleaned.count("<number>")
    caps   = sum(1 for w in cleaned.split() if len(w)>=3 and w.isupper())
    return [urls, emails, money, nums, caps]

# ================================================================
# Dataset
# ================================================================
class TextDataset(Dataset):
    def __init__(self, df, vocab, label2id, max_len):
        self.vocab = vocab
        self.max_len = max_len
        self.label2id = label2id
        self.texts = df["body"].tolist()   # already cleaned
        self.labels = encode_labels(df["label"], label2id).astype(int).tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        ids = torch.tensor(encode_text(self.texts[i], self.vocab, self.max_len), dtype=torch.long)
        feats = torch.tensor(extra_features_from_text(self.texts[i]), dtype=torch.float)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return (ids, feats), y

# ================================================================
# DAN model
# ================================================================
class DanClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, n_classes=2, dropout=0.3, feat_dim=5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.ln = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim + feat_dim, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, n_classes)
        )

    def forward(self, x_ids, feats):
        emb = self.emb(x_ids)                      # [B, L, D]
        mask = (x_ids != 0).float()                # [B, L]
        lengths = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / lengths  # mean
        pooled = self.ln(pooled)
        h = torch.cat([pooled, feats], dim=1)
        logits = self.fc(self.dropout(h))
        return logits

# ================================================================
# Train / Eval
# ================================================================
def train_one_epoch(model, loader, criterion, optim, device, scheduler=None, grad_clip=1.0):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for (x_ids, feats), y in loader:
        x_ids, feats, y = x_ids.to(device), feats.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x_ids, feats)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        if scheduler is not None: scheduler.step()
        total_loss += loss.item() * x_ids.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x_ids.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, label_names) -> Tuple[float, str, float]:
    model.eval()
    ys, ps = [], []
    for (x_ids, feats), y in loader:
        x_ids, feats, y = x_ids.to(device), feats.to(device), y.to(device)
        logits = model(x_ids, feats)
        preds = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(preds.cpu().numpy().tolist())
    acc = (np.array(ys) == np.array(ps)).mean()
    # if "phishing" exists, use it as positive; else default to class 1
    pos_index = label_names.index("phishing") if "phishing" in label_names else 1
    f1 = f1_score(ys, ps, average="binary", pos_label=pos_index, zero_division=0)
    report = classification_report(ys, ps, target_names=label_names, digits=3, zero_division=0)
    return acc, report, f1

# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="DAN DL (phishing vs non-phishing) with cleaned CSV export.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", default="model_out_dan")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--remove_stopwords", action="store_true")
    ap.add_argument("--stem_tokens", action="store_true")
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load & clean ---
    raw_train = read_csv(args.train_csv)
    raw_test  = read_csv(args.test_csv)

    def apply_clean(df):
        df = df.copy()
        df["body"] = df["body"].apply(lambda s: clean_text(
            s, remove_stopwords=args.remove_stopwords, stem=args.stem_tokens
        ))
        return df

    train_df = apply_clean(raw_train)
    test_df  = apply_clean(raw_test)

    # Normalize labels
    train_df["label"] = normalize_label_series(train_df["label"])
    test_df["label"]  = normalize_label_series(test_df["label"])

    # Save cleaned CSVs so you can inspect
    clean_train_path = os.path.join(args.out_dir, "train_clean.csv")
    clean_test_path  = os.path.join(args.out_dir, "test_clean.csv")
    train_df[["body","label"]].to_csv(clean_train_path, index=False, encoding="utf-8")
    test_df[["body","label"]].to_csv(clean_test_path,  index=False, encoding="utf-8")
    print(f"Saved cleaned CSVs:\n  {clean_train_path}\n  {clean_test_path}")

    # Label maps (union)
    all_labels = pd.concat([train_df["label"], test_df["label"]], axis=0)
    label2id = build_label_map(all_labels)
    id2label = {v:k for k,v in label2id.items()}
    n_classes = len(label2id)
    label_names = [id2label[i] for i in range(n_classes)]
    print(f"Labels: {label2id}")

    # Stratified train/val
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=SEED)

    # Vocab from TRAIN ONLY
    vocab = build_vocab(train_df["body"].tolist(), min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab)}")

    # Datasets & loaders
    train_ds = TextDataset(train_df, vocab, label2id, max_len=args.max_len)
    val_ds   = TextDataset(val_df,   vocab, label2id, max_len=args.max_len)
    test_ds  = TextDataset(test_df,  vocab, label2id, max_len=args.max_len)

    # Weighted sampler for imbalance
    counts = train_df["label"].value_counts().to_dict()
    sample_weights = [1.0 / counts[lbl] for lbl in train_df["label"].tolist()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weights in loss
    cls_weights = torch.tensor([1.0 / counts[id2label[i]] for i in range(n_classes)], dtype=torch.float, device=device)

    model = DanClassifier(len(vocab), emb_dim=args.emb_dim, n_classes=n_classes, dropout=0.3, feat_dim=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # OneCycle LR (smooth warmup/cooldown)
    steps_per_epoch = max(1, len(train_ld))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs
    )

    # Train w/ early stopping on val F1
    best_f1, bad, patience = 0.0, 0, args.patience
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion, optimizer, device, scheduler=scheduler, grad_clip=1.0)
        val_acc, _, val_f1 = evaluate(model, val_ld, device, label_names)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_acc={val_acc:.3f} f1={val_f1:.3f}")

        if val_f1 >= best_f1:
            best_f1, bad = val_f1, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Load best & test
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "model.pt"), map_location=device))
    test_acc, test_report, test_f1 = evaluate(model, test_ld, device, label_names)
    print("\n=== Test report ===")
    print(test_report)
    print(f"Test accuracy: {test_acc:.3f} | Test F1: {test_f1:.3f}")

    # Save predictions CSV for inspection
    preds_path = os.path.join(args.out_dir, "test_predictions.csv")
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for (x_ids, feats), y in test_ld:
            x_ids, feats = x_ids.to(device), feats.to(device)
            logits = model(x_ids, feats)
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    inv = {i: label_names[i] for i in range(n_classes)}
    df_out = test_df.copy().reset_index(drop=True)
    df_out["gold"] = [inv[i] for i in encode_labels(test_df["label"], label2id)]
    df_out["pred"] = [inv[i] for i in ps]
    df_out.to_csv(preds_path, index=False, encoding="utf-8")
    print(f"Saved predictions to: {preds_path}")

    # Save artifacts
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({k:v for k,v in build_label_map(pd.concat([train_df["label"], test_df["label"]])).items()}, f)
    meta = {
        "max_len": args.max_len, "emb_dim": args.emb_dim, "min_freq": args.min_freq,
        "lr": args.lr, "epochs": args.epochs, "weight_decay": args.weight_decay,
        "patience": args.patience, "remove_stopwords": args.remove_stopwords, "stem_tokens": args.stem_tokens
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Artifacts saved in: {args.out_dir}")

if __name__ == "__main__":
    main()
