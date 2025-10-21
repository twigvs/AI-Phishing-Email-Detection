# pip install torch pandas scikit-learn transformers
import argparse, json, re, os, random, html, unicodedata
from typing import List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

# ================================================================
# Utils
# ================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def read_csv(path):
    df = pd.read_csv(path, encoding="latin1")
    if "body" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: 'body' and 'label'")
    df = df[["body", "label"]].dropna()
    return df

def build_label_map(series):
    if pd.api.types.is_numeric_dtype(series):
        uniq = sorted(series.astype(int).unique().tolist())
    else:
        uniq = sorted(series.astype(str).unique().tolist())
    return {lbl: i for i, lbl in enumerate(uniq)}

def encode_labels(series, label2id):
    return series.map(
        lambda x: label2id[int(x)] if (isinstance(x, (int, np.integer)) and int(x) in label2id)
        else label2id[str(x)]
    )

def build_vocab(texts, min_freq=2, max_size=50000):
    PAD, UNK = "<pad>", "<unk>"
    freq = {}
    for txt in texts:
        for tok in simple_tokenize(txt):
            freq[tok] = freq.get(tok, 0) + 1
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
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids

def normalize_label_value(v):
    if isinstance(v, (int, np.integer, float, np.floating)):
        return "phishing" if int(v) != 0 else "legitimate"
    s = str(v).strip().lower()
    if s in {"phish","phishing","spam","malicious","fraud","scam","bad"}: return "phishing"
    if s in {"legit","legitimate","ham","benign","clean","safe","good"}: return "legitimate"
    if s in {"1","true","yes"}: return "phishing"
    if s in {"0","false","no"}: return "legitimate"
    return s

def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_label_value)

# =================================================================
# CLEANING LAYER
# =================================================================
_URL_RE   = re.compile(r'\b((?:https?://|www\.)\S+)\b', re.IGNORECASE)
_EMAIL_MASK_RE = re.compile(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b')
_PHONE_RE = re.compile(r'\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{2,4}\b')
_MONEY_RE = re.compile(r'(?:(?<=\s)|^)(?:[\$£€]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:usd|aud|eur|gbp))(?=\s|$)', re.IGNORECASE)
_NUM_RE   = re.compile(r'\b\d+(?:[\.,]\d+)?\b')

_AT_VARIANTS  = r'(?:@|\(at\)|\[at\]|\s+@\s+|\s*\(at\)\s*|\s*\[at\]\s*)'
_DOT_VARIANTS = r'(?:\.|\(dot\)|\[dot\]|\s*\.\s*|\s*\(dot\)\s*|\s*\[dot\]\s*|\s+dot\s+)'

_OBFUSC_EMAIL_RE = re.compile(
    r'\b([A-Za-z0-9._%+-]+)\s*' + _AT_VARIANTS + r'\s*' +
    r'([A-Za-z0-9-]+(?:\s*' + _DOT_VARIANTS + r'\s*[A-Za-z0-9-]+)*)'
    r'\b', re.IGNORECASE
)

_REPLY_SPLITTERS = [
    r'^>+ .*', r'^On .+ wrote:\s*$', r'^-{2,}\s*Original Message\s*-{2,}\s*$',
    r'^From:\s.*$', r'^Sent:\s.*$', r'^Subject:\s.*$',
]
_REPLY_SPLITTER_RE = re.compile('|'.join(_REPLY_SPLITTERS), re.IGNORECASE | re.MULTILINE)

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","with","by",
    "from","is","are","was","were","be","been","being","as","it","this","that","these","those",
    "i","you","he","she","we","they","me","him","her","them","my","your","our","their","so"
}

_BRAND_HINTS = {
    "paypal","apple","microsoft","google","outlook","office365","amazon",
    "anz","nab","commbank","westpac","cba","dhl","fedex","ups","bank",
    "netflix","facebook","instagram","linkedin","mygov","ato","irs"
}
_PATH_HINTS = {"login","verify","reset","password","invoice","update","confirm","secure"}

def _normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return ''.join(ch for ch in s if (ch.isprintable() or ch in '\n\t '))

def _strip_html_tags(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r'<\s*br\s*/?>', '\n', s, flags=re.IGNORECASE)
    s = re.sub(r'<[^>]+>', ' ', s)
    return s

def _reconstruct_obfuscated_emails(s: str) -> str:
    def _normalize_domain(dom: str) -> str:
        parts = re.split(_DOT_VARIANTS, dom)
        parts = [p.strip().strip('.- ') for p in parts if p.strip()]
        return '.'.join(parts)
    def _sub(m: re.Match) -> str:
        local = m.group(1)
        domain = _normalize_domain(m.group(2))
        return f'{local}@{domain}'
    return _OBFUSC_EMAIL_RE.sub(_sub, s)

def _replace_url_with_features(keep_numbers: bool):
    def inner(m: re.Match) -> str:
        url = m.group(1)
        try:
            p = urlparse(url if "://" in url else "http://" + url)
            host = (p.netloc or "").lower()
            path = (p.path or "").lower()
        except Exception:
            host, path = "", ""
        tokens = ["<url>"]
        if host:
            tokens.append(f"domain:{host}")
            parts = host.split(".")
            if len(parts) >= 2:
                tokens.append(f"tld:{parts[-1]}")
            for b in _BRAND_HINTS:
                if b in host:
                    tokens.append(f"brand:{b}")
        if path:
            for h in _PATH_HINTS:
                if f"/{h}" in path or path.startswith(h):
                    tokens.append(f"path:{h}")
        if keep_numbers:
            for m2 in re.finditer(r'\b\d{2,6}\b', url):
                tokens.append(f"numtok:{m2.group(0)}")
        return " " + " ".join(tokens) + " "
    return inner

def _mask_patterns(s: str, *, keep_numbers: bool) -> str:
    s = _reconstruct_obfuscated_emails(s)
    s = _URL_RE.sub(_replace_url_with_features(keep_numbers), s)
    s = _EMAIL_MASK_RE.sub(' <email> ', s)
    s = _MONEY_RE.sub(' <money> ', s)
    if not keep_numbers:
        s = _PHONE_RE.sub(' <phone> ', s)
        s = _NUM_RE.sub(' <number> ', s)
    return s

def _strip_reply_blocks(s: str) -> str:
    m = _REPLY_SPLITTER_RE.search(s)
    return s[:m.start()].strip() if m else s

def clean_text(
    text: str,
    *,
    remove_stopwords: bool = False,
    stem: bool = False,
    drop_long_tokens: int = 30,
    keep_numbers: bool = True,
    keep_reply_blocks: bool = False
) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    text = _normalize_unicode(text)
    text = _strip_html_tags(text)
    if not keep_reply_blocks:
        text = _strip_reply_blocks(text)
    text = _mask_patterns(text, keep_numbers=keep_numbers)
    text = text.lower()
    raw_tokens = [t for t in re.split(r"\W+", text) if t]
    pruned: List[str] = []
    for t in raw_tokens:
        if drop_long_tokens and len(t) > drop_long_tokens:
            continue
        if remove_stopwords and t in _STOPWORDS:
            continue
        if stem:
            for suf in ('ing','edly','ed','ly','es','s'):
                if t.endswith(suf) and len(t) > len(suf) + 2:
                    t = t[:-len(suf)]
                    break
        pruned.append(t)
    return ' '.join(pruned)

def simple_tokenize(text: str):
    return [t for t in re.split(r"\W+", text) if t]

# ================================================================
# Extra engineered features
# ================================================================
def extra_features_from_text(text: str) -> List[float]:
    urls   = text.count("<url>")
    emails = text.count("<email>")
    money  = text.count("<money>")
    nums   = text.count("<number>") + text.count("numtok:")
    caps   = sum(1 for w in text.split() if len(w) >= 3 and w.isupper())
    brands = sum(1 for w in text.split() if w.startswith("brand:"))
    paths  = sum(1 for w in text.split() if w.startswith("path:"))
    
    # Additional features
    length = len(text.split())
    avg_word_len = np.mean([len(w) for w in text.split()]) if text.split() else 0
    unique_ratio = len(set(text.split())) / max(len(text.split()), 1)
    
    return [urls, emails, money, nums, caps, brands, paths, length, avg_word_len, unique_ratio]

# ================================================================
# Dataset
# ================================================================
class TextDataset(Dataset):
    def __init__(self, df, vocab, label2id, max_len):
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len
        self.texts = df["body"].tolist()
        self.labels = encode_labels(df["label"], label2id).astype(int).tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        ids = torch.tensor(encode_text(self.texts[i], self.vocab, self.max_len), dtype=torch.long)
        feats = torch.tensor(extra_features_from_text(self.texts[i]), dtype=torch.float)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return (ids, feats), y

# ================================================================
# Enhanced DAN Model with multiple improvements
# ================================================================
class EnhancedDanClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, n_classes=2, dropout=0.4, feat_dim=10):
        super().__init__()
        # Larger embedding with better initialization
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.emb.weight[2:])  # Skip PAD and UNK
        
        # Multi-head attention for better pooling
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        
        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128)
        )
        
        # Deeper classifier with residual connections
        self.fc1 = nn.Linear(emb_dim + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout + 0.1)
        
    def forward(self, x_ids, feats):
        # Embedding
        emb = self.emb(x_ids)  # [B, L, D]
        
        # Create attention mask
        mask = (x_ids == 0)  # True for padding
        
        # Self-attention for contextual pooling
        attn_out, _ = self.self_attn(emb, emb, emb, key_padding_mask=mask)
        attn_out = self.ln1(attn_out + emb)  # Residual connection
        
        # Mean pooling (ignoring padding)
        pad_mask = (x_ids != 0).float().unsqueeze(-1)  # [B, L, 1]
        lengths = torch.clamp(pad_mask.sum(dim=1), min=1.0)
        mean_pooled = (attn_out * pad_mask).sum(dim=1) / lengths
        
        # Max pooling for salient features
        masked_attn = attn_out.masked_fill(mask.unsqueeze(-1), float('-inf'))
        max_pooled = masked_attn.max(dim=1)[0]
        max_pooled = torch.where(torch.isinf(max_pooled), torch.zeros_like(max_pooled), max_pooled)
        
        # Combine pooling strategies
        pooled = mean_pooled + 0.3 * max_pooled
        pooled = self.ln2(pooled)
        
        # Process extra features
        feat_emb = self.feat_proj(feats)
        
        # Concatenate
        h = torch.cat([pooled, feat_emb], dim=1)
        
        # Deep classifier with residual connections
        h1 = F.relu(self.fc1(self.dropout(h)))
        h2 = F.relu(self.fc2(self.dropout_heavy(h1)))
        h3 = F.relu(self.fc3(self.dropout(h2)))
        
        logits = self.fc_out(h3)
        return logits

# ================================================================
# Focal Loss for handling class imbalance
# ================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ================================================================
# Train / Eval with Label Smoothing
# ================================================================
def train_one_epoch(model, loader, criterion, optim, device, scheduler=None, max_grad_norm=1.0):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for (x_ids, feats), y in loader:
        x_ids, feats, y = x_ids.to(device), feats.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x_ids, feats)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * x_ids.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x_ids.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, label_names) -> Tuple[float, float, str]:
    model.eval()
    ys, ps = [], []
    for (x_ids, feats), y in loader:
        x_ids, feats, y = x_ids.to(device), feats.to(device), y.to(device)
        logits = model(x_ids, feats)
        preds = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(preds.cpu().numpy().tolist())
    macro_f1 = f1_score(ys, ps, average="macro")
    acc = (np.array(ys) == np.array(ps)).mean()
    report = classification_report(ys, ps, target_names=label_names, digits=4, zero_division=0)
    return acc, macro_f1, report

# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="Enhanced DAN with modern deep learning techniques.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--out_dir", default="model_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--emb_dim", type=int, default=512)
    ap.add_argument("--remove_stopwords", action="store_true")
    ap.add_argument("--stem_tokens", action="store_true")
    ap.add_argument("--keep_numbers", action="store_true")
    ap.add_argument("--keep_reply_blocks", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and clean
    raw_train_df = read_csv(args.train_csv)
    raw_test_df  = read_csv(args.test_csv)

    def apply_clean(df):
        df = df.copy()
        df["body"] = df["body"].apply(lambda s: clean_text(
            s,
            remove_stopwords=args.remove_stopwords,
            stem=args.stem_tokens,
            keep_numbers=args.keep_numbers,
            keep_reply_blocks=args.keep_reply_blocks
        ))
        return df

    train_df = apply_clean(raw_train_df)
    test_df  = apply_clean(raw_test_df)

    train_df["label"] = normalize_label_series(train_df["label"])
    test_df["label"]  = normalize_label_series(test_df["label"])

    clean_train_path = os.path.join(args.out_dir, "train_clean.csv")
    clean_test_path  = os.path.join(args.out_dir, "test_clean.csv")
    train_df[["body","label"]].to_csv(clean_train_path, index=False, encoding="utf-8")
    test_df[["body","label"]].to_csv(clean_test_path,  index=False, encoding="utf-8")
    print(f"Saved cleaned CSVs")

    all_labels = pd.concat([train_df["label"], test_df["label"]], axis=0)
    label2id = build_label_map(all_labels)
    id2label = {v: k for k, v in label2id.items()}
    n_classes = len(label2id)
    print(f"Labels: {label2id}")

    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
    )

    vocab = build_vocab(train_df["body"].tolist(), min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab)}")

    train_ds = TextDataset(train_df, vocab, label2id, max_len=args.max_len)
    val_ds   = TextDataset(val_df,   vocab, label2id, max_len=args.max_len)
    test_ds  = TextDataset(test_df,  vocab, label2id, max_len=args.max_len)

    tr_counts = train_df["label"].value_counts().to_dict()
    print("Train label counts:", tr_counts)

    # Class weights for focal loss
    weights = [1.0 / tr_counts[id2label[i]] for i in range(n_classes)]
    weights = torch.tensor(weights, dtype=torch.float, device=device)
    weights = weights / weights.sum() * n_classes  # Normalize

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = EnhancedDanClassifier(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        n_classes=n_classes,
        dropout=0.4,
        feat_dim=10
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Focal Loss
    criterion = FocalLoss(alpha=weights, gamma=args.focal_gamma)

    # AdamW with warmup
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler
    total_steps = len(train_ld) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-6)

    best_val_f1 = 0.0
    patience, bad = args.patience, 0

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion, optim, device, scheduler)
        val_acc, val_f1, _ = evaluate(model, val_ld, device, [id2label[i] for i in range(n_classes)])
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1; bad = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
            print(f"  ✓ Best model saved (val_f1={val_f1:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break

    # Final eval
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "model.pt"), map_location=device))
    acc, macro_f1, report = evaluate(model, test_ld, device, [id2label[i] for i in range(n_classes)])
    print("\n=== TEST SET REPORT ===")
    print(report)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

    # Save artifacts
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) if isinstance(v, (int, np.integer)) else v for k, v in label2id.items()}, f)
    meta = {
        "max_len": args.max_len, "emb_dim": args.emb_dim, "min_freq": args.min_freq,
        "remove_stopwords": args.remove_stopwords, "stem_tokens": args.stem_tokens,
        "keep_numbers": args.keep_numbers, "keep_reply_blocks": args.keep_reply_blocks,
        "lr": args.lr, "epochs": args.epochs, "weight_decay": args.weight_decay,
        "patience": args.patience, "focal_gamma": args.focal_gamma
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    
    print(f"\nAll artifacts saved to {args.out_dir}/")

if __name__ == "__main__":
    main()