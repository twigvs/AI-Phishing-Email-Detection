# api/app.py
import os
import io
from typing import List, Tuple
from flask import Flask, request, jsonify
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# ---- Config ----
MODEL_PATH = os.environ.get("MODEL_PATH", "roberta-base")  # set to your fine-tuned checkpoint path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- App ----
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# ---- Load model once ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# Build id<->label maps (fallback to binary phishing/legitimate if not present)
ID2LABEL = getattr(model.config, "id2label", None) or {0: "legitimate", 1: "phishing"}
LABEL2ID = {v: int(k) for k, v in getattr(model.config, "label2id", {v: k for k, v in ID2LABEL.items()}).items()}

REQUIRED_COLS = ["subject", "body", "label"]

def _concat_subject_body(subj: str, body: str) -> str:
    subj = "" if pd.isna(subj) else str(subj)
    body = "" if pd.isna(body) else str(body)
    return f"{subj}\n\n{body}".strip()

def _normalize_labels_to_ids(y: List) -> List[int]:
    """
    Accepts labels as ints or strings; converts to class IDs expected by the model.
    """
    out = []
    for v in y:
        if pd.isna(v):
            out.append(None)
            continue
        # try int
        try:
            out.append(int(v))
            continue
        except:
            pass
        # try string map
        s = str(v).strip()
        if s in LABEL2ID:
            out.append(LABEL2ID[s])
        elif s.lower() in {k.lower(): v for k, v in LABEL2ID.items()}:
            # case-insensitive match
            canonical = next(k for k in LABEL2ID if k.lower() == s.lower())
            out.append(LABEL2ID[canonical])
        else:
            out.append(None)
    return out

def _predict(texts: List[str], batch_size: int = 32) -> Tuple[List[int], List[float]]:
    preds, maxprobs = []
    preds, maxprobs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            probs = F.softmax(out.logits, dim=-1).cpu()
            pmax, pid = probs.max(dim=1)
            preds.extend(pid.tolist())
            maxprobs.extend(pmax.tolist())
    return preds, maxprobs

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "labels": ID2LABEL})

@app.route("/evaluate_csv", methods=["POST"])
def evaluate_csv():
    """
    Upload a CSV with columns: subject, body, label
    Returns JSON with metrics: accuracy, precision, recall, f1, confusion matrix, support
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Use form-data key 'file'."}), 400

    try:
        df = pd.read_csv(request.files['file'])
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    # Validate columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {', '.join(missing)}"}), 400

    # Prepare inputs
    texts = [_concat_subject_body(s, b) for s, b in zip(df["subject"], df["body"])]
    y_true_ids = _normalize_labels_to_ids(df["label"].tolist())

    # Filter rows with invalid labels
    valid_idx = [i for i, v in enumerate(y_true_ids) if v is not None]
    if not valid_idx:
        return jsonify({"error": "No valid labels after normalization. Ensure labels match model config (label2id) or are integers."}), 400

    texts_valid = [texts[i] for i in valid_idx]
    y_true_valid = [y_true_ids[i] for i in valid_idx]

    # Inference
    y_pred, conf = _predict(texts_valid, batch_size=int(request.form.get("batch_size", 32)))

    # Metrics
    labels_sorted = sorted(ID2LABEL.keys())  # numeric IDs
    acc = accuracy_score(y_true_valid, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true_valid, y_pred, labels=labels_sorted, zero_division=0)
    cm = confusion_matrix(y_true_valid, y_pred, labels=labels_sorted)

    # Misclassified examples (first 10)
    errors = []
    id2name = ID2LABEL
    for i, (t, yp, yt, pconf) in enumerate(zip(texts_valid, y_pred, y_true_valid, conf)):
        if yp != yt and len(errors) < 10:
            errors.append({
                "index_in_valid": i,
                "true": id2name[yt],
                "pred": id2name[yp],
                "confidence": round(float(pconf), 4),
                "preview": t[:300]
            })

    # Build per-class report
    per_class = []
    for i, cls_id in enumerate(labels_sorted):
        per_class.append({
            "class_id": int(cls_id),
            "class_name": id2name[cls_id],
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i])
        })

    result = {
        "n_rows": int(len(df)),
        "n_used_for_eval": int(len(texts_valid)),
        "labels": {int(k): v for k, v in id2name.items()},
        "accuracy": round(float(acc), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels_order": [id2name[i] for i in labels_sorted],
            "matrix": cm.tolist()
        },
        "examples_misclassified": errors
    }
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
