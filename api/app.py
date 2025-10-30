# api/app.py
import os
from typing import List, Tuple, Dict

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# --------- Configuration ---------

MODEL_PATH = os.environ.get("MODEL_PATH", "roberta-base")  # or your fine-tuned dir
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
REQUIRED_COLS = ["subject", "body", "label"]

# --------- App ---------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# ---------- Load Model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


def _normalize_id2label_from_config(cfg) -> Dict[int, str]:
    """
    Build a robust {id:int -> label:str} map from HF config, even if keys
    are strings like 'LABEL_0' or values vary in casing.
    Falls back to {0:'legitimate', 1:'phishing'} if nothing present.
    """
    # Prefer config.id2label first
    id2 = getattr(cfg, "id2label", None)
    fixed: Dict[int, str] = {}

    if id2:
        for k, v in id2.items():
            # k might be int, "0", or "LABEL_0"
            if isinstance(k, int):
                fixed[k] = str(v)
            else:
                ks = str(k)
                if ks.isdigit():
                    fixed[int(ks)] = str(v)
                elif "LABEL_" in ks:
                    try:
                        fixed[int(ks.split("_")[-1])] = str(v)
                    except Exception:
                        pass
        if fixed:
            return fixed

    # Else try label2id
    l2i = getattr(cfg, "label2id", None)
    if l2i:
        for lbl, idx in l2i.items():
            try:
                i = int(idx) if isinstance(idx, (int, str)) else idx
                fixed[i] = str(lbl)
            except Exception:
                pass
        if fixed:
            return fixed

    # Default binary
    return {0: "Legitimate", 1: "Phishing"}

# Human readable labelling, instead of presenting a 1 or a 0
ID2LABEL: Dict[int, str] = {0: "Legitimate", 1: "Phishing"}
LABEL2ID: Dict[str, int] = {"Legitimate": 0, "Phishing": 1}
LABEL2ID_CI: Dict[str, int] = {"Legitimate": 0, "Phishing": 1}

# Which class index corresponds to "phishing" (used for probability reporting)
PHISH_IDX = next(
    (i for i, name in ID2LABEL.items() if str(name).lower() in {"phishing", "phish", "spam"}),
    1 if 1 in ID2LABEL else sorted(ID2LABEL.keys())[-1],
)


# -------------- Utlilities --------------
def _concat_subject_body(subj: str, body: str) -> str:
    subj = "" if pd.isna(subj) else str(subj)
    body = "" if pd.isna(body) else str(body)
    return f"{subj}\n\n{body}".strip()


def _normalize_labels_to_ids(y: List) -> List[int]:
    """
    Accept labels as ints or strings; convert to class IDs expected by the model.
    Unknown values -> None.
    """
    out = []
    for v in y:
        if pd.isna(v):
            out.append(None)
            continue
        # try int directly
        try:
            out.append(int(v))
            continue
        except Exception:
            pass
        # try string (case-insensitive)
        s = str(v).strip()
        lid = LABEL2ID.get(s)
        if lid is not None:
            out.append(lid)
            continue
        lid = LABEL2ID_CI.get(s.lower())
        if lid is not None:
            out.append(lid)
            continue
        out.append(None)
    return out


def _predict(texts: List[str], batch_size: int = 32) -> Tuple[List[int], List[float], List[List[float]]]:
    """
    Returns:
      preds:      List[int]      argmax class ids
      maxprobs:   List[float]    max softmax prob per row
      all_probs:  List[List[float]] prob vector per row (len = num_labels)
    """
    preds, maxprobs, all_probs = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits  # [B, C] or [B] if single-logit (rare)
            if logits.ndim == 1:
                # Single logit -> sigmoid; fabricate a 2-class view [legit, phish]
                p = torch.sigmoid(logits).unsqueeze(1)          # [B,1]
                probs = torch.cat([1 - p, p], dim=1)            # [B,2]
            else:
                probs = F.softmax(logits, dim=-1)

        probs_cpu = probs.detach().cpu().numpy()
        pmax = probs_cpu.max(axis=1)
        pid = probs_cpu.argmax(axis=1)

        preds.extend(pid.tolist())
        maxprobs.extend(pmax.astype(float).tolist())
        all_probs.extend(probs_cpu.astype(float).tolist())
    return preds, maxprobs, all_probs

# ------------- Routing -------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "labels": {int(k): v for k, v in ID2LABEL.items()},
            "model_path": MODEL_PATH,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts:
      - {"text": "..."}  OR
      - {"subject": "...", "body": "..."}
    Returns:
      {"label": "<readable label>", "score": <probability of phishing>}
    """
    data = request.get_json(silent=True, force=True) or {}
    if "text" in data and isinstance(data["text"], str):
        text = data["text"]
    else:
        subject = data.get("subject", "") or ""
        body = data.get("body", "") or ""
        text = _concat_subject_body(subject, body)

    if not text.strip():
        return jsonify({"error": "no text provided"}), 400

    # Run single prediction
    preds, _, all_probs = _predict([text], batch_size=1)
    pred_id = int(preds[0])

    # Probability of phishing (by PHISH_IDX)
    probs_vec = all_probs[0]
    # Guard in case model num_labels < = PHISH_IDX
    if PHISH_IDX < len(probs_vec):
        phish_prob = float(probs_vec[PHISH_IDX])
    else:
        # fallback: if only one prob or mismatch, use max prob when predicted phishing else 1-max
        maxp = float(max(probs_vec)) if probs_vec else 0.5
        phish_prob = maxp if pred_id == PHISH_IDX else (1.0 - maxp)

    label = ID2LABEL.get(pred_id, "phishing" if pred_id == PHISH_IDX else "legitimate")
    return jsonify({"label": label, "score": phish_prob}), 200


@app.route("/evaluate_csv", methods=["POST"])
def evaluate_csv():
    """
    Upload a CSV (multipart/form-data) with columns: subject, body, label
    Returns JSON with metrics: accuracy, per-class PRF, confusion matrix, examples
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use form-data key 'file'."}), 400

    try:
        df = pd.read_csv(request.files["file"])
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
        return jsonify(
            {
                "error": "No valid labels after normalization. Ensure labels match model config (label2id) or are integers."
            }
        ), 400

    texts_valid = [texts[i] for i in valid_idx]
    y_true_valid = [y_true_ids[i] for i in valid_idx]

    # Inference
    batch_size = 32
    try:
        if "batch_size" in request.form:
            batch_size = max(1, int(request.form.get("batch_size", 32)))
    except Exception:
        pass

    y_pred, _, all_probs = _predict(texts_valid, batch_size=batch_size)

    # Metrics
    labels_sorted = sorted(ID2LABEL.keys())  # numeric IDs
    acc = accuracy_score(y_true_valid, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_valid, y_pred, labels=labels_sorted, zero_division=0
    )
    cm = confusion_matrix(y_true_valid, y_pred, labels=labels_sorted)

    # Misclassified examples (first 10)
    errors = []
    for idx_in_valid, (t, yp, yt, probs) in enumerate(zip(texts_valid, y_pred, y_true_valid, all_probs)):
        if yp != yt and len(errors) < 10:
            errors.append(
                {
                    "index_in_valid": idx_in_valid,
                    "true": ID2LABEL.get(yt, str(yt)),
                    "pred": ID2LABEL.get(yp, str(yp)),
                    "confidence_pred": float(max(probs)) if probs else None,
                    "preview": t[:300],
                }
            )

    # Per-class report
    per_class = []
    for i, cls_id in enumerate(labels_sorted):
        per_class.append(
            {
                "class_id": int(cls_id),
                "class_name": ID2LABEL.get(cls_id, str(cls_id)),
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }
        )

    result = {
        "n_rows": int(len(df)),
        "n_used_for_eval": int(len(texts_valid)),
        "labels": {int(k): v for k, v in ID2LABEL.items()},
        "accuracy": round(float(acc), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels_order": [ID2LABEL[i] for i in labels_sorted],
            "matrix": cm.tolist(),
        },
        "examples_misclassified": errors,
    }
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)