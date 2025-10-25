"""
PySimpleGUI frontend for AI-Phishing-Email-Detection repo.
Features:
 - Enter Subject + Body (or load CSV) and get a phishing probability + label, should say phishing or legitimate to be user-friendly.
 - Two inference modes:
    1) API mode: POST to http://localhost:5000/predict (expects JSON {subject, body})
    2) Local mode: either HuggingFace transformer model or sklearn pickle + vectorizer
 - Simple logging and CSV batch predict save

 First please run app.py on a separate PowerShell window. This enables the api.
 Then run pygui.py to launch the UI. Please note that CSV may take some time to process, or even may crash if the file is too large.
 Enjoy :)
"""

import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict

import PySimpleGUI as sg
import requests
import pandas as pd
import numpy as np

# Optional heavy dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    import pickle
    from sklearn.base import BaseEstimator
except Exception:
    pickle = None
    BaseEstimator = None

# ---------- Configuration with app.py ----------
API_URL = "http://localhost:5000/predict"     # API mode endpoint, will reference app.py, which must be running on a seperate process
DEFAULT_TRANSFORMER_PATH = "models/roberta"   # local HF model dir if present
DEFAULT_SKLEARN_PICKLE = "models/model.pkl"   # fallback sklearn model path
DEFAULT_VECTORIZER_PICKLE = "models/vectorizer.pkl"  # if using sklearn
# ---------------------------------------------------

sg.theme("DefaultNoMoreNagging")

layout = [
    [sg.Text("AI Phishing Email Detector", font=("Montserrat", 18, "bold"))],
    [sg.Frame("Suspected Phishing Input", [
        [sg.Text("Subject", size=(8,1)), sg.Input(key="-SUBJECT-", expand_x=True)],
        [sg.Text("Body", size=(8,1)), sg.Multiline(key="-BODY-", size=(80,12))],
        [sg.Text("Or load a CSV (subject,body columns):"), sg.Input(key="-CSVPATH-", enable_events=True, visible=False),
         sg.FileBrowse("Load CSV", target="-CSVPATH-", file_types=(("CSV Files","*.csv"),))],
    ])],
    [sg.Frame("Choose a Mode & Model", [
        [sg.Radio("API mode (POST to running service)", "MODE", default=True, key="-MODE_API-"),
         sg.Radio("Local transformer model (HF)", "MODE", key="-MODE_HF-"),
         sg.Radio("Local sklearn pickle", "MODE", key="-MODE_SK-")],
        [sg.Text("Transformer dir / model name:"), sg.Input(DEFAULT_TRANSFORMER_PATH, key="-HF_PATH-"),
         sg.Button("Load HF", key="-LOAD_HF-")],
        [sg.Text("Sklearn model pickle:"), sg.Input(DEFAULT_SKLEARN_PICKLE, key="-SK_PATH-"),
         sg.Button("Load SK", key="-LOAD_SK-")],
    ], expand_x=True)],
    [sg.Button("Predict Single", key="-PREDICT-"), sg.Button("Batch Predict CSV", key="-PREDICT_CSV-"),
     sg.Button("Clear"), sg.Button("Exit")],
    [sg.Frame("Output", [
        [sg.Text("Label:", size=(10,1)), sg.Text("", key="-LABEL-", font=("Montserrat",12,"bold"))],
        [sg.Text("Score (prob. 0-1):", size=(18,1)), sg.Text("", key="-SCORE-")],
        [sg.Text("Model Details:"), sg.Text("", key="-MODELINFO-")],
        [sg.Multiline(key="-LOG-", size=(100,8), disabled=True)]
    ])]
]

window = sg.Window("AI Phishing Email Detector - Group 10 / ICT30016", layout, resizable=True, finalize=True)

# Globals for local AI models
hf_tokenizer = None
hf_model = None
sk_model = None
sk_vectorizer = None
model_info_text = ""

# ---------- Helpers ----------

def log(msg: str):
    current = window["-LOG-"].get()
    window["-LOG-"].update(current + msg + "\n")

def try_api_predict(subject: str, body: str, api_url=API_URL) -> Dict:
    """POST JSON to API and return parsed response expected as JSON {'label':..., 'score':...}"""
    payload = {"subject": subject, "body": body}
    try:
        r = requests.post(api_url, json=payload, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}")

def load_hf_model(dir_or_name: str):
    global hf_tokenizer, hf_model, model_info_text
    if AutoTokenizer is None:
        raise RuntimeError("transformers/torch not installed in this environment.")
    log(f"Loading HF model from: {dir_or_name} ...")
    hf_tokenizer = AutoTokenizer.from_pretrained(dir_or_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(dir_or_name)
    hf_model.eval()
    model_info_text = f"HF model: {dir_or_name}"
    window["-MODELINFO-"].update(model_info_text)
    log("HF model loaded.")

def hf_predict(subject: str, body: str) -> Dict:
    """Run inference with HF transformer; returns {'label','score'} where score is phishing prob"""
    if hf_tokenizer is None or hf_model is None:
        raise RuntimeError("HF model not loaded.")
    text = (subject or "") + "\n\n" + (body or "")
    enc = hf_tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = hf_model(**enc)
        logits = out.logits.squeeze(0).cpu().numpy()
        # Assume binary classification: [logit_legit, logit_phish] or single logit
        if logits.shape == (2,):
            # softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            phish_prob = float(probs[1])
            label = "phishing" if phish_prob > 0.5 else "legitimate"
        else:
            # single logit -> sigmoid
            phish_prob = float(1 / (1 + np.exp(-logits)))
            label = "phishing" if phish_prob > 0.5 else "legitimate"
    return {"label": label, "score": phish_prob}

def load_sk_model(model_path: str, vect_path: Optional[str] = None):
    global sk_model, sk_vectorizer, model_info_text
    log(f"Loading SK model from: {model_path}")
    with open(model_path, "rb") as f:
        sk_model = pickle.load(f)
    if vect_path and Path(vect_path).exists():
        with open(vect_path, "rb") as f:
            sk_vectorizer = pickle.load(f)
    model_info_text = f"SK model: {model_path} (vectorizer: {vect_path if sk_vectorizer is not None else 'none'})"
    window["-MODELINFO-"].update(model_info_text)
    log("SK model loaded.")

def sk_predict(subject: str, body: str) -> Dict:
    if sk_model is None:
        raise RuntimeError("Sklearn model not loaded.")
    text = (subject or "") + " " + (body or "")
    if sk_vectorizer is not None:
        X = sk_vectorizer.transform([text])
    else:
        # if model expects raw text, attempt to predict directly
        X = [text]
    # try predict_proba if available
    if hasattr(sk_model, "predict_proba"):
        probs = sk_model.predict_proba(X)
        # phishing class is labelled 1 or 'phishing', find index
        classes = getattr(sk_model, "classes_", None)
        if classes is not None:
            try:
                idx = list(classes).index(1)
            except Exception:
                # fallback: assume second column is phishing
                idx = 1 if probs.shape[1] > 1 else 0
        else:
            idx = 1 if probs.shape[1] > 1 else 0
        phish_prob = float(probs[0, idx])
    else:
        pred = sk_model.predict(X)[0]
        phish_prob = 1.0 if pred in (1, "phishing", "phish") else 0.0
    label = "phishing" if phish_prob > 0.5 else "legitimate"
    return {"label": label, "score": phish_prob}

# ---------- Event loop ----------
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    if event == "Clear":
        window["-SUBJECT-"].update("")
        window["-BODY-"].update("")
        window["-LABEL-"].update("")
        window["-SCORE-"].update("")
        window["-LOG-"].update("")
        continue

    if event == "-LOAD_HF-":
        path = values["-HF_PATH-"]
        try:
            load_hf_model(path)
        except Exception as e:
            log("HF load error: " + str(e))
            log(traceback.format_exc())

    if event == "-LOAD_SK-":
        path = values["-SK_PATH-"]
        vect = DEFAULT_VECTORIZER_PICKLE
        try:
            load_sk_model(path, vect)
        except Exception as e:
            log("SK load error: " + str(e))
            log(traceback.format_exc())

    if event == "-PREDICT-":
        subject = values["-SUBJECT-"]
        body = values["-BODY-"]
        # Run prediction in background to keep GUI responsive
        def do_predict():
            try:
                if values["-MODE_API-"]:
                    resp = try_api_predict(subject, body, API_URL)
                    lbl = resp.get("label", "unknown")
                    score = resp.get("score", resp.get("prob", None))
                elif values["-MODE_HF-"]:
                    resp = hf_predict(subject, body)
                    lbl, score = resp["label"], resp["score"]
                else:
                    resp = sk_predict(subject, body)
                    lbl, score = resp["label"], resp["score"]
                window.write_event_value("-PRED_DONE-", (lbl, score))
            except Exception as e:
                window.write_event_value("-PRED_ERROR-", str(e))

        threading.Thread(target=do_predict, daemon=True).start()
        log("Prediction started...")

    if event == "-PRED_DONE-":
        lbl, score = values[event]
        window["-LABEL-"].update(lbl)
        window["-SCORE-"].update(f"{score:.4f}")
        log(f"Result: {lbl} ({score:.4f})")

    if event == "-PRED_ERROR-":
        err = values[event]
        log("Prediction error: " + str(err))

    if event == "-PREDICT_CSV-":
        csv_path = values["-CSVPATH-"]
        if not csv_path:
            sg.popup("Please load a CSV first (must have 'subject' and 'body' columns).")
            continue
        try:
            df = pd.read_csv(csv_path)
            if "subject" not in df.columns or "body" not in df.columns:
                sg.popup("CSV must have 'subject' and 'body' columns.")
                continue
            results = []
            for _, row in df.iterrows():
                s = str(row.get("subject", "") or "")
                b = str(row.get("body", "") or "")
                if values["-MODE_API-"]:
                    r = try_api_predict(s, b, API_URL)
                    label, score = r.get("label",""), r.get("score",0.0)
                elif values["-MODE_HF-"]:
                    r = hf_predict(s, b)
                    label, score = r["label"], r["score"]
                else:
                    r = sk_predict(s, b)
                    label, score = r["label"], r["score"]
                results.append({"subject": s, "body": b, "label": label, "score": score})
            out_df = pd.DataFrame(results)
            out_name = Path(csv_path).with_suffix(".predictions.csv")
            out_df.to_csv(out_name, index=False)
            sg.popup(f"Batch predictions saved to {out_name}")
            log(f"Batch done, saved to {out_name}")
        except Exception as e:
            log("Batch error: " + str(e))
            log(traceback.format_exc())

window.close()