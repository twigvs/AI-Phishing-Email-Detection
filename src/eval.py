
# src/eval.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(true_labels, pred_labels):
    print(classification_report(true_labels, pred_labels, digits=3))
    print("Confusion matrix:\n", confusion_matrix(true_labels, pred_labels))
