
# src/model.py
from transformers import RobertaForSequenceClassification

def build_model(num_labels: int = 2, model_name: str = "roberta-base"):
    return RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
