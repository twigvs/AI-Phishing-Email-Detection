
# src/data.py
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {'text','label'} <= set(df.columns), "CSV must have 'text' and 'label' columns"
    return df

def split_df(df: pd.DataFrame, test_size=0.1, val_size=0.1, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, test_size=test_size+val_size, stratify=df['label'], random_state=seed)
    rel_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=1-rel_val, stratify=temp_df['label'], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
