
# tests/test_data_loading.py
import pandas as pd
from src.data import split_df

def test_split_shapes():
    df = pd.DataFrame({'text': ['a']*100, 'label': [0]*50 + [1]*50})
    tr, va, te = split_df(df, test_size=0.1, val_size=0.1, seed=1)
    assert len(tr) == 80 and len(va) == 10 and len(te) == 10
