import pandas as pd
import os

filepath = 'c:/Users/adeL/Desktop/GoodBot/Ver2/history_cache/XAUUSDm_W1.pkl'
if os.path.exists(filepath):
    df = pd.read_pickle(filepath)
    print(f"Total rows: {len(df)}")
    if not df.empty:
        print(f"Start: {df.index[0]}")
        print(f"End: {df.index[-1]}")
else:
    print(f"File not found: {filepath}")
