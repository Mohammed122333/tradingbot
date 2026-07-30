import os
import pandas as pd
import glob

history_dir = r"C:\Users\adeL\Desktop\GoodBot\Ver2\QuantHistory"
files = glob.glob(os.path.join(history_dir, "XAUUSDm_*.csv"))

SPREAD_VALUE = 240.0

for file_path in files:
    print(f"Updating {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    if 'spread' in df.columns:
        df['spread'] = SPREAD_VALUE
        df.to_csv(file_path, index=False)
        print(f"  -> Successfully updated spread to {SPREAD_VALUE}")
    else:
        print(f"  -> WARNING: No 'spread' column found in {os.path.basename(file_path)}")
        
print("All done!")
