import pandas as pd
import os

files = [
    '/Users/jluque/.gemini/antigravity/scratch/house_flipping/backend/data_ROI.xlsx',
    '/Users/jluque/.gemini/antigravity/scratch/house_flipping/backend/data/data_ROI_Valencia.xlsx'
]

for f in files:
    if os.path.exists(f):
        print(f"--- Headers for {os.path.basename(f)} ---")
        try:
            df = pd.read_excel(f, nrows=0)
            print(list(df.columns))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
