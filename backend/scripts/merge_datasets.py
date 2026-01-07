# synthetic + real → final CSV
import pandas as pd
from pathlib import Path

SYNTHETIC_PATH = Path("backend/data/processed/ipc_training_data_synthetic.csv")
REAL_PATH = Path("backend/data/processed/ipc_training_data_real.csv")  # optional
OUT_PATH = Path("backend/data/processed/ipc_training_data.csv")

dfs = []

if SYNTHETIC_PATH.exists():
    dfs.append(pd.read_csv(SYNTHETIC_PATH))

if REAL_PATH.exists():
    dfs.append(pd.read_csv(REAL_PATH))

if not dfs:
    raise ValueError("No datasets found to merge")

final_df = pd.concat(dfs, ignore_index=True)

# Remove duplicates & clean
final_df.drop_duplicates(subset=["case_text", "ipc_sections"], inplace=True)
final_df.dropna(inplace=True)

final_df.to_csv(OUT_PATH, index=False)

print(f"Final training dataset created: {len(final_df)} samples")
