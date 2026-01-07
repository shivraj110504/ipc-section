# sections → ipc_sections.csv
import json
import pandas as pd
from pathlib import Path

IN_JSON = Path("backend/data/processed/ipc_sections.json")
OUT_CSV = Path("backend/data/processed/ipc_sections.csv")

with open(IN_JSON, "r", encoding="utf-8") as f:
    sections = json.load(f)

rows = []
for s in sections:
    rows.append({
        "section_number": s["section_number"],
        "section_title": s["title"],
        "section_text": s["text"].strip()
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("IPC sections CSV created")
