# IPC PDF → ipc_pages.json
import pdfplumber
import json
from pathlib import Path

RAW_PDF = Path("backend/data/raw/IPC_186045.pdf")
OUT_JSON = Path("backend/data/processed/ipc_pages.json")

pages_data = []

with pdfplumber.open(RAW_PDF) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            pages_data.append({
                "page_number": i + 1,
                "text": text.strip()
            })

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(pages_data, f, indent=2, ensure_ascii=False)

print("IPC PDF converted to page-wise JSON")
