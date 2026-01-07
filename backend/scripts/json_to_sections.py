# pages → ipc_sections.json
import json
import re
from pathlib import Path

IN_JSON = Path("backend/data/processed/ipc_pages.json")
OUT_JSON = Path("backend/data/processed/ipc_sections.json")

with open(IN_JSON, "r", encoding="utf-8") as f:
    pages = json.load(f)

sections = []
current = None

SECTION_REGEX = re.compile(r'^(\d{1,3}[A-Z]?)\.\s+(.*)')

for page in pages:
    for line in page["text"].split("\n"):
        line = line.strip()
        match = SECTION_REGEX.match(line)

        if match:
            if current:
                sections.append(current)

            current = {
                "section_number": match.group(1),
                "title": match.group(2),
                "text": ""
            }
        elif current:
            current["text"] += line + " "

if current:
    sections.append(current)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(sections, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(sections)} IPC sections")
