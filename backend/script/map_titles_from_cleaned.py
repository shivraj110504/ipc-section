from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENRICHED_PATH = ROOT / "data" / "ipc_enriched_v1.json"
CLEANED_PATH = ROOT / "data" / "ipc_cleaned_v4.json"
OUTPUT_PATH = ROOT / "data" / "ipc_enriched_v1_final.json"
MISSING_LOG_PATH = ROOT / "data" / "missing_title_mapping.log"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    enriched = load_json(ENRICHED_PATH)
    cleaned = load_json(CLEANED_PATH)

    cleaned_title_by_section = {
        rec.get("section_number"): rec.get("section_title")
        for rec in cleaned
        if isinstance(rec, dict)
    }

    output_records = []
    unmatched_sections = []
    titles_added_count = 0

    for rec in enriched:
        if not isinstance(rec, dict):
            output_records.append(rec)
            continue

        sec = rec.get("section_number")
        mapped_title = cleaned_title_by_section.get(sec)

        new_rec = {}
        title_inserted = False

        for key, value in rec.items():
            new_rec[key] = value
            if key == "section_number":
                if mapped_title is not None:
                    new_rec["title"] = mapped_title
                    titles_added_count += 1
                else:
                    unmatched_sections.append(str(sec))
                title_inserted = True

        if not title_inserted:
            if mapped_title is not None:
                new_rec["title"] = mapped_title
                titles_added_count += 1
            else:
                unmatched_sections.append(str(sec))

        output_records.append(new_rec)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with MISSING_LOG_PATH.open("w", encoding="utf-8") as f:
        if unmatched_sections:
            for sec in unmatched_sections:
                f.write(f"{sec}\n")
        else:
            f.write("NONE\n")

    # validations
    total_records = len(enriched)
    final_total_records = len(output_records)

    unique_before = len({rec.get("section_number") for rec in enriched if isinstance(rec, dict)})
    unique_after = len({rec.get("section_number") for rec in output_records if isinstance(rec, dict)})

    required_fields = ["section_number", "title", "full_text", "summary", "keywords", "offence_type"]
    schema_valid = True
    for rec in output_records:
        if not isinstance(rec, dict):
            schema_valid = False
            break
        for field in required_fields:
            if field not in rec:
                schema_valid = False
                break
            if rec[field] is None:
                schema_valid = False
                break
            if isinstance(rec[field], str) and rec[field] == "":
                schema_valid = False
                break
        if not schema_valid:
            break

    no_other_fields_modified = True
    if len(enriched) != len(output_records):
        no_other_fields_modified = False
    else:
        for before, after in zip(enriched, output_records):
            if not isinstance(before, dict) or not isinstance(after, dict):
                if before != after:
                    no_other_fields_modified = False
                    break
                continue
            before_clone = dict(before)
            after_clone = dict(after)
            after_clone.pop("title", None)
            if before_clone != after_clone:
                no_other_fields_modified = False
                break

    report = {
        "total_records": final_total_records,
        "titles_added_count": titles_added_count,
        "unmatched_sections": unmatched_sections,
        "schema_valid": schema_valid,
        "no_other_fields_modified": no_other_fields_modified,
        "record_count_unchanged": total_records == final_total_records,
        "unique_section_count_unchanged": unique_before == unique_after,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
