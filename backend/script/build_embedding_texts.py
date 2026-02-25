import json
from pathlib import Path
import re


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "ipc_enriched_v1.json"
EXPECTED_COUNT = 522


def _to_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def build_embedding_texts() -> list[dict[str, str]]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if len(dataset) != EXPECTED_COUNT:
        raise ValueError(
            f"Dataset length mismatch: expected {EXPECTED_COUNT}, got {len(dataset)}"
        )

    embedding_texts: list[dict[str, str]] = []

    for item in dataset:
        section_number = _to_text(item.get("section_number"))
        title = _to_text(item.get("title"))
        title = title.rstrip(".")
        summary = _to_text(item.get("summary"))
        summary = summary.rstrip(".")
        keywords_raw = item.get("keywords", [])

        if isinstance(keywords_raw, list):
            keywords = ", ".join(_to_text(keyword) for keyword in keywords_raw)
        else:
            keywords = _to_text(keywords_raw)

        embedding_text = (
            f"Section {section_number}: {title}. "
            f"Summary: {summary}. "
            f"Keywords: {keywords}."
        )

        embedding_texts.append(
            {
                "id": section_number,
                "embedding_text": embedding_text,
            }
        )

    if len(embedding_texts) != EXPECTED_COUNT:
        raise ValueError(
            f"Embedding texts length mismatch: expected {EXPECTED_COUNT}, got {len(embedding_texts)}"
        )

    return embedding_texts


def main() -> None:
    embedding_texts = build_embedding_texts()
    print("Total sections processed: 522")
    print("Sample embedding text (first item):")
    print(embedding_texts[0]["embedding_text"])


if __name__ == "__main__":
    main()
