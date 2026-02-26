import json
import os
from pathlib import Path
from typing import Any

import chromadb
import requests


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def _check_keys():
    print(f"--- API KEY CHECK: OPENROUTER_API_KEY present? {'Yes' if OPENROUTER_API_KEY else 'No'} ---")
    if OPENROUTER_API_KEY:
        print(f"--- API KEY CHECK: OPENROUTER_API_KEY starts with: {OPENROUTER_API_KEY[:4]}... ---")

MODEL = "openai/text-embedding-3-small"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
PERSIST_DIRECTORY = "./chroma_ipc_v1"
COLLECTION_NAME = "ipc_sections_v1"
TOP_K = 7


def _resolve_persist_directory() -> str:
    configured = Path(PERSIST_DIRECTORY)
    local_to_file = Path(__file__).resolve().parent / PERSIST_DIRECTORY
    workspace_root = Path(__file__).resolve().parent.parent / PERSIST_DIRECTORY.lstrip("./")
    script_folder = (
        Path(__file__).resolve().parent.parent / "script" / PERSIST_DIRECTORY.lstrip("./")
    )

    candidates = [configured, local_to_file, workspace_root, script_folder]
    for candidate in candidates:
        if candidate.exists():
            print(f"--- DATABASE FOUND AT: {candidate} ---")
            return str(candidate)
    
    print(f"--- WARNING: DATABASE NOT FOUND IN CANDIDATES. FALLING BACK TO: {candidates[0]} ---")
    return str(candidates[0])


def _embed_text(text: str) -> list[float]:
    _check_keys()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "input": text,
    }
    print(f"--- CALLING OPENROUTER EMBEDDING: {MODEL} ---")
    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        print(f"--- OPENROUTER API ERROR: {response.status_code} - {response.text} ---")
        response.raise_for_status()
    
    body = response.json()
    if "data" not in body or not body["data"]:
        print(f"--- OPENROUTER UNEXPECTED RESPONSE: {body} ---")
        raise ValueError("Invalid embedding response")

    print(f"--- EMBEDDING SUCCESSFUL ---")
    return body["data"][0]["embedding"]


def _section_sort_key(section_number: str) -> tuple[int, str]:
    section_number = str(section_number).strip()
    digits = ""
    suffix = ""
    for char in section_number:
        if char.isdigit() and suffix == "":
            digits += char
        else:
            suffix += char
    numeric_part = int(digits) if digits else 0
    return numeric_part, suffix


def _normalize_keywords(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return value
    return value


def _format_result(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "section_number": str(metadata.get("section_number", "")),
        "title": metadata.get("title", ""),
        "summary": metadata.get("summary", ""),
        "keywords": _normalize_keywords(metadata.get("keywords", [])),
        "full_text": metadata.get("full_text", ""),
        "offence_type": metadata.get("offence_type", ""),
    }


import traceback

def _keyword_search_fallback(text: str) -> list[tuple[dict[str, Any], float]]:
    print("--- KEYWORD SEARCH FALLBACK STARTING ---")
    try:
        # Resolve dataset path
        dataset_path = Path(__file__).resolve().parent.parent / "data" / "ipc_enriched_v1.json"
        
        if not dataset_path.exists():
            print(f"--- FALLBACK ERROR: DATASET NOT FOUND AT {dataset_path} ---")
            return []
            
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        query_terms = [t.lower() for t in text.split() if len(t) > 2]
        if not query_terms:
            query_terms = [text.lower().strip()]
            
        scored_results = []
        for item in dataset:
            # Check title, summary, and keywords
            search_blob = f"{item.get('title', '')} {item.get('summary', '')} {' '.join(item.get('keywords', []))}".lower()
            
            # Simple match count
            match_count = sum(1 for term in query_terms if term in search_blob)
            
            if match_count > 0:
                # Score is based on match density
                score = 0.5 + (min(match_count / max(len(query_terms), 1), 1.0) * 0.4)
                scored_results.append((_format_result(item), score))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        print(f"--- KEYWORD SEARCH FOUND {len(scored_results)} RESULTS ---")
        return scored_results[:TOP_K]
        
    except Exception as e:
        print(f"--- KEYWORD SEARCH FATAL ERROR: {str(e)} ---")
        traceback.print_exc()
        return []


def _retrieve_with_scores(incident_text: str) -> list[tuple[dict[str, Any], float]]:
    try:
        persist_dir = _resolve_persist_directory()
        print(f"--- ATTEMPTING CHROMA INIT AT {persist_dir} ---")
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(name=COLLECTION_NAME)

        if incident_text.strip() == "":
            all_rows = collection.get(include=["metadatas"])
            metadatas = all_rows.get("metadatas", [])
            ordered = sorted(
                metadatas,
                key=lambda row: _section_sort_key(str(row.get("section_number", ""))),
            )
            top_rows = ordered[:TOP_K]
            return [(_format_result(row), 0.0) for row in top_rows]

        query_embedding = _embed_text(incident_text)
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["embeddings", "metadatas", "distances"],
        )

        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        print(f"--- DATABASE: FOUND {len(metadatas)} CANDIDATES ---")

        rows: list[tuple[dict[str, Any], float]] = []
        for metadata, distance in zip(metadatas, distances):
            similarity = 1.0 - float(distance)
            rows.append((_format_result(metadata), similarity))

        rows.sort(
            key=lambda row: (
                -row[1],
                _section_sort_key(str(row[0].get("section_number", ""))),
            )
        )
        return rows[:TOP_K]
    except Exception as e:
        print(f"--- CHROMA ERROR DETECTED: {str(e)} ---")
        traceback.print_exc()
        print("--- SWITCHING TO FAILSAFE KEYWORD SEARCH ---")
        return _keyword_search_fallback(incident_text)


def retrieve_sections(incident_text: str) -> list[dict]:
    ranked = _retrieve_with_scores(incident_text)
    return [item for item, _ in ranked]


def _test_determinism() -> None:
    deterministic_query = (
        "A person entered another person's home at night and stole cash and jewelry."
    )
    outputs = [retrieve_sections(deterministic_query) for _ in range(5)]
    assert all(output == outputs[0] for output in outputs), "Non-deterministic retrieval detected"


def _test_edge_cases() -> None:
    edge_cases = [
        "",
        "   ",
        (
            "The accused repeatedly issued threats over several months, forced entry into a property, "
            "caused physical injury during confrontation, and removed financial documents and valuables "
            "without consent, while witnesses observed intimidation, damage to property, and attempted "
            "destruction of records before law enforcement intervention."
        ),
        "1234567890 987654321",
    ]

    for case in edge_cases:
        ranked = _retrieve_with_scores(case)
        assert len(ranked) == TOP_K, f"Output length is not {TOP_K}"
        print("Top 5 sections:")
        print([row[0]["section_number"] for row in ranked])
        print([row[1] for row in ranked])


def main() -> None:
    _test_determinism()
    _test_edge_cases()


if __name__ == "__main__":
    main()
