import json
import os
from pathlib import Path

import chromadb
import requests

from build_embedding_texts import build_embedding_texts, DATASET_PATH


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
MODEL = "openai/text-embedding-3-small"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

PERSIST_DIRECTORY = "./chroma_ipc_v1"
COLLECTION_NAME = "ipc_sections_v1"

EXPECTED_COUNT = 522


def load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_embedding(text: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "input": text,
    }
    response = requests.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


def main() -> None:
    # Load embedding texts
    embedding_texts = build_embedding_texts()
    assert len(embedding_texts) == EXPECTED_COUNT, (
        f"Expected {EXPECTED_COUNT} embedding texts, got {len(embedding_texts)}"
    )

    # Load original dataset for metadata
    dataset = load_dataset()
    assert len(dataset) == EXPECTED_COUNT, (
        f"Expected {EXPECTED_COUNT} dataset items, got {len(dataset)}"
    )

    # Build section_number -> metadata mapping
    metadata_map = {str(item["section_number"]): item for item in dataset}

    # Generate embeddings (maintain original ordering)
    vectors: list[list[float]] = []
    for et in embedding_texts:
        embedding = generate_embedding(et["embedding_text"])
        vectors.append(embedding)

    assert len(vectors) == EXPECTED_COUNT, (
        f"Expected {EXPECTED_COUNT} vectors, got {len(vectors)}"
    )

    # Initialize ChromaDB persistent client
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    # Delete existing collection if exists to avoid duplicate IDs
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    # Create collection
    collection = client.create_collection(name=COLLECTION_NAME)

    # Prepare data for insertion
    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for i, et in enumerate(embedding_texts):
        section_id = et["id"]
        ids.append(section_id)
        embeddings.append(vectors[i])

        original = metadata_map[section_id]
        metadatas.append({
            "section_number": str(original.get("section_number", "")),
            "title": str(original.get("title", "")),
            "summary": str(original.get("summary", "")),
            "keywords": json.dumps(original.get("keywords", [])),
            "full_text": str(original.get("full_text", "")),
            "offence_type": str(original.get("offence_type", "")),
        })

    # Insert into collection
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    # Post-insert validation
    assert collection.count() == EXPECTED_COUNT, (
        f"Expected {EXPECTED_COUNT} items in collection, got {collection.count()}"
    )

    # Fetch 1 sample record for validation
    sample = collection.get(ids=[ids[0]], include=["embeddings", "metadatas"])
    assert len(sample["embeddings"][0]) > 0, "Sample embedding is empty"
    required_fields = ["section_number", "title", "summary", "keywords", "full_text", "offence_type"]
    assert all(key in sample["metadatas"][0] for key in required_fields), "Metadata fields missing"

    # Print exact output
    vector_dim = len(vectors[0])
    print(f"Total sections embedded: {EXPECTED_COUNT}")
    print(f"Chroma collection: {COLLECTION_NAME}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Persistence directory: {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    main()
