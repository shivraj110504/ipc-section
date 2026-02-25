# IPC Section Prediction — RAG-based Legal Awareness Engine

> **AI-assisted Indian Penal Code (IPC) section prediction from plain-English incident descriptions.**

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Data Pipeline](#data-pipeline)
7. [Configuration & Thresholds](#configuration--thresholds)
8. [Setup & Installation](#setup--installation)
9. [Environment Variables](#environment-variables)
10. [Running the API](#running-the-api)
11. [API Reference](#api-reference)
12. [Testing](#testing)
13. [Validation Guard](#validation-guard)
14. [Security](#security)

---

## Overview

This system predicts the most applicable **Indian Penal Code (IPC) section** for a given incident description. It is designed for citizen-facing legal awareness — users describe an incident in plain English and receive a structured prediction with the IPC section, a confidence score, and an explanation.

**Key Design Principles:**

- **Deterministic** — Temperature `0.0` for maximum repeatability.
- **Grounded** — The LLM can only choose from sections retrieved by the vector database; it cannot hallucinate sections.
- **Strict Validation** — Every LLM response is parsed, validated, and sanitized before reaching the user.
- **Graceful Fallback** — Any failure at any stage produces a safe, structured fallback response.

---

## How It Works

The system follows a **Retrieval-Augmented Generation (RAG)** pipeline:

```
User Input (incident text)
        │
        ▼
┌──────────────────────┐
│   FastAPI Endpoint    │   POST /ipc/predict
│   (main.py)          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Similarity Gate    │   Embed query → ChromaDB cosine search
│   (retrieve_sections)│   Top-7 candidates ranked by similarity
└──────────┬───────────┘
           │
           │  similarity < -0.60 ?  ──▶  FALLBACK (no prediction)
           │
           ▼
┌──────────────────────┐
│   Prompt Builder     │   Constructs constrained LLM prompt
│   (llm_instruction   │   with candidate sections + rules
│    _template.py)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Gemini LLM Call    │   POST to Google Generative AI API
│   (ipc_reasoning     │   Model: gemini-2.5-flash
│    _engine.py)       │   Temperature: 0.0
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Validation Guard   │   Parse JSON, verify section is from
│   (llm_validation    │   allowed list, check confidence,
│    _guard.py)        │   sanitize output
└──────────┬───────────┘
           │
           │  validation fails ?  ──▶  FALLBACK (no prediction)
           │
           ▼
┌──────────────────────┐
│   API Response       │   Structured JSON with IPC section,
│   Mapping            │   title, confidence (0–100), explanation,
│                      │   suggestion, and disclaimer
└──────────────────────┘
```

---

## Architecture

| Layer            | Component                     | Responsibility                                       |
| ---------------- | ----------------------------- | ---------------------------------------------------- |
| **API**          | `main.py`                     | FastAPI endpoint, input validation, response mapping |
| **Reasoning**    | `ipc_reasoning_engine.py`     | Similarity gate, Gemini LLM call, title resolution   |
| **Retrieval**    | `retrieve_sections.py`        | Embedding generation, ChromaDB vector search         |
| **Prompt**       | `llm_instruction_template.py` | Constrained prompt construction with decision rules  |
| **Validation**   | `llm_validation_guard.py`     | JSON parsing, schema enforcement, confidence gating  |
| **Schema**       | `schemas.py`                  | Pydantic request model                               |
| **Vector Store** | `chroma_ipc_v1/`              | Pre-built ChromaDB collection (522 IPC sections)     |
| **Data**         | `data/ipc_enriched_v1.json`   | Enriched IPC dataset (source of truth)               |

---

## Tech Stack

| Category            | Technology                                   |
| ------------------- | -------------------------------------------- |
| **Language**        | Python 3.11                                  |
| **API Framework**   | FastAPI + Uvicorn                            |
| **LLM Provider**    | Google Gemini API (`gemini-2.5-flash`)       |
| **Embeddings**      | OpenRouter (`openai/text-embedding-3-small`) |
| **Vector Database** | ChromaDB (persistent local storage)          |
| **Validation**      | Pydantic, custom JSON guard                  |
| **Data Format**     | JSON, JSON Schema (Draft 2020-12)            |

---

## Project Structure

```
[Rebuild] IPC Prediction/
│
├── .env                            # API keys (git-ignored)
├── .gitignore                      # Git exclusions
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── ipc_raw_sections.json       # Raw IPC sections (scraped)
│   ├── ipc_cleaned_v4.json         # Cleaned section text
│   ├── ipc_enriched_v1.json        # Enriched dataset (summary, keywords, offence_type)
│   ├── ipc_enriched_v1_draft.json  # Draft enrichment output
│   └── ipc_enriched_v1.schema.json # JSON Schema for validation
│
├── script/
│   ├── main.py                     # FastAPI app & /ipc/predict endpoint
│   ├── schemas.py                  # Pydantic request model (CaseInput)
│   ├── ipc_reasoning_engine.py     # Core prediction pipeline
│   ├── retrieve_sections.py        # ChromaDB retrieval + embedding
│   ├── llm_instruction_template.py # Prompt builder
│   ├── llm_validation_guard.py     # LLM output validation & sanitization
│   │
│   ├── build_embedding_texts.py    # Constructs embedding text from enriched data
│   ├── generate_and_store_embeddings.py  # One-time: generates & stores embeddings
│   ├── map_titles_from_cleaned.py  # Maps titles from cleaned dataset
│   ├── purify_full_text.py         # Removes editorial noise from full text
│   ├── test_enrichment_single.py   # Single-section enrichment test
│   │
│   ├── validate_retrieval.py       # 20-case retrieval validation suite
│   ├── test_stability.py           # 8-category stability & stress tests
│   │
│   └── chroma_ipc_v1/             # ChromaDB persistent storage (git-ignored)
│       ├── chroma.sqlite3
│       └── <segment_data>/
│
└── IPC_Pred_Rebuild/               # Python virtual environment (git-ignored)
```

---

## Data Pipeline

The enriched dataset was built through a multi-stage pipeline:

| Stage | Script                             | Description                                            |
| ----- | ---------------------------------- | ------------------------------------------------------ |
| 1     | _(external)_                       | Raw IPC text scraped → `ipc_raw_sections.json`         |
| 2     | `purify_full_text.py`              | Remove editorial amendments and noise                  |
| 3     | `map_titles_from_cleaned.py`       | Map section titles from cleaned dataset                |
| 4     | `test_enrichment_single.py`        | LLM-based enrichment (summary, keywords, offence_type) |
| 5     | `build_embedding_texts.py`         | Construct embedding text per section                   |
| 6     | `generate_and_store_embeddings.py` | Generate embeddings & store in ChromaDB                |

**Final output:** 522 IPC sections stored in ChromaDB with metadata (section_number, title, summary, keywords, full_text, offence_type).

> **Note:** The ChromaDB store (`chroma_ipc_v1/`) is pre-built. You do **not** need to re-run the data pipeline unless the enriched dataset changes.

---

## Configuration & Thresholds

| Parameter              | Value   | File                      | Purpose                                                                                                   |
| ---------------------- | ------- | ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `SIMILARITY_THRESHOLD` | `-0.60` | `ipc_reasoning_engine.py` | Minimum cosine similarity to proceed to LLM. Queries below this score are irrelevant and return fallback. |
| `MIN_CONFIDENCE`       | `0.30`  | `llm_validation_guard.py` | Minimum LLM confidence to accept a prediction. Below this, fallback is returned.                          |
| `TOP_K`                | `7`     | `retrieve_sections.py`    | Number of candidate sections retrieved from ChromaDB.                                                     |
| `temperature`          | `0.0`   | `ipc_reasoning_engine.py` | Gemini generation temperature. Set to 0 for determinism.                                                  |
| `timeout`              | `60s`   | `ipc_reasoning_engine.py` | HTTP request timeout for the Gemini API call.                                                             |

---

## Setup & Installation

### Prerequisites

- **Python 3.11+**
- **pip**
- API keys for **Google AI Studio (Gemini)** and **OpenRouter**

### Steps

```bash
# 1. Clone / navigate to the project directory
cd "[Rebuild] IPC Prediction"

# 2. Create virtual environment
python -m venv IPC_Pred_Rebuild

# 3. Activate virtual environment
# Windows (Git Bash / MSYS2)
source IPC_Pred_Rebuild/Scripts/activate
# Windows (CMD)
IPC_Pred_Rebuild\Scripts\activate.bat
# Windows (PowerShell)
IPC_Pred_Rebuild\Scripts\Activate.ps1
# Linux / macOS
source IPC_Pred_Rebuild/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root (already git-ignored):

```dotenv
# Gemini API key (Google AI Studio) — used for LLM reasoning
GEMINI_API_KEY=your_gemini_api_key_here

# OpenRouter API key — used for embedding generation
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Before running the server**, export them into your shell:

```bash
# Option A: Source the .env file
set -a && source .env && set +a

# Option B: Export manually
export GEMINI_API_KEY="your_key"
export OPENROUTER_API_KEY="your_key"
```

If either key is missing, the system raises a `RuntimeError` at startup — it will **not** start silently with missing credentials.

---

## Running the API

```bash
# Ensure virtual environment is activated and env vars are set
set -a && source .env && set +a

# Start the server
uvicorn script.main:app --host 127.0.0.1 --port 8000
```

The API is now live at `http://127.0.0.1:8000`.

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Reference

### `POST /ipc/predict`

Predict the most applicable IPC section for an incident description.

#### Request

```json
{
  "text": "He cheated me by taking money and not delivering the goods."
}
```

| Field  | Type   | Constraints   |
| ------ | ------ | ------------- |
| `text` | string | Min length: 7 |

#### Successful Prediction Response

```json
{
  "prediction": {
    "ipc_section": "IPC 420",
    "title": "Cheating and dishonestly inducing delivery of property.",
    "confidence": 95
  },
  "explanation": "The incident describes an act of cheating where money was taken and goods were not delivered...",
  "suggestion": "Consider consulting a legal professional.",
  "disclaimer": "This is an AI-assisted legal awareness tool."
}
```

#### Fallback Response (irrelevant or ambiguous input)

```json
{
  "prediction": {
    "ipc_section": null,
    "title": "",
    "confidence": 0
  },
  "explanation": "The described incident does not clearly fall under a specific IPC section.",
  "suggestion": "Document all relevant evidence.",
  "disclaimer": "This is an AI-assisted legal awareness tool."
}
```

#### Insufficient Input Response

```json
{
  "prediction": null,
  "message": "Please describe the incident with sufficient details.",
  "disclaimer": "This tool requires incident details to provide a legal prediction."
}
```

#### Response Fields

| Field                    | Type             | Description                                             |
| ------------------------ | ---------------- | ------------------------------------------------------- |
| `prediction.ipc_section` | `string \| null` | Predicted section (e.g., `"IPC 420"`) or `null`         |
| `prediction.title`       | `string`         | Section title from the enriched dataset                 |
| `prediction.confidence`  | `integer`        | Confidence score, 0–100                                 |
| `explanation`            | `string`         | Plain-English reasoning for the prediction              |
| `suggestion`             | `string`         | Randomly selected general legal suggestion              |
| `disclaimer`             | `string`         | Fixed: `"This is an AI-assisted legal awareness tool."` |

---

## Testing

### Stability & Stress Tests (8 categories)

```bash
set -a && source .env && set +a
python -m script.test_stability
```

Covers:

1. **Strong IPC Case** — Expects valid section, confidence, title
2. **Low Similarity Case** — Expects fallback (irrelevant input)
3. **Ambiguous Case** — Accepts prediction or graceful fallback
4. **Repeatability** — 10 identical runs, checks determinism
5. **Invalid LLM JSON** — 12 malformed responses, all must fallback
6. **API Failure Simulation** — Bad URLs, expects graceful fallback
7. **Empty / Short Input** — Edge cases, expects fallback
8. **Confidence Boundary** — Tests MIN_CONFIDENCE threshold edges

### Retrieval Validation (20 cases)

```bash
cd script && python validate_retrieval.py
```

Validates that the correct IPC section appears in the Top-7 for 20 curated test descriptions, plus 4 edge cases.

---

## Validation Guard

The `llm_validation_guard.py` module enforces strict rules on every LLM response before it reaches the user:

| Check                                               | Action on Failure                        |
| --------------------------------------------------- | ---------------------------------------- |
| Response is not valid JSON                          | → Fallback                               |
| Missing required keys                               | → Fallback                               |
| `predicted_sections` not a list of exactly 1 string | → Fallback                               |
| Section not in allowed list                         | → Fallback                               |
| Confidence not numeric or NaN/Inf                   | → Fallback                               |
| Confidence below `0.30`                             | → Fallback                               |
| Explanation empty                                   | → Fallback                               |
| Markdown code fences                                | → Stripped, then parsed                  |
| Whitespace in section number                        | → Trimmed, then validated                |
| Extra JSON keys from LLM                            | → Ignored (only required keys extracted) |

**Fallback response** (consistent across all failure modes):

```json
{
  "predicted_sections": [],
  "confidence": 0.0,
  "explanation": "The described incident does not clearly fall under a specific IPC section."
}
```

---

## Security

- **No hardcoded API keys** — All keys are loaded from environment variables via `os.getenv()`.
- **Startup validation** — Missing keys raise `RuntimeError` immediately; the server will not start with missing credentials.
- **`.env` is git-ignored** — Listed in `.gitignore` to prevent accidental commits.
- **No key logging** — Keys are never printed, logged, or included in error responses.
- **CORS enabled** — Configured for development (`allow_origins=["*"]`). Restrict origins before production deployment.

---

<p align="center"><em>Built for VOIS Hackathon 2026</em></p>
