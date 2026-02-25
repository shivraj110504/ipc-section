import os

import requests

try:
    from script.llm_instruction_template import build_ipc_reasoning_prompt
    from script.retrieve_sections import _retrieve_with_scores
    from script.llm_validation_guard import validate_llm_response
except ImportError:
    from llm_instruction_template import build_ipc_reasoning_prompt
    from retrieve_sections import _retrieve_with_scores
    from llm_validation_guard import validate_llm_response


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

GEMINI_MODEL = "models/gemini-2.5-flash"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

SIMILARITY_THRESHOLD = -0.60

def _fallback_response() -> dict:
    return {
        "predicted_sections": [],
        "confidence": 0.0,
        "explanation": "The described incident does not clearly fall under a specific IPC section.",
        "title": "",
    }


def run_similarity_gate(incident_text: str) -> dict:
    try:
        ranked_candidates = _retrieve_with_scores(incident_text)
        if not ranked_candidates:
            return _fallback_response()

        top_similarity = float(ranked_candidates[0][1])
        if top_similarity < SIMILARITY_THRESHOLD:
            return _fallback_response()

        candidate_sections = [metadata for metadata, _ in ranked_candidates]
        allowed_section_numbers = [
            str(section.get("section_number", "")).strip() for section in candidate_sections
        ]

        prompt = build_ipc_reasoning_prompt(incident_text, candidate_sections)

        return {
            "incident_text": incident_text,
            "candidate_sections": candidate_sections,
            "allowed_section_numbers": allowed_section_numbers,
            "llm_prompt": prompt,
        }
    except Exception:
        return _fallback_response()


def predict_ipc_section(incident_text: str) -> dict:
    try:
        gate_result = run_similarity_gate(incident_text)

        if "llm_prompt" not in gate_result:
            gate_result.setdefault("title", "")
            return gate_result

        llm_prompt = gate_result["llm_prompt"]
        allowed_section_numbers = gate_result["allowed_section_numbers"]

        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "parts": [{"text": llm_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
            },
        }

        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        body = response.json()
        raw_response = body["candidates"][0]["content"]["parts"][0]["text"]

        validated = validate_llm_response(raw_response, allowed_section_numbers)

        title = ""
        if validated.get("predicted_sections"):
            predicted_section = str(validated["predicted_sections"][0]).strip()
            for candidate in gate_result.get("candidate_sections", []):
                if str(candidate.get("section_number", "")).strip() == predicted_section:
                    title = str(candidate.get("title", "")).strip()
                    break

        validated["title"] = title
        return validated
    except Exception:
        return _fallback_response()