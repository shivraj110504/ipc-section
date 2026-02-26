import os

import requests

try:
    from .llm_instruction_template import build_ipc_reasoning_prompt
    from .retrieve_sections import _retrieve_with_scores
    from .llm_validation_guard import validate_llm_response
except (ImportError, ValueError):
    from llm_instruction_template import build_ipc_reasoning_prompt
    from retrieve_sections import _retrieve_with_scores
    from llm_validation_guard import validate_llm_response


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def _check_gemini_key():
    print(f"--- API KEY CHECK: GEMINI_API_KEY present? {'Yes' if GEMINI_API_KEY else 'No'} ---")
    if GEMINI_API_KEY:
        print(f"--- API KEY CHECK: GEMINI_API_KEY starts with: {GEMINI_API_KEY[:4]}... ---")

GEMINI_MODEL = "models/gemini-1.5-flash"  # Switched to standard model
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

SIMILARITY_THRESHOLD = -5.0  # Temporarily broadened for debugging

def _fallback_response() -> dict:
    return {
        "predicted_sections": [],
        "confidence": 0.0,
        "explanation": "The described incident does not clearly fall under a specific IPC section.",
        "title": "",
    }


def run_similarity_gate(incident_text: str) -> dict:
    try:
        print(f"--- RUNNING GATE FOR: {incident_text[:50]}... ---")
        ranked_candidates = _retrieve_with_scores(incident_text)
        if not ranked_candidates:
            print("--- GATE: NO CANDIDATES FOUND ---")
            return _fallback_response()

        top_similarity = float(ranked_candidates[0][1])
        print(f"--- GATE: TOP SIMILARITY = {top_similarity} (THRESHOLD = {SIMILARITY_THRESHOLD}) ---")
        
        if top_similarity < SIMILARITY_THRESHOLD:
            print("--- GATE: SIMILARITY BELOW THRESHOLD ---")
            return _fallback_response()

        candidate_sections = [metadata for metadata, _ in ranked_candidates]
        allowed_section_numbers = [
            str(section.get("section_number", "")).strip() for section in candidate_sections
        ]

        prompt = build_ipc_reasoning_prompt(incident_text, candidate_sections)
        print("--- GATE: PROMPT BUILT ---")

        return {
            "incident_text": incident_text,
            "candidate_sections": candidate_sections,
            "allowed_section_numbers": allowed_section_numbers,
            "llm_prompt": prompt,
        }
    except Exception as e:
        print(f"--- GATE ERROR: {str(e)} ---")
        return _fallback_response()


def predict_ipc_section(incident_text: str) -> dict:
    try:
        _check_gemini_key()
        gate_result = run_similarity_gate(incident_text)

        if "llm_prompt" not in gate_result:
            print("--- PREDICT: GATE FAILED OR RETURNED FALLBACK ---")
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

        # List of models to try
        models_to_try = [GEMINI_MODEL, "models/gemini-1.5-flash"]
        raw_response = None
        current_model_used = None

        for model_name in models_to_try:
            print(f"--- CALLING GEMINI: {model_name} ---")
            api_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"{model_name}:generateContent?key={GEMINI_API_KEY}"
            )
            
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                
                if response.status_code == 200:
                    body = response.json()
                    raw_response = body["candidates"][0]["content"]["parts"][0]["text"]
                    current_model_used = model_name
                    print(f"--- GEMINI SUCCESS WITH: {model_name} ---")
                    break
                else:
                    print(f"--- GEMINI API ERROR ({model_name}): {response.status_code} - {response.text} ---")
            except Exception as inner_e:
                print(f"--- GEMINI REQUEST FAILED ({model_name}): {str(inner_e)} ---")

        if not raw_response:
            print("--- ALL GEMINI MODELS FAILED ---")
            return _fallback_response()

        print(f"--- GEMINI RAW RESPONSE FROM {current_model_used}: {raw_response[:100]}... ---")

        validated = validate_llm_response(raw_response, allowed_section_numbers)

        title = ""
        if validated.get("predicted_sections"):
            predicted_section = str(validated["predicted_sections"][0]).strip()
            for candidate in gate_result.get("candidate_sections", []):
                if str(candidate.get("section_number", "")).strip() == predicted_section:
                    title = str(candidate.get("title", "")).strip()
                    break

        validated["title"] = title
        print(f"--- PREDICTION SUCCESSFUL: {validated.get('predicted_sections')} ---")
        return validated
    except Exception as e:
        print(f"--- PREDICT ERROR: {str(e)} ---")
        return _fallback_response()
