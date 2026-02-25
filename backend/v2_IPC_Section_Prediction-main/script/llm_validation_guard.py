import json
import math
import re


MIN_CONFIDENCE = 0.30

_REQUIRED_KEYS = {"predicted_sections", "confidence", "explanation"}


def _fallback_response() -> dict:
    return {
        "predicted_sections": [],
        "confidence": 0.0,
        "explanation": "The described incident does not clearly fall under a specific IPC section.",
    }


def _normalize_allowed_sections(allowed_section_numbers: list[str]) -> set[str]:
    normalized: set[str] = set()
    for value in allowed_section_numbers:
        text = str(value).strip()
        if text:
            normalized.add(text)
    return normalized


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) that LLMs often wrap around JSON."""
    stripped = text.strip()
    pattern = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)
    match = pattern.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def validate_llm_response(raw_response: str, allowed_section_numbers: list[str]) -> dict:
    try:
        cleaned = _strip_markdown_fences(raw_response)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return _fallback_response()

        if not isinstance(parsed, dict):
            return _fallback_response()

        # Accept response if it contains at least the required keys (ignore extras)
        if not _REQUIRED_KEYS.issubset(set(parsed.keys())):
            return _fallback_response()

        predicted_sections = parsed.get("predicted_sections")
        confidence = parsed.get("confidence")
        explanation = parsed.get("explanation")

        if not isinstance(predicted_sections, list):
            return _fallback_response()

        if len(predicted_sections) != 1:
            return _fallback_response()

        section_value = predicted_sections[0]
        if not isinstance(section_value, str):
            return _fallback_response()

        # Strip whitespace from section number instead of rejecting
        sanitized_section = section_value.strip()
        if not sanitized_section:
            return _fallback_response()

        allowed_set = _normalize_allowed_sections(allowed_section_numbers)
        if sanitized_section not in allowed_set:
            return _fallback_response()

        if isinstance(confidence, str):
            return _fallback_response()

        if not isinstance(confidence, (int, float)):
            return _fallback_response()

        confidence_value = float(confidence)
        if not math.isfinite(confidence_value):
            return _fallback_response()

        clamped_confidence = max(0.0, min(1.0, confidence_value))
        if not (0.0 <= clamped_confidence <= 1.0):
            return _fallback_response()

        if clamped_confidence < MIN_CONFIDENCE:
            return _fallback_response()

        if not isinstance(explanation, str):
            return _fallback_response()

        sanitized_explanation = explanation.strip()
        if not sanitized_explanation:
            return _fallback_response()

        return {
            "predicted_sections": [sanitized_section],
            "confidence": clamped_confidence,
            "explanation": sanitized_explanation,
        }
    except Exception:
        return _fallback_response()