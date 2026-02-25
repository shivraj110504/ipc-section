import json
import os
import requests
import re
from pathlib import Path
from jsonschema import Draft202012Validator

# =========================
# CONFIG
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
MODEL = "meta-llama/llama-3-8b-instruct"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "http://localhost",
    "Content-Type": "application/json"
}

# =========================
# LOAD INPUT + FILE PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "ipc_cleaned_v4.json"
SCHEMA_PATH = BASE_DIR / "ipc_enriched_v1.schema.json"
DRAFT_PATH = BASE_DIR / "data" / "ipc_enriched_v1_draft.json"
FAILED_LOG_PATH = BASE_DIR / "data" / "failed_sections.log"

with INPUT_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

with SCHEMA_PATH.open("r", encoding="utf-8") as f:
    schema = json.load(f)

validator = Draft202012Validator(schema)
item_validator = Draft202012Validator(schema["items"])

# =========================
# EDITORIAL NOISE PATTERNS
# =========================

EDITORIAL_NOISE_PATTERNS = [
    r"\bthe act has been amended\b",
    r"\bsubs?\.\s*by\s*act\b",
    r"\brep\.?\s*,?\s*by\s*",
    r"\bins\.?\s*by\s*act\b",
    r"\bibid\.?\b",
    r"\bomitted by act\b",
    r"\breg\.\s*\d+\s*of\b",
    r"\bfor\s+\"the states\"\b"
]

# Template summary opening patterns that indicate low quality
TEMPLATE_SUMMARY_PATTERNS = [
    r"^it clarifies the legal effect of",
    r"^it defines the meaning of",
    r"^it explains how",
    r"^this provision addresses",
    r"^this section deals with the",
    r"^this provision deals with",
    r"^this law deals with",
    r"^this rule addresses"
]

# Fragment keyword patterns (mechanically constructed, not natural language)
FRAGMENT_KEYWORD_PATTERNS = [
    r"^india\s+except$",
    r"^indian\s+penal$",
    r"^whole\s+india$",
    r"^person\s+said$",
    r"^word\s+\w+$",
    r"^words\s+\w+$",
    r"^sense\s+expression$",
    r"^male\s+female$",
    r"^penal\s+code$",
    r"^singular\s+plural$",
    r"^act\s+shall$",
    r"^every\s+person$",
    r"^person\s+person$",
    r"^shall\s+be$"
]

# =========================
# PROMPTS
# =========================

system_prompt = """You are a legal data enrichment engine for the Indian Penal Code.

Your task: given an IPC section, produce a structured JSON record with a high-quality
summary and keywords suitable for semantic search and citizen-facing legal retrieval.

SUMMARY RULES:
- Write 2 to 4 complete, fluent sentences.
- Explain what the provision actually does in plain English.
- Do NOT start with template patterns like "It clarifies the legal effect of...",
  "This provision addresses...", "It defines the meaning of...", "It explains how...".
- Start each summary differently. Vary sentence structure.
- Do NOT hallucinate legal meaning beyond what the text says.
- Do NOT mention "IPC" or "Section" by number.

KEYWORD RULES:
- Produce exactly 5 to 8 keywords.
- Each keyword must be a multi-word phrase (at least 3 words, ideally 3-5 words).
- Keywords must be lowercase.
- Keywords must be natural-language phrases a citizen might type when searching for
  legal help. Examples of GOOD keywords:
    "punishment for crimes committed in india"
    "definition of public servant under criminal law"
    "what counts as movable property in law"
    "gender neutral language in criminal code"
    "applicability of criminal code across india"
- Do NOT produce mechanical token fragments like "india except", "sense expression",
  "male female", "indian penal", "whole india".
- Do NOT produce overly generic phrases like "criminal law", "legal provision".
- Each keyword should capture a distinct aspect of the provision.

OFFENCE_TYPE must be one of:
Property Crime | Violent Crime | Fraud / Cheating | Sexual Offence |
Public Servant Offence | Abetment | General Exception | Punishment | Other

FULL_TEXT RULES:
- Include ONLY the statutory wording of the provision.
- Remove all editorial amendments, footnotes, and amendment numbering fragments
  (e.g., "The Act has been amended...", "Subs. by Act...", "Rep. by...", "ibid.").
- Preserve illustrations and explanations that are part of the statutory text.

Output ONLY valid JSON. No markdown, no explanation outside the JSON."""

# =========================
# API CALL
# =========================

def call_llm(section, attempt=1):
    user_prompt = f"""Transform this IPC section into structured enrichment JSON.

Input:
  law_type: IPC
  section_number: {section["section_number"]}
  section_title: {section["section_title"]}
  full_text: {section["bare_text"]}

Return JSON:
{{
  "law_type": "IPC",
  "section_number": "{section["section_number"]}",
  "section_title": "{section["section_title"]}",
  "full_text": "<statutory text only, no amendments>",
  "summary": "<2-4 fluent sentences, plain English, no template openings>",
  "keywords": ["<5-8 natural multi-word citizen-search phrases, at least 3 words each>"],
  "offence_type": "<one of the allowed categories>"
}}

CRITICAL KEYWORD GUIDANCE:
- Think about what a citizen would type into a search engine if they had a legal
  problem related to this section.
- Each keyword phrase must be at least 3 words long.
- Good: "applicability of criminal code across india"
- Bad: "india except" or "penal code" or "criminal law"

CRITICAL SUMMARY GUIDANCE:
- Even for short definitional sections, you MUST write at least 2 complete sentences.
- Explain the definition AND describe when/how it matters in practice.

Return ONLY the JSON object."""

    # Slightly higher temperature on retry to get varied output
    temp = 0.4 if attempt == 1 else 0.55

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temp,
        "top_p": 1,
        "max_tokens": 1200
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()

# =========================
# TEXT CLEANING UTILITIES
# =========================

def normalize_space(text):
    text = str(text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text


def clean_full_text_statutory(text):
    """Strip editorial amendment tails and noise from statutory text."""
    cleaned = normalize_space(text)

    tail_markers = [
        r"\bThe\s+Act\s+has\s+been\s+amended\b",
        r"\bRep\.?\s*,?\s*by\b",
        r"\bSubs?\.?\s*by\s*Act\b",
        r"\bIns\.?\s*by\s*Act\b",
        r"\bibid\.?\b",
        r"\bomitted\s+by\s+Act\b"
    ]
    for marker in tail_markers:
        m = re.search(marker, cleaned, flags=re.IGNORECASE)
        if m:
            cleaned = cleaned[:m.start()].rstrip(" ,;:-")
            break

    cleaned = re.sub(r"\b\d+\.?\s*(for|ins\.?|rep\.?|subs\.?|ibid\.?)[^.;:]*[.;:]?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\s*\*\s*\*\s*\*+", " ", cleaned)
    cleaned = normalize_space(cleaned)

    return cleaned if len(cleaned) >= 10 else normalize_space(text)


def sentence_count(text):
    parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(parts)


# =========================
# QUALITY CHECKS (REJECTION-BASED)
# =========================

def is_fragment_keyword(kw):
    """Return True if the keyword looks like a mechanical fragment."""
    kw = kw.strip().lower()
    words = kw.split()
    if len(words) < 2:
        return True

    for pattern in FRAGMENT_KEYWORD_PATTERNS:
        if re.match(pattern, kw):
            return True

    # Two short tokens with no semantic connection
    if len(words) == 2 and all(len(w) <= 4 for w in words):
        return True

    return False


def is_generic_keyword(kw):
    """Return True if the keyword is too generic to be useful for retrieval."""
    kw = kw.strip().lower()
    generic_phrases = {
        "criminal law", "indian law", "legal provision", "penal code",
        "law and order", "legal system", "criminal justice",
        "legal framework", "rule of law", "criminal code",
        "law enforcement", "legal rights"
    }
    return kw in generic_phrases


def is_template_summary(summary):
    """Return True if the summary follows a template pattern."""
    summary_lower = summary.strip().lower()
    for pattern in TEMPLATE_SUMMARY_PATTERNS:
        if re.match(pattern, summary_lower):
            return True
    return False


def keywords_quality_ok(keywords):
    """Check that all keywords meet quality standards. Returns (ok, reason)."""
    if not isinstance(keywords, list):
        return False, "keywords is not a list"
    if len(keywords) < 5 or len(keywords) > 8:
        return False, f"keyword count {len(keywords)} outside 5-8 range"
    if len(set(keywords)) != len(keywords):
        return False, "duplicate keywords"

    for kw in keywords:
        if not isinstance(kw, str) or kw.strip() == "":
            return False, "empty or non-string keyword"
        if len(kw.split()) < 2:
            return False, f"single-word keyword: '{kw}'"
        if len(kw) < 3:
            return False, f"keyword too short: '{kw}'"
        if is_fragment_keyword(kw):
            return False, f"fragment keyword: '{kw}'"
        if is_generic_keyword(kw):
            return False, f"generic keyword: '{kw}'"

    return True, "OK"


def summary_quality_ok(summary):
    """Check that the summary meets quality standards. Returns (ok, reason)."""
    if not isinstance(summary, str) or len(summary) < 50:
        return False, "summary too short"
    sc = sentence_count(summary)
    if sc < 2:
        return False, "summary has fewer than 2 sentences"
    if sc > 4:
        return False, "summary has more than 4 sentences"
    if is_template_summary(summary):
        return False, "summary uses template opening pattern"
    return True, "OK"


# =========================
# NORMALIZATION LAYER (MINIMAL)
# =========================

def normalize_record(record, source_section):
    """
    Minimal normalization. Does NOT generate or repair keywords/summaries.
    Only cleans formatting and enforces source-of-truth fields.
    """
    normalized = record.copy()

    # Source-of-truth fields
    normalized["law_type"] = "IPC"
    normalized["section_number"] = str(source_section.get("section_number", "")).strip()
    normalized["section_title"] = str(source_section.get("section_title", "")).strip()

    # Clean full_text from source (strip editorial noise)
    normalized["full_text"] = clean_full_text_statutory(source_section.get("bare_text", ""))

    # --- Keywords: minimal cleanup only (lowercase, trim, dedup) ---
    raw_keywords = normalized.get("keywords", [])
    cleaned = []
    seen = set()
    for kw in raw_keywords:
        if not isinstance(kw, str):
            continue
        kw = normalize_space(kw).lower().strip()
        kw = re.sub(r"[^a-z0-9\s\-]", " ", kw)
        kw = normalize_space(kw)
        if kw and kw not in seen:
            seen.add(kw)
            cleaned.append(kw)

    normalized["keywords"] = cleaned[:8]

    # --- Summary: preserve LLM output, just normalize whitespace ---
    normalized["summary"] = normalize_space(normalized.get("summary", ""))

    # --- offence_type: map to allowed values ---
    allowed_offence_types = {
        "Property Crime", "Violent Crime", "Fraud / Cheating",
        "Sexual Offence", "Public Servant Offence", "Abetment",
        "General Exception", "Punishment", "Other"
    }
    offence_raw = str(normalized.get("offence_type", "")).strip()
    matched = None
    for option in allowed_offence_types:
        if offence_raw.lower() == option.lower():
            matched = option
            break
    normalized["offence_type"] = matched if matched else "Other"

    return normalized


# =========================
# VALIDATION
# =========================

def has_no_empty_fields(record):
    for value in record.values():
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
    return True


def validate_enriched(record, source_section):
    """Full validation: schema + quality checks. Returns (ok, reason)."""

    # Schema validation
    errors = sorted(item_validator.iter_errors(record), key=lambda e: e.path)
    if errors:
        return False, f"schema: {errors[0].message}"

    # Empty fields
    if not has_no_empty_fields(record):
        return False, "record contains empty fields"

    # Editorial noise in full_text
    if any(re.search(p, record.get("full_text", ""), flags=re.IGNORECASE)
           for p in EDITORIAL_NOISE_PATTERNS):
        return False, "full_text contains editorial amendment noise"

    # Keyword quality
    kw_ok, kw_reason = keywords_quality_ok(record.get("keywords", []))
    if not kw_ok:
        return False, f"keyword quality: {kw_reason}"

    # Summary quality
    sum_ok, sum_reason = summary_quality_ok(record.get("summary", ""))
    if not sum_ok:
        return False, f"summary quality: {sum_reason}"

    return True, "OK"


# =========================
# PARSE HELPERS
# =========================

def repair_json_string(text):
    """Attempt to fix common LLM JSON issues: unescaped quotes inside string values."""
    # Find the JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    candidate = text[start:end + 1]

    # Replace smart quotes with straight quotes
    candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')
    candidate = candidate.replace("\u2018", "'").replace("\u2019", "'")

    # Try to fix unescaped quotes within string values by processing line by line
    lines = candidate.split("\n")
    fixed_lines = []
    for line in lines:
        # Match lines like:  "key": "value with "problematic" quotes",
        m = re.match(r'^(\s*"[^"]+"\s*:\s*")(.*)(",?\s*)$', line)
        if m:
            prefix, value, suffix = m.group(1), m.group(2), m.group(3)
            # Escape any unescaped double quotes inside the value
            value = value.replace('\\"', '\x00')  # protect already-escaped
            value = value.replace('"', '\\"')
            value = value.replace('\x00', '\\"')  # restore
            fixed_lines.append(prefix + value + suffix)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try repairing common LLM JSON issues
    repaired = repair_json_string(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    raise json.JSONDecodeError("No JSON object found", text, 0)


# =========================
# BATCH PIPELINE (SECTIONS 1-25)
# =========================

START_SECTION = 1
END_SECTION = 100

if FAILED_LOG_PATH.exists():
    FAILED_LOG_PATH.unlink()

draft_records = []

source_sections = [
    row for row in data
    if START_SECTION <= int(re.match(r"\d+", str(row.get("section_number", "0"))).group(0)) <= END_SECTION
]

processed = 0
added = 0

for section in source_sections:
    processed += 1
    section_number = str(section.get("section_number", "")).strip()

    success = False
    last_error = "unknown error"

    for attempt in range(1, 4):
        try:
            result = call_llm(section, attempt=attempt)
            if "choices" not in result or not result["choices"]:
                raise ValueError(f"LLM response missing choices: {result}")

            raw_output = result["choices"][0]["message"]["content"]
            parsed = extract_json_object(raw_output)
            normalized = normalize_record(parsed, section)

            is_valid, reason = validate_enriched(normalized, section)
            if not is_valid:
                last_error = reason
                print(f"[{processed}/{len(source_sections)}] {section_number}: attempt {attempt} rejected ({reason})")
                continue

            draft_records.append(normalized)
            with DRAFT_PATH.open("w", encoding="utf-8") as f:
                json.dump(draft_records, f, ensure_ascii=False, indent=2)

            added += 1
            success = True
            print(f"[{processed}/{len(source_sections)}] {section_number}: accepted (total={added})")
            break

        except Exception as exc:
            last_error = str(exc)
            print(f"[{processed}/{len(source_sections)}] {section_number}: attempt {attempt} error ({exc})")

    if not success:
        with FAILED_LOG_PATH.open("a", encoding="utf-8") as logf:
            logf.write(f"{section_number} | {last_error}\n")
        print(f"[{processed}/{len(source_sections)}] {section_number}: FAILED after retry -> logged")

print("\n" + "=" * 60)
print("BATCH COMPLETE")
print(f"Sections processed: {processed}")
print(f"Records accepted: {added}")
print(f"Records rejected: {processed - added}")
print(f"Draft: {DRAFT_PATH}")
print("=" * 60)

# =========================
# POST-RUN VERIFICATION
# =========================

print("\n--- POST-RUN VERIFICATION ---\n")

# 1. Schema valid
schema_valid = True
try:
    validator.validate(draft_records)
except Exception:
    schema_valid = False

# 2. No fragmented keywords
no_fragmented_keywords = all(
    not any(is_fragment_keyword(kw) for kw in r.get("keywords", []))
    for r in draft_records
)

# 3. Keywords semantically natural (multi-word, no generics, no fragments)
keywords_semantically_natural = all(
    keywords_quality_ok(r.get("keywords", []))[0]
    for r in draft_records
)

# 4. Summaries non-template style
summaries_non_template_style = all(
    not is_template_summary(r.get("summary", ""))
    for r in draft_records
)
# Also check that summaries are not all identical
summary_texts = [r.get("summary", "") for r in draft_records]
if len(summary_texts) > 1 and len(set(summary_texts)) < len(summary_texts) * 0.8:
    summaries_non_template_style = False

# 5. No editorial noise in full_text
no_editorial_noise_in_full_text = all(
    not any(re.search(p, r.get("full_text", ""), flags=re.IGNORECASE)
            for p in EDITORIAL_NOISE_PATTERNS)
    for r in draft_records
)

print(f"schema_valid={schema_valid}")
print(f"no_fragmented_keywords={no_fragmented_keywords}")
print(f"keywords_semantically_natural={keywords_semantically_natural}")
print(f"summaries_non_template_style={summaries_non_template_style}")
print(f"no_editorial_noise_in_full_text={no_editorial_noise_in_full_text}")

# Print sample for inspection
if draft_records:
    print("\n--- SAMPLE RECORD (first accepted) ---")
    print(json.dumps(draft_records[0], indent=2, ensure_ascii=False))
