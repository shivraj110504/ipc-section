"""
Phase 4 – Stability & Stress Testing
Covers all 8 test categories for the RAG-based IPC prediction system.
"""

import sys
import json
import time

# ---------------------------------------------------------------------------
# Imports – use fallback pattern so the script works from workspace root
# ---------------------------------------------------------------------------
try:
    from script.ipc_reasoning_engine import predict_ipc_section, _fallback_response, SIMILARITY_THRESHOLD
    from script.llm_validation_guard import validate_llm_response, MIN_CONFIDENCE
    from script.retrieve_sections import _retrieve_with_scores
except ImportError:
    from ipc_reasoning_engine import predict_ipc_section, _fallback_response, SIMILARITY_THRESHOLD
    from llm_validation_guard import validate_llm_response, MIN_CONFIDENCE
    from retrieve_sections import _retrieve_with_scores

PASS = "PASS"
FAIL = "FAIL"
FALLBACK_EXPLANATION = "The described incident does not clearly fall under a specific IPC section."

results: list[tuple[str, str, str]] = []  # (category, test_name, status)


def record(category: str, name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((category, name, status))
    mark = "+" if passed else "X"
    msg = f"  [{mark}] {name}"
    if detail and not passed:
        msg += f"  -- {detail}"
    print(msg)


# ===================================================================
# CATEGORY 1 – STRONG IPC CASE
# ===================================================================
def test_category_1():
    print("\n=== CATEGORY 1: Strong IPC Case ===")
    text = "He cheated me by taking money and not delivering the goods."
    out = predict_ipc_section(text)

    record("CAT1", "Returns dict", isinstance(out, dict))
    record("CAT1", "predicted_sections present", "predicted_sections" in out)
    record("CAT1", "confidence present", "confidence" in out)
    record("CAT1", "explanation present", "explanation" in out)
    record("CAT1", "title present", "title" in out)

    sections = out.get("predicted_sections", [])
    conf = out.get("confidence", -1)
    expl = out.get("explanation", "")
    title = out.get("title", "")

    has_section = isinstance(sections, list) and len(sections) == 1
    record("CAT1", "Exactly one section", has_section)

    if has_section:
        record("CAT1", "Section is string", isinstance(sections[0], str) and len(sections[0]) > 0)
    else:
        record("CAT1", "Section is string", False, f"sections={sections}")

    record("CAT1", "Confidence is float", isinstance(conf, (int, float)))
    record("CAT1", "Confidence 0.3-1.0", 0.3 <= conf <= 1.0, f"conf={conf}")
    record("CAT1", "Explanation non-empty", isinstance(expl, str) and len(expl.strip()) > 0)
    record("CAT1", "Title non-empty", isinstance(title, str) and len(title.strip()) > 0, f"title={repr(title)}")
    record("CAT1", "No extra keys", set(out.keys()) == {"predicted_sections", "confidence", "explanation", "title"}, f"keys={set(out.keys())}")

    print(f"  >> Section: {sections}, Confidence: {conf}")
    print(f"  >> Title: {title}")
    print(f"  >> Explanation: {expl[:120]}...")
    return out


# ===================================================================
# CATEGORY 2 – LOW SIMILARITY CASE
# ===================================================================
def test_category_2():
    print("\n=== CATEGORY 2: Low Similarity Case ===")
    text = "The weather is nice today."
    out = predict_ipc_section(text)

    sections = out.get("predicted_sections", ["NOT_EMPTY"])
    conf = out.get("confidence", -1)
    expl = out.get("explanation", "")

    record("CAT2", "Returns dict", isinstance(out, dict))
    record("CAT2", "Empty sections", sections == [])
    record("CAT2", "Confidence is 0.0", conf == 0.0, f"conf={conf}")
    record("CAT2", "Fallback explanation", expl == FALLBACK_EXPLANATION, f"expl={expl[:80]}")
    record("CAT2", "Title empty", out.get("title", None) == "", f"title={repr(out.get('title'))}")

    # Verify similarity gate actually blocked
    ranked = _retrieve_with_scores(text)
    top_sim = float(ranked[0][1]) if ranked else 0.0
    record("CAT2", "Top sim < threshold", top_sim < SIMILARITY_THRESHOLD, f"top_sim={top_sim:.4f}")
    print(f"  >> Top similarity: {top_sim:.4f} (threshold: {SIMILARITY_THRESHOLD})")


# ===================================================================
# CATEGORY 3 – AMBIGUOUS CASE
# ===================================================================
def test_category_3():
    print("\n=== CATEGORY 3: Ambiguous Case ===")
    text = "They argued loudly in public."
    out = predict_ipc_section(text)

    sections = out.get("predicted_sections", [])
    conf = out.get("confidence", -1)
    expl = out.get("explanation", "")
    title = out.get("title", "")

    record("CAT3", "Returns dict", isinstance(out, dict))
    record("CAT3", "Sections list", isinstance(sections, list))
    record("CAT3", "Sections len 0 or 1", len(sections) <= 1)

    if sections:
        record("CAT3", "Confidence >= 0.3", conf >= 0.3, f"conf={conf}")
        record("CAT3", "Explanation non-empty", len(expl.strip()) > 0)
        record("CAT3", "No hallucinated section", isinstance(sections[0], str) and sections[0].strip() != "")
        print(f"  >> Predicted: {sections[0]}, Confidence: {conf}, Title: {title}")
    else:
        record("CAT3", "Fallback confidence", conf == 0.0, f"conf={conf}")
        record("CAT3", "Fallback explanation", expl == FALLBACK_EXPLANATION)
        print(f"  >> Fallback triggered")


# ===================================================================
# CATEGORY 4 – REPEATABILITY TEST (10 runs)
# ===================================================================
def test_category_4():
    print("\n=== CATEGORY 4: Repeatability Test (10 runs) ===")
    text = "He cheated me by taking money and not delivering the goods."
    outputs = []
    for i in range(10):
        out = predict_ipc_section(text)
        outputs.append(out)
        sys.stdout.write(f"  Run {i+1}/10 → Section: {out.get('predicted_sections')}, Conf: {out.get('confidence')}\n")
        sys.stdout.flush()

    # Compare all outputs (ignoring that suggestion is not in this dict)
    first = outputs[0]
    all_same_section = all(o.get("predicted_sections") == first.get("predicted_sections") for o in outputs)
    all_same_conf = all(o.get("confidence") == first.get("confidence") for o in outputs)
    all_same_title = all(o.get("title") == first.get("title") for o in outputs)
    all_same_expl = all(o.get("explanation") == first.get("explanation") for o in outputs)

    record("CAT4", "Section identical x10", all_same_section)
    record("CAT4", "Confidence identical x10", all_same_conf)
    record("CAT4", "Title identical x10", all_same_title)
    record("CAT4", "Explanation identical x10", all_same_expl)


# ===================================================================
# CATEGORY 5 – INVALID LLM JSON SIMULATION
# ===================================================================
def test_category_5():
    print("\n=== CATEGORY 5: Invalid LLM JSON Simulation ===")
    allowed = ["378", "420", "452"]
    fb_guard = {"predicted_sections": [], "confidence": 0.0, "explanation": FALLBACK_EXPLANATION}

    # 5a: Not JSON at all
    r = validate_llm_response("This is not JSON", allowed)
    record("CAT5", "Non-JSON → fallback", r == fb_guard)

    # 5b: JSON but wrong type
    r = validate_llm_response("[1, 2, 3]", allowed)
    record("CAT5", "Array JSON → fallback", r == fb_guard)

    # 5c: Missing keys
    r = validate_llm_response('{"predicted_sections": ["378"]}', allowed)
    record("CAT5", "Missing keys → fallback", r == fb_guard)

    # 5d: Extra keys — now accepted (extra keys are stripped)
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": 0.8, "explanation": "theft", "extra": true}', allowed)
    expected_extra = {"predicted_sections": ["378"], "confidence": 0.8, "explanation": "theft"}
    record("CAT5", "Extra keys → passes", r == expected_extra, f"got={r}")

    # 5e: Multi-section
    r = validate_llm_response('{"predicted_sections": ["378", "420"], "confidence": 0.8, "explanation": "theft"}', allowed)
    record("CAT5", "Multi-section → fallback", r == fb_guard)

    # 5f: Section not in allowed
    r = validate_llm_response('{"predicted_sections": ["999"], "confidence": 0.8, "explanation": "theft"}', allowed)
    record("CAT5", "Section not allowed → fallback", r == fb_guard)

    # 5g: Whitespace-wrapped section — now stripped and accepted
    r = validate_llm_response('{"predicted_sections": [" 378 "], "confidence": 0.8, "explanation": "theft"}', allowed)
    expected_ws = {"predicted_sections": ["378"], "confidence": 0.8, "explanation": "theft"}
    record("CAT5", "Whitespace section → passes", r == expected_ws, f"got={r}")

    # 5h: Confidence as string
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": "0.8", "explanation": "theft"}', allowed)
    record("CAT5", "String confidence → fallback", r == fb_guard)

    # 5i: Empty explanation
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": 0.8, "explanation": ""}', allowed)
    record("CAT5", "Empty explanation → fallback", r == fb_guard)

    # 5j: Valid input passes
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": 0.8, "explanation": "This involves theft."}', allowed)
    expected_valid = {"predicted_sections": ["378"], "confidence": 0.8, "explanation": "This involves theft."}
    record("CAT5", "Valid JSON → passes", r == expected_valid, f"got={r}")

    # 5k: Markdown-wrapped JSON (backticks) — now extracted and accepted
    r = validate_llm_response('```json\n{"predicted_sections": ["378"], "confidence": 0.8, "explanation": "theft"}\n```', allowed)
    expected_md = {"predicted_sections": ["378"], "confidence": 0.8, "explanation": "theft"}
    record("CAT5", "Markdown-wrapped → passes", r == expected_md, f"got={r}")

    # 5l: Empty string
    r = validate_llm_response("", allowed)
    record("CAT5", "Empty string → fallback", r == fb_guard)


# ===================================================================
# CATEGORY 6 – API FAILURE SIMULATION
# ===================================================================
def test_category_6():
    print("\n=== CATEGORY 6: API Failure Simulation ===")

    # We simulate by temporarily overriding the API URL to a bad endpoint
    try:
        from script import ipc_reasoning_engine as eng
    except ImportError:
        import ipc_reasoning_engine as eng

    original_url = eng.GEMINI_API_URL

    # 6a: Bad URL (connection failure)
    eng.GEMINI_API_URL = "https://localhost:1/nonexistent"
    r = eng.predict_ipc_section("Someone stole my wallet from my bag while I was on the bus.")
    record("CAT6", "Bad URL → fallback", r.get("predicted_sections") == [] and r.get("confidence") == 0.0)

    # 6b: Invalid URL scheme
    eng.GEMINI_API_URL = "not-a-url"
    r = eng.predict_ipc_section("Someone stole my wallet from my bag while I was on the bus.")
    record("CAT6", "Invalid URL → fallback", r.get("predicted_sections") == [] and r.get("confidence") == 0.0)

    # Restore
    eng.GEMINI_API_URL = original_url
    record("CAT6", "URL restored", eng.GEMINI_API_URL == original_url)


# ===================================================================
# CATEGORY 7 – EMPTY / SHORT INPUT
# ===================================================================
def test_category_7():
    print("\n=== CATEGORY 7: Empty / Short Input ===")

    for label, text in [("Empty string", ""), ("Whitespace only", "   "), ("Very short", "Hi"), ("Under 10 chars", "Help me")]:
        out = predict_ipc_section(text)
        is_fallback = out.get("predicted_sections") == [] and out.get("confidence") == 0.0
        record("CAT7", f"{label} → fallback", is_fallback, f"out={out}")


# ===================================================================
# CATEGORY 8 – CONFIDENCE BOUNDARY TEST
# ===================================================================
def test_category_8():
    print("\n=== CATEGORY 8: Confidence Boundary Test ===")
    allowed = ["378", "420"]
    fb = _fallback_response()
    # Remove title key for comparison with guard output (guard doesn't add title)
    fb_guard = {k: v for k, v in fb.items() if k != "title"}

    # 8a: Confidence 0.29 → below MIN_CONFIDENCE (0.30) → fallback
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": 0.29, "explanation": "theft"}', allowed)
    record("CAT8", "conf=0.29 → fallback", r == fb_guard, f"r={r}")

    # 8b: Confidence 0.30 → exactly at threshold → should pass
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": 0.30, "explanation": "theft"}', allowed)
    record("CAT8", "conf=0.30 → passes", r.get("predicted_sections") == ["378"] and r.get("confidence") == 0.3, f"r={r}")

    # 8c: Confidence 1.2 → clamped to 1.0 → passes
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": 1.2, "explanation": "theft"}', allowed)
    record("CAT8", "conf=1.2 → clamped 1.0", r.get("confidence") == 1.0 and r.get("predicted_sections") == ["378"], f"r={r}")

    # 8d: Confidence -0.5 → clamped to 0.0 → below MIN → fallback
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": -0.5, "explanation": "theft"}', allowed)
    record("CAT8", "conf=-0.5 → fallback", r == fb_guard, f"r={r}")

    # 8e: Confidence 0.0 → below MIN → fallback
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": 0.0, "explanation": "theft"}', allowed)
    record("CAT8", "conf=0.0 → fallback", r == fb_guard, f"r={r}")

    # 8f: Confidence NaN → fallback
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": NaN, "explanation": "theft"}', allowed)
    record("CAT8", "conf=NaN → fallback", r == fb_guard, f"r={r}")

    # 8g: Confidence Infinity → fallback
    r = validate_llm_response('{"predicted_sections": ["378"], "confidence": Infinity, "explanation": "theft"}', allowed)
    record("CAT8", "conf=Infinity → fallback", r == fb_guard, f"r={r}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("PHASE 4 – STABILITY & STRESS TESTING")
    print("=" * 60)

    # Categories 5, 7, 8 are offline (no API calls)
    test_category_5()
    test_category_7()
    test_category_8()

    # Category 6 needs similarity gate to pass, so requires embedding API
    test_category_6()

    # Categories 1, 2, 3 require live Gemini API calls
    cat1_out = test_category_1()
    test_category_2()
    test_category_3()

    # Category 4 requires 10 live calls
    test_category_4()

    # SUMMARY
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for _, _, s in results if s == PASS)
    failed = sum(1 for _, _, s in results if s == FAIL)
    print(f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}")

    if failed > 0:
        print("\nFAILED TESTS:")
        for cat, name, status in results:
            if status == FAIL:
                print(f"  [{cat}] {name}")

    print("\n" + ("ALL TESTS PASSED – SYSTEM DEMO SAFE" if failed == 0 else "SOME TESTS FAILED – REVIEW REQUIRED"))
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
