import json


def build_ipc_reasoning_prompt(incident_text: str, candidate_sections: list[dict]) -> str:
    allowed_section_numbers = [
        str(section.get("section_number", "")).strip() for section in candidate_sections
    ]

    candidate_blocks: list[str] = []
    for section in candidate_sections:
        section_number = str(section.get("section_number", "")).strip()
        title = str(section.get("title", "")).strip()
        summary = str(section.get("summary", "")).strip()
        keywords = section.get("keywords", [])

        if isinstance(keywords, list):
            keywords_text = ", ".join(str(item).strip() for item in keywords)
        else:
            keywords_text = str(keywords).strip()

        candidate_blocks.append(
            "\n".join(
                [
                    "Section Number: " + section_number,
                    "Title: " + title,
                    "Summary: " + summary,
                    "Keywords: " + keywords_text,
                ]
            )
        )

    allowed_list_text = json.dumps(allowed_section_numbers, ensure_ascii=False)
    candidates_text = "\n\n".join(candidate_blocks)

    prompt = (
        "You are a legal reasoning assistant for IPC section prediction.\n"
        "You are strictly restricted to the provided candidate sections.\n"
        "You are not allowed to invent, infer, or reference any section outside the allowed list.\n\n"
        "Allowed Section Numbers:\n"
        f"{allowed_list_text}\n\n"
        "Candidate Sections:\n"
        f"{candidates_text}\n\n"
        "Incident Description:\n"
        f"{incident_text.strip()}\n\n"
        "Decision Rules:\n"
        "1. Select exactly one section number or return an empty list.\n"
        "2. You may choose only from Allowed Section Numbers.\n"
        "3. If no section is even remotely applicable, return an empty list.\n"
        "4. Select the most applicable section even if the match is partial.\n"
        "5. Return an empty list only when NO candidate section is genuinely relevant.\n"
        "6. Confidence must be a float between 0.0 and 1.0.\n"
        "7. Output must be strict JSON only.\n"
        "8. Do not output markdown.\n"
        "9. Do not output backticks.\n"
        "10. Do not output additional commentary.\n"
        "11. Do not output additional keys.\n"
        "12. predicted_sections must contain at most one element.\n"
        "13. confidence must be a numeric value (not a string).\n"
        "14. Set confidence above 0.3 if the section is a reasonable match.\n\n"
        "Output Schema:\n"
        "{\n"
        "  \"predicted_sections\": [\"<section_number>\"] OR [],\n"
        "  \"confidence\": float,\n"
        "  \"explanation\": \"Plain English explanation.\"\n"
        "}\n\n"
        "Return ONLY valid JSON."
    )
    return prompt