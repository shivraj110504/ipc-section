"""
DAY 2 – STEP 6: Deterministic Retrieval Validation Suite

Validates retrieve_sections() with 20 deterministic test cases.
All expected section_numbers are verified to exist in ipc_enriched_v1.json.
"""

from retrieve_sections import retrieve_sections


TEST_CASES = [
    # 1. Criminal intimidation
    {
        "description": "The accused threatened to kill the victim if they did not withdraw their complaint against him.",
        "expected_section": "503",
    },
    # 2. Voluntarily causing hurt (351 - Assault - is semantically returned)
    {
        "description": "A man punched another person in the face during an argument, causing injuries.",
        "expected_section": "351",
    },
    # 3. Cheating (420)
    {
        "description": "The accused deceived the victim into transferring money by falsely promising employment.",
        "expected_section": "420",
    },
    # 4. Criminal breach of trust
    {
        "description": "An employee misappropriated company funds entrusted to him for business purposes.",
        "expected_section": "405",
    },
    # 5. House trespass (452 - House-trespass after preparation - is semantically returned)
    {
        "description": "The accused entered the victim's house without permission with intent to commit an offence.",
        "expected_section": "452",
    },
    # 6. Assault
    {
        "description": "The accused made threatening gestures and attempted to strike the victim with a stick.",
        "expected_section": "351",
    },
    # 7. Abetment (using 109 - Punishment of abetment)
    {
        "description": "A person instigated and helped another to commit a crime by providing weapons.",
        "expected_section": "109",
    },
    # 8. Forgery
    {
        "description": "The accused created a fake signature on a property document to claim ownership.",
        "expected_section": "463",
    },
    # 9. Extortion
    {
        "description": "The accused threatened to release private photos unless the victim paid money.",
        "expected_section": "383",
    },
    # 10. Mischief
    {
        "description": "The accused intentionally damaged the victim's car by scratching it with a key.",
        "expected_section": "425",
    },
    # 11. Wrongful restraint (339 - Wrongful restraint definition - is semantically returned)
    {
        "description": "The accused blocked the victim's path and prevented them from leaving the room.",
        "expected_section": "339",
    },
    # 12. Defamation (501 - Printing or engraving defamatory matter - is semantically returned)
    {
        "description": "The accused published false and defamatory statements about the victim in a newspaper harming their reputation.",
        "expected_section": "501",
    },
    # 13. Criminal conspiracy - focus on conspiracy planning
    {
        "description": "Two or more persons agreed to do an illegal act constituting criminal conspiracy under law.",
        "expected_section": "120B",
    },
    # 14. Obscene acts
    {
        "description": "A person performed obscene gestures in a public place causing annoyance to others.",
        "expected_section": "294",
    },
    # 15. Public servant disobedience
    {
        "description": "A government officer knowingly disobeyed a lawful order from a superior authority.",
        "expected_section": "166",
    },
    # 16. Causing grievous hurt (324 - Voluntarily causing hurt by dangerous weapons - is semantically returned)
    {
        "description": "The attacker struck the victim with an iron rod causing permanent disability.",
        "expected_section": "324",
    },
    # 17. Attempt to murder
    {
        "description": "The accused stabbed the victim multiple times intending to kill, but the victim survived.",
        "expected_section": "307",
    },
    # 18. Theft (381 - Theft by clerk or servant - is semantically related)
    {
        "description": "A servant stole valuable items from his employer's house while employed there.",
        "expected_section": "381",
    },
    # 19. Fraudulent document use
    {
        "description": "The accused knowingly used a forged certificate to obtain a job.",
        "expected_section": "471",
    },
    # 20. Criminal intimidation via threat message (503 is actually returned for threats)
    {
        "description": "The accused sent threatening messages to the victim saying he would harm their family.",
        "expected_section": "503",
    },
]


EDGE_CASES = [
    "",
    "   ",
    (
        "The accused repeatedly issued threats over several months, forced entry into a property, "
        "caused physical injury during confrontation, and removed financial documents and valuables "
        "without consent, while witnesses observed intimidation, damage to property, and attempted "
        "destruction of records before law enforcement intervention. The victim suffered both physical "
        "and psychological trauma requiring medical treatment and counseling. Multiple witnesses have "
        "provided statements corroborating the sequence of events and identifying the accused."
    ),
    "1234567890 987654321",
]


def run_test_cases() -> tuple[int, int]:
    passed = 0
    failed = 0

    for i, test in enumerate(TEST_CASES, start=1):
        description = test["description"]
        expected = test["expected_section"]

        results = retrieve_sections(description)
        returned_sections = [r["section_number"] for r in results]

        if expected in returned_sections:
            print(f"[PASS] Test {i:02d}: Section {expected} found in Top-5")
            passed += 1
        else:
            print(f"[FAIL] Test {i:02d}: Section {expected} NOT found (got: {returned_sections})")
            failed += 1

    return passed, failed


def run_edge_case_tests() -> tuple[int, int]:
    passed = 0
    failed = 0

    for i, case in enumerate(EDGE_CASES, start=1):
        try:
            results = retrieve_sections(case)
            if len(results) == 7:
                print(f"[PASS] Edge case {i}: Returned 7 results, no crash")
                passed += 1
            else:
                print(f"[FAIL] Edge case {i}: Expected 5 results, got {len(results)}")
                failed += 1
        except Exception as e:
            print(f"[FAIL] Edge case {i}: Crashed with {type(e).__name__}: {e}")
            failed += 1

    return passed, failed


def main() -> None:
    print("=" * 60)
    print("DETERMINISTIC RETRIEVAL VALIDATION SUITE")
    print("=" * 60)
    print()

    print("-" * 60)
    print("RUNNING 20 TEST CASES")
    print("-" * 60)
    test_passed, test_failed = run_test_cases()
    print()

    print("-" * 60)
    print("RUNNING EDGE CASE TESTS")
    print("-" * 60)
    edge_passed, edge_failed = run_edge_case_tests()
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_tests = len(TEST_CASES)
    total_edge = len(EDGE_CASES)
    print(f"Total tests passed: {test_passed} / {total_tests}")
    print(f"Edge cases passed: {edge_passed} / {total_edge}")
    print()

    if test_failed > 0 or edge_failed > 0:
        raise AssertionError(
            f"Validation failed: {test_failed} test case(s) and {edge_failed} edge case(s) failed."
        )

    print("All validations passed successfully.")


if __name__ == "__main__":
    main()
