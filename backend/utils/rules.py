# rule-based IPC validation
"""
Rule-based IPC hints.
Used to reinforce ML predictions safely.
"""

RULE_KEYWORDS = {
    "378": ["stolen", "theft", "took", "wallet", "mobile", "property"],
    "420": ["cheated", "fraud", "scam", "dishonest", "misrepresentation"],
    "323": ["assaulted", "hit", "hurt", "beaten"],
    "326": ["knife", "weapon", "grievous", "severe injury"],
    "506": ["threatened", "intimidation", "warning"],
    "354": ["harassed", "outraged", "molested", "woman"]
}

SENSITIVE_IPCS = {"326", "354"}

def rule_based_scores(text: str) -> dict:
    """
    Returns a score per IPC section based on keyword matches.
    """
    scores = {}
    matched_ipcs = set()
    for ipc, keywords in RULE_KEYWORDS.items():
        # score = 0
        for kw in keywords:
            if kw in text:
        #         score += 1
        # if score > 0:
        #     scores[ipc] = score
                scores[ipc] = scores.get(ipc, 0) + 1
                matched_ipcs.add(ipc)

    return scores, matched_ipcs
