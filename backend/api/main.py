#FastAPI IPC Prediction API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

from .schemas import CaseInput
from utils.ipc_actions_library import IPC_EXPLANATIONS
from utils.text_cleaning import clean_text
from utils.rules import rule_based_scores

app = FastAPI(title="IPC Prediction API")

# ----------- CORS setup -----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load ML artifacts
model = joblib.load("ml/ipc_model.pkl")
mlb = joblib.load("ml/ipc_labels.pkl")

# Load IPC law knowledge
ipc_df = pd.read_csv("data/processed/ipc_sections.csv")

@app.post("/ipc/predict")
def predict_ipc(case: CaseInput):
    #--------------No text added-----------------------------
    raw_text = case.text.strip()

    if not raw_text or len(raw_text) < 10:
        return {
            "prediction": None,
            "message": "Please describe the incident with sufficient details.",
            "disclaimer": "This tool requires incident details to provide a legal prediction."
        }

    raw_text = case.text
    cleaned_text = clean_text(raw_text)

    # ML prediction
    probs = model.predict_proba([cleaned_text])[0]
    ml_scores = dict(zip(mlb.classes_, probs))

    # Rule-based hints
    # rule_scores = rule_based_scores(cleaned_text)
    rule_scores, matched_ipcs = rule_based_scores(cleaned_text)

    # HARD OVERRIDE FOR SENSITIVE IPCs
    if "354" in matched_ipcs:
        ipc_code = "354"
        details = IPC_EXPLANATIONS["354"]

        return {
            "prediction": {
                "ipc_section": "IPC 354",
                "title": details["title"],
                "confidence": 75
            },
            "explanation": details["simple_explanation"],
            "why": details["why"],
            "suggestion": details["suggestion"],
            "disclaimer": "This is an AI-assisted legal awareness tool and does not constitute legal advice."
        }

    # Combine ML + rules
    # final_scores = {}
    # for ipc in ml_scores:
    #     final_scores[ipc] = ml_scores[ipc] + (0.05 * rule_scores.get(ipc, 0))

    final_scores = {}

    for ipc in ml_scores:
        # rule_score = rule_scores.get(ipc, 0)

        # Strong priority for sensitive IPCs
        # if ipc in ["354"] and rule_score > 0:
        #     final_scores[ipc] = ml_scores[ipc] + (0.4 * rule_score)
        # else:
        #     final_scores[ipc] = ml_scores[ipc] + (0.15 * rule_score)
        final_scores[ipc] = ml_scores[ipc] + (0.15 * rule_scores.get(ipc, 0))

    # Select best IPC
    predicted_ipc = max(final_scores, key=final_scores.get)
    # confidence = min(int(final_scores[predicted_ipc] * 100), 85)
    confidence = min(
    int((final_scores[predicted_ipc] / sum(final_scores.values())) * 100),
    90)


    # Fetch IPC explanation
    matching_rows = ipc_df[ipc_df["section_number"] == int(predicted_ipc)]
    if matching_rows.empty:
        ipc_row = {"section_title": "Section details not available", "section_text": "No detailed explanation available in the database."}
    else:
        ipc_row = matching_rows.iloc[0]

    explanation_text = ipc_row["section_text"] if pd.notna(ipc_row["section_text"]) else "Detailed section text not available in the database."
    explanation = explanation_text[:400] + "..." if len(explanation_text) > 400 else explanation_text

    ipc_code = str(predicted_ipc)

    details = IPC_EXPLANATIONS.get(ipc_code)

    if details:
        return {
            "prediction": {
                "ipc_section": f"IPC {ipc_code}",
                "title": details["title"],
                "confidence": confidence
            },
            "explanation": details["simple_explanation"],
            "why": details["why"],
            "suggestion": details["suggestion"],
            "disclaimer": "This is an AI-assisted legal awareness tool."
        }
    else:
        return {
            "prediction": {
                "ipc_section": f"IPC {ipc_code}",
                "title": "Section details not available",
                "confidence": confidence
            },
            "explanation": "Detailed explanation not available for this section.",
            "why": "This section may not have predefined explanations in the system.",
            "suggestion": "Consult legal resources or authorities for more information.",
            "disclaimer": "This is an AI-assisted legal awareness tool."
        }


# @app.get("/")
# def root():
#     return {
#         "status": "ok",
#         "service": "IPC Prediction API"
#     }
