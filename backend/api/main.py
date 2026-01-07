#FastAPI IPC Prediction API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

from api.schemas import CaseInput
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
model = joblib.load("ipc_model.pkl")
mlb = joblib.load("ipc_labels.pkl")

# Load IPC law knowledge
ipc_df = pd.read_csv("data/processed/ipc_sections.csv")

@app.post("/ipc/predict")
def predict_ipc(case: CaseInput):
    raw_text = case.text
    cleaned_text = clean_text(raw_text)

    # ML prediction
    probs = model.predict_proba([cleaned_text])[0]
    ml_scores = dict(zip(mlb.classes_, probs))

    # Rule-based hints
    rule_scores = rule_based_scores(cleaned_text)

    # Combine ML + rules
    final_scores = {}
    for ipc in ml_scores:
        final_scores[ipc] = ml_scores[ipc] + (0.05 * rule_scores.get(ipc, 0))

    # Select best IPC
    predicted_ipc = max(final_scores, key=final_scores.get)
    confidence = min(int(final_scores[predicted_ipc] * 100), 85)

    # Fetch IPC explanation
    matching_rows = ipc_df[ipc_df["section_number"] == int(predicted_ipc)]
    if matching_rows.empty:
        ipc_row = {"section_title": "Section details not available", "section_text": "No detailed explanation available in the database."}
    else:
        ipc_row = matching_rows.iloc[0]

    explanation_text = ipc_row["section_text"] if pd.notna(ipc_row["section_text"]) else "Detailed section text not available in the database."
    explanation = explanation_text[:400] + "..." if len(explanation_text) > 400 else explanation_text

    return {
        "input_text": raw_text,
        "prediction": {
            "ipc_section": f"IPC {predicted_ipc}",
            "title": ipc_row["section_title"],
            "confidence": f"{confidence}%"
        },
        "explanation": explanation,
        "suggestion": "It is advisable to consult a legal professional or approach appropriate authorities.",
        "disclaimer": "This is an AI-assisted legal awareness tool and does not constitute legal advice."
    }

# @app.get("/")
# def root():
#     return {
#         "status": "ok",uvicorn api.main:app --reload

#         "service": "IPC Prediction API"
#     }
