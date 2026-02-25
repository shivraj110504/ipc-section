import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from script.schemas import CaseInput
    from script.ipc_reasoning_engine import predict_ipc_section
except ImportError:
    from schemas import CaseInput
    from ipc_reasoning_engine import predict_ipc_section

app = FastAPI(title="IPC Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUGGESTIONS = [
    "Consider consulting a legal professional.",
    "You may approach the nearest police station.",
    "Document all relevant evidence.",
    "Ensure your safety before taking further action.",
    "Seek immediate help if the situation escalates.",
]


@app.post("/ipc/predict")
def predict_ipc(case: CaseInput):
    raw_text = case.text.strip()

    if not raw_text or len(raw_text) < 10:
        return {
            "prediction": None,
            "message": "Please describe the incident with sufficient details.",
            "disclaimer": "This tool requires incident details to provide a legal prediction.",
        }

    rag_output = predict_ipc_section(raw_text)

    if rag_output.get("predicted_sections"):
        ipc_code = rag_output["predicted_sections"][0]
        title = rag_output.get("title", "")
        confidence = round(rag_output.get("confidence", 0.0) * 100)
    else:
        ipc_code = None
        title = ""
        confidence = 0

    suggestion = random.choice(SUGGESTIONS)

    return {
        "prediction": {
            "ipc_section": f"IPC {ipc_code}" if ipc_code else None,
            "title": title,
            "confidence": confidence,
        },
        "explanation": rag_output.get("explanation", ""),
        "suggestion": suggestion,
        "disclaimer": "This is an AI-assisted legal awareness tool.",
    }
