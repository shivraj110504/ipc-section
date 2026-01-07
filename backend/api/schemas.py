# request / response models
from pydantic import BaseModel
from typing import List

class CaseInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    ipc_section: str
    confidence: str
    explanation: str
    suggestion: str
    disclaimer: str
