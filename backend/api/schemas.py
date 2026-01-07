# request / response models
from pydantic import BaseModel, Field
from typing import List

class CaseInput(BaseModel):
    text: str = Field(..., min_length=7)

class PredictionResponse(BaseModel):
    ipc_section: str
    confidence: str
    explanation: str
    suggestion: str
    disclaimer: str
