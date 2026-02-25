from pydantic import BaseModel, Field


class CaseInput(BaseModel):
    text: str = Field(..., min_length=7)
