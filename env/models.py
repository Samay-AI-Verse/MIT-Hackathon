from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    age: int = Field(..., ge=0, le=120)
    conditions: List[str] = Field(default_factory=list)


class PrescriptionItem(BaseModel):
    medicine: str
    dosage: str
    valid: bool


class Observation(BaseModel):
    patient_info: PatientInfo
    symptoms: List[str] = Field(default_factory=list)
    prescription: List[PrescriptionItem] = Field(default_factory=list)
    inventory: Dict[str, int] = Field(default_factory=dict)
    urgency: Literal["low", "medium", "high"]
    notes: Optional[str] = None


class Action(BaseModel):
    action_type: Literal["dispense", "suggest_alternative", "reject", "request_info"]
    medicine: Optional[str] = None
    reason: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    details: Optional[str] = None
