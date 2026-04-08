from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Action, Observation


def normalize_medicine(medicine: Optional[str]) -> Optional[str]:
    if medicine is None:
        return None
    cleaned = medicine.strip().lower()
    return cleaned or None


def serialize_model(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def check_interactions(medicines: Sequence[str], db) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    normalized = [normalize_medicine(medicine) for medicine in medicines if normalize_medicine(medicine)]
    for index, first in enumerate(normalized):
        for second in normalized[index + 1 :]:
            if db.has_interaction(first, second):
                pairs.append((first, second))
    return pairs


def contraindication_hits(medicine: Optional[str], conditions: Iterable[str], db) -> List[str]:
    if not medicine:
        return []
    normalized_conditions = {condition.strip().lower() for condition in conditions}
    return [
        condition
        for condition in db.get_contraindications(medicine)
        if condition.strip().lower() in normalized_conditions
    ]


def risk_score(
    action: Action,
    obs: Observation,
    db,
    background_meds: Optional[Sequence[str]] = None,
) -> float:
    if action.action_type in {"dispense", "suggest_alternative"} and not action.medicine:
        return 1.0

    medicine = normalize_medicine(action.medicine)
    if medicine is None:
        return 0.1 if action.action_type == "request_info" else 0.0

    if not db.is_valid_medicine(medicine):
        return 1.0

    risk = 0.0
    prescribed = [normalize_medicine(item.medicine) for item in obs.prescription]
    valid_prescribed = [normalize_medicine(item.medicine) for item in obs.prescription if item.valid]
    active_meds = list(valid_prescribed)
    for current in background_meds or []:
        normalized = normalize_medicine(current)
        if normalized:
            active_meds.append(normalized)

    if action.action_type == "dispense" and medicine not in prescribed:
        risk += 0.7
    if action.action_type == "dispense" and obs.inventory.get(medicine, 0) <= 0:
        risk += 0.9
    if action.action_type == "suggest_alternative" and obs.inventory.get(medicine, 0) <= 0:
        risk += 0.7

    if contraindication_hits(medicine, obs.patient_info.conditions, db):
        risk += 0.8

    for active_med in active_meds:
        if active_med != medicine and db.has_interaction(medicine, active_med):
            risk += 0.9
            break

    return min(1.0, round(risk, 2))


def action_signature(action: Action) -> Dict[str, Optional[str]]:
    return {
        "action_type": action.action_type,
        "medicine": normalize_medicine(action.medicine),
        "reason": (action.reason or "").strip().lower() or None,
    }
