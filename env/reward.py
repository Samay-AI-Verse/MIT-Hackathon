from typing import Dict, List, Tuple

from .models import Reward
from .utils import action_signature, check_interactions, contraindication_hits, normalize_medicine, risk_score


def _delay_penalty(urgency: str, step_count: int) -> float:
    if urgency == "high" and step_count > 1:
        return -0.5
    if urgency == "medium" and step_count > 2:
        return -0.2
    return 0.0


def compute_reward(
    action,
    obs,
    db,
    prev_obs,
    urgency,
    step_count,
    max_steps,
    scenario: Dict,
    action_history: List[Dict],
    hidden_context: Dict,
) -> Tuple[Reward, bool, Dict]:
    details: List[str] = []
    info: Dict = {"task_id": scenario["task_id"]}
    done = False
    score = 0.0

    signature = action_signature(action)
    if action_history and signature == action_history[-1]:
        score -= 0.2
        details.append("repeated previous action")

    if action.action_type in {"dispense", "suggest_alternative"} and not signature["medicine"]:
        return Reward(score=-1.0, details="medicine is required for this action"), True, {
            **info,
            "result": "invalid_action",
            "risk": 1.0,
        }

    medicine = normalize_medicine(action.medicine)
    if medicine and not db.is_valid_medicine(medicine):
        return Reward(score=-1.0, details=f"{medicine} is not in formulary"), True, {
            **info,
            "result": "unknown_medicine",
            "risk": 1.0,
        }

    prescribed_meds = [normalize_medicine(item.medicine) for item in obs.prescription]
    valid_prescribed = [normalize_medicine(item.medicine) for item in obs.prescription if item.valid]
    invalid_prescribed = [normalize_medicine(item.medicine) for item in obs.prescription if not item.valid]
    available_alternatives = scenario.get("acceptable_alternatives", [])
    primary_medicine = normalize_medicine(scenario.get("primary_medicine"))
    background_meds = hidden_context.get("background_meds", [])
    interaction_pairs = (
        check_interactions([medicine] + valid_prescribed + list(background_meds), db)
        if medicine
        else []
    )
    contraindications = contraindication_hits(medicine, obs.patient_info.conditions, db)
    risk = risk_score(action, obs, db, background_meds=background_meds)

    if action.action_type == "request_info":
        if scenario.get("needs_clarification") and not hidden_context.get("clarified", False):
            score += 0.3
            info["result"] = "clarification_requested"
            details.append("requested missing or ambiguous information")
        else:
            score -= 0.1
            info["result"] = "unnecessary_request_info"
            details.append("request for extra information was not needed")

    elif action.action_type == "reject":
        reject_is_safe = scenario.get("reject_is_safe", False)
        if reject_is_safe and (invalid_prescribed or interaction_pairs or risk >= 0.8):
            score += 1.0
            done = True
            info["result"] = "safe_rejection"
            details.append("unsafe order rejected")
        elif scenario.get("needs_clarification") and not hidden_context.get("clarified", False):
            score += 0.3
            done = True
            info["result"] = "conservative_rejection"
            details.append("rejected before obtaining clarification")
        else:
            score -= 1.0
            done = True
            info["result"] = "incorrect_rejection"
            details.append("rejected a fillable order")

    elif action.action_type == "dispense":
        if medicine not in prescribed_meds:
            score -= 1.0
            done = True
            info["result"] = "wrong_medicine"
            details.append("medicine not present on prescription")
        elif medicine in invalid_prescribed:
            score -= 1.0
            done = True
            info["result"] = "dispensed_invalid_prescription_item"
            details.append("prescription item is invalid")
        elif prev_obs.inventory.get(medicine, 0) <= 0:
            score -= 1.0
            done = True
            info["result"] = "out_of_stock_dispense"
            details.append("dispensed unavailable medicine")
        elif interaction_pairs or contraindications or risk >= 0.8:
            score -= 1.0
            done = True
            info["result"] = "unsafe_dispense"
            if interaction_pairs:
                details.append(f"interaction risk: {interaction_pairs}")
            if contraindications:
                details.append(f"contraindicated for {', '.join(contraindications)}")
        elif medicine == primary_medicine and scenario.get("goal") == "dispense_primary":
            score += 1.0
            done = True
            info["result"] = "correct_dispense"
            details.append("correct medicine dispensed")
        elif medicine in available_alternatives and scenario.get("goal") == "dispense_alternative":
            score += 0.7
            done = True
            info["result"] = "acceptable_alternative_dispensed"
            details.append("safe alternative dispensed")
        else:
            score -= 1.0
            done = True
            info["result"] = "incorrect_dispense"
            details.append("dispense decision did not satisfy task goal")

    elif action.action_type == "suggest_alternative":
        if not medicine or prev_obs.inventory.get(medicine, 0) <= 0:
            score -= 1.0
            done = True
            info["result"] = "invalid_alternative"
            details.append("alternative unavailable")
        elif medicine not in available_alternatives:
            score -= 1.0
            done = True
            info["result"] = "unsupported_alternative"
            details.append("suggested alternative is not clinically preferred")
        elif interaction_pairs or contraindications or risk >= 0.8:
            score -= 1.0
            done = True
            info["result"] = "unsafe_alternative"
            details.append("suggested alternative is unsafe")
        else:
            score += 0.7
            done = True
            info["result"] = "valid_alternative"
            details.append("safe in-stock alternative suggested")

    score += _delay_penalty(urgency, step_count)
    if score < 0 and "delay" not in " ".join(details) and _delay_penalty(urgency, step_count) < 0:
        details.append("delay penalty applied due to urgency")

    if not done and step_count >= max_steps:
        score = min(score, -0.5)
        done = True
        info["result"] = "max_steps_exceeded"
        details.append("episode ended after exhausting step budget")

    final_score = max(-1.0, min(1.0, round(score, 2)))
    info["risk"] = risk
    return Reward(score=final_score, details="; ".join(details) or None), done, info
