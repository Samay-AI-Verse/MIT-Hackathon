from pharmasim.env.models import PatientInfo, PrescriptionItem
from pharmasim.env.scenario_logic import build_initial_state, step_scenario


TASK_CONFIG = {
    "task_id": "hard_unsafe_invalid_combo",
    "difficulty": "hard",
    "description": "Handle an urgent, partially invalid order with a potential high-risk interaction.",
    "goal": "reject_unsafe",
    "primary_medicine": "amoxicillin",
    "acceptable_alternatives": [],
    "reject_is_safe": True,
    "needs_clarification": True,
    "clarification_note": (
        "Clarification from nurse: the antibiotic add-on was missing prescriber authentication, "
        "and the patient is actively anticoagulated on warfarin."
    ),
    "background_meds": ["warfarin"],
    "max_steps": 5,
    "observation": {
        "patient_info": PatientInfo(age=71, conditions=["atrial fibrillation"]).dict(),
        "symptoms": ["facial swelling", "post-procedure dental pain"],
        "prescription": [
            PrescriptionItem(medicine="warfarin", dosage="5mg nightly", valid=True).dict(),
            PrescriptionItem(medicine="amoxicillin", dosage="500mg three times daily", valid=False).dict(),
        ],
        "inventory": {"warfarin": 6, "amoxicillin": 11, "ibuprofen": 7, "acetaminophen": 20},
        "urgency": "high",
        "notes": "Handwritten antibiotic add-on is difficult to verify; recent oral surgery noted.",
    },
}


def init_state():
    return build_initial_state(TASK_CONFIG)


def step(state, action, db):
    return step_scenario(state, action, db)
