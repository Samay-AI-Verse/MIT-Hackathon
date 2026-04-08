from pharmasim.env.models import PatientInfo, PrescriptionItem
from pharmasim.env.scenario_logic import build_initial_state, step_scenario


TASK_CONFIG = {
    "task_id": "medium_stockout_alternative",
    "difficulty": "medium",
    "description": "Recognize a stockout and recommend the safest in-stock substitute.",
    "goal": "suggest_alternative",
    "primary_medicine": "ibuprofen",
    "acceptable_alternatives": ["acetaminophen"],
    "reject_is_safe": False,
    "needs_clarification": False,
    "background_meds": [],
    "max_steps": 4,
    "observation": {
        "patient_info": PatientInfo(age=46, conditions=["osteoarthritis"]).dict(),
        "symptoms": ["joint pain", "morning stiffness"],
        "prescription": [
            PrescriptionItem(medicine="ibuprofen", dosage="400mg three times daily", valid=True).dict()
        ],
        "inventory": {"ibuprofen": 0, "acetaminophen": 14, "naproxen": 0},
        "urgency": "medium",
        "notes": "Prescriber allows therapeutic substitution if original NSAID is out of stock.",
    },
}


def init_state():
    return build_initial_state(TASK_CONFIG)


def step(state, action, db):
    return step_scenario(state, action, db)
