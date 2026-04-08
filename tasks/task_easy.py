from pharmasim.env.models import PatientInfo, PrescriptionItem
from pharmasim.env.scenario_logic import build_initial_state, step_scenario


TASK_CONFIG = {
    "task_id": "easy_valid_dispense",
    "difficulty": "easy",
    "description": "Dispense a valid, in-stock medicine for a straightforward prescription.",
    "goal": "dispense_primary",
    "primary_medicine": "ibuprofen",
    "acceptable_alternatives": ["acetaminophen"],
    "reject_is_safe": False,
    "needs_clarification": False,
    "background_meds": [],
    "max_steps": 4,
    "observation": {
        "patient_info": PatientInfo(age=30, conditions=["hypertension"]).dict(),
        "symptoms": ["headache", "mild fever"],
        "prescription": [
            PrescriptionItem(medicine="ibuprofen", dosage="200mg twice daily", valid=True).dict()
        ],
        "inventory": {"ibuprofen": 18, "acetaminophen": 9, "aspirin": 4},
        "urgency": "low",
        "notes": "Patient confirms no known drug allergies.",
    },
}


def init_state():
    return build_initial_state(TASK_CONFIG)


def step(state, action, db):
    return step_scenario(state, action, db)
