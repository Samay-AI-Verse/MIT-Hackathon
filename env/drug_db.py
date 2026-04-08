from typing import Dict, List


class DrugDB:
    def __init__(self) -> None:
        self.medicines: Dict[str, Dict[str, List[str]]] = {
            "acetaminophen": {
                "alternatives": ["ibuprofen"],
                "interactions": [],
                "contraindications": ["severe liver disease"],
            },
            "amoxicillin": {
                "alternatives": ["penicillin", "azithromycin"],
                "interactions": ["warfarin"],
                "contraindications": ["penicillin allergy"],
            },
            "aspirin": {
                "alternatives": ["acetaminophen"],
                "interactions": ["ibuprofen", "warfarin"],
                "contraindications": ["peptic ulcer disease"],
            },
            "azithromycin": {
                "alternatives": ["amoxicillin"],
                "interactions": ["warfarin"],
                "contraindications": [],
            },
            "ibuprofen": {
                "alternatives": ["acetaminophen", "naproxen"],
                "interactions": ["warfarin", "aspirin", "clopidogrel"],
                "contraindications": ["chronic kidney disease", "peptic ulcer disease"],
            },
            "naproxen": {
                "alternatives": ["ibuprofen", "acetaminophen"],
                "interactions": ["warfarin", "aspirin", "clopidogrel"],
                "contraindications": ["chronic kidney disease", "peptic ulcer disease"],
            },
            "penicillin": {
                "alternatives": ["amoxicillin", "azithromycin"],
                "interactions": [],
                "contraindications": ["penicillin allergy"],
            },
            "warfarin": {
                "alternatives": [],
                "interactions": ["amoxicillin", "ibuprofen", "naproxen", "aspirin", "azithromycin"],
                "contraindications": ["active bleeding"],
            },
            "clopidogrel": {
                "alternatives": [],
                "interactions": ["ibuprofen", "naproxen"],
                "contraindications": [],
            },
        }

    def normalize(self, medicine: str) -> str:
        return medicine.strip().lower()

    def get_record(self, medicine: str) -> Dict[str, List[str]]:
        return self.medicines.get(self.normalize(medicine), {})

    def get_alternatives(self, medicine: str) -> List[str]:
        return list(self.get_record(medicine).get("alternatives", []))

    def get_interactions(self, medicine: str) -> List[str]:
        return list(self.get_record(medicine).get("interactions", []))

    def get_contraindications(self, medicine: str) -> List[str]:
        return list(self.get_record(medicine).get("contraindications", []))

    def is_valid_medicine(self, medicine: str) -> bool:
        return self.normalize(medicine) in self.medicines

    def has_interaction(self, medicine_a: str, medicine_b: str) -> bool:
        med_a = self.normalize(medicine_a)
        med_b = self.normalize(medicine_b)
        return med_b in self.get_interactions(med_a) or med_a in self.get_interactions(med_b)
