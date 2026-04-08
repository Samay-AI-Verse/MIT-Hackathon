from copy import deepcopy
from typing import Dict

from .models import Observation
from .reward import compute_reward
from .utils import action_signature, serialize_model


def build_initial_state(config: Dict) -> Dict:
    observation = Observation(**config["observation"])
    return {
        "task_id": config["task_id"],
        "difficulty": config["difficulty"],
        "description": config["description"],
        "scenario": deepcopy(config),
        "observation": serialize_model(observation),
        "step_count": 0,
        "max_steps": config.get("max_steps", 5),
        "clarified": False,
        "history": [],
        "done": False,
    }


def _apply_clarification(state: Dict) -> None:
    scenario = state["scenario"]
    if state.get("clarified") or not scenario.get("clarification_note"):
        return

    observation = Observation(**state["observation"])
    notes = observation.notes or ""
    observation.notes = (
        f"{notes} {scenario['clarification_note']}".strip() if notes else scenario["clarification_note"]
    )
    state["observation"] = serialize_model(observation)
    state["clarified"] = True


def step_scenario(state: Dict, action, db):
    previous_observation = Observation(**state["observation"])
    state["step_count"] += 1
    was_clarified = state.get("clarified", False)

    if action.action_type == "request_info":
        _apply_clarification(state)

    current_observation = Observation(**state["observation"])
    reward, done, info = compute_reward(
        action=action,
        obs=current_observation,
        prev_obs=previous_observation,
        db=db,
        urgency=current_observation.urgency,
        step_count=state["step_count"],
        max_steps=state["max_steps"],
        scenario=state["scenario"],
        action_history=state["history"],
        hidden_context={
            "clarified": was_clarified,
            "background_meds": state["scenario"].get("background_meds", []),
        },
    )

    state["done"] = done
    state["history"].append(action_signature(action))
    return current_observation, reward, done, info
