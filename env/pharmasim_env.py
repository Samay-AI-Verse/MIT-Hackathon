from typing import Any, Dict

from .drug_db import DrugDB
from .models import Action, Observation
from .utils import serialize_model


class PharmaSimEnv:
    def __init__(self, task_scenario):
        self.task = task_scenario
        self.state_data: Dict[str, Any] = {}
        self.done = False
        self.history = []
        self.db = DrugDB()
        self.reset()

    def reset(self) -> Observation:
        self.state_data = self.task.init_state()
        self.done = False
        self.history = []
        return Observation(**self.state_data["observation"])

    def step(self, action: Action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        obs, reward, done, info = self.task.step(self.state_data, action, self.db)
        self.state_data["observation"] = serialize_model(obs)
        self.done = done
        transition = {
            "action": serialize_model(action),
            "reward": serialize_model(reward),
            "observation": serialize_model(obs),
            "done": done,
            "info": info,
        }
        self.history.append(transition)
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return self.state_data
