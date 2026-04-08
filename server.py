import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from pharmasim.env.models import Action
from pharmasim.env.pharmasim_env import PharmaSimEnv
from pharmasim.env.utils import serialize_model
from pharmasim.tasks import TASK_REGISTRY

app = FastAPI(title="PharmaSim", version="1.0.0")
CURRENT_TASK = "easy"
ENV = PharmaSimEnv(TASK_REGISTRY[CURRENT_TASK])


def _task_or_404(task_name: str):
    task = TASK_REGISTRY.get(task_name)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task '{task_name}'.")
    return task


@app.get("/")
def root():
    return {
        "name": "PharmaSim",
        "status": "ok",
        "current_task": CURRENT_TASK,
        "tasks": list(TASK_REGISTRY),
        "endpoints": ["/reset", "/step", "/state", "/tasks"],
    }


@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "task_id": module.TASK_CONFIG["task_id"],
            "difficulty": module.TASK_CONFIG["difficulty"],
            "description": module.TASK_CONFIG["description"],
        }
        for name, module in TASK_REGISTRY.items()
    }


@app.post("/reset")
def reset(task: str = "easy"):
    global CURRENT_TASK, ENV
    CURRENT_TASK = task
    ENV = PharmaSimEnv(_task_or_404(task))
    observation = ENV.reset()
    return {"task": task, "observation": serialize_model(observation)}


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = ENV.step(action)
    return {
        "observation": serialize_model(observation),
        "reward": serialize_model(reward),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return ENV.state()
