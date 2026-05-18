import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from env.models import Action
from env.pharmasim_env import PharmaSimEnv
from env.utils import serialize_model
from graders import grader_easy, grader_hard, grader_medium
from tasks import task_easy, task_hard, task_medium

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Internal mapping for the client builder script
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
TEMPERATURE = 0.0
MAX_TOKENS = 160
MAX_STEPS = 5

TASKS = [
    ("easy", task_easy, grader_easy),
    ("medium", task_medium, grader_medium),
    ("hard", task_hard, grader_hard),
]


def build_client():
    if not OPENAI_API_KEY or not MODEL_NAME:
        return None
    kwargs = {"api_key": OPENAI_API_KEY}
    if API_BASE_URL:
        kwargs["base_url"] = API_BASE_URL
    return OpenAI(**kwargs)


def observation_json(observation) -> str:
    if hasattr(observation, "model_dump_json"):
        return observation.model_dump_json(indent=2)
    return observation.json(indent=2)


def parse_model_action(response_text: str):
    if not response_text:
        return None
    fenced = re.search(r"\{.*\}", response_text, re.DOTALL)
    candidate = fenced.group(0) if fenced else response_text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    try:
        return Action(**payload)
    except Exception:
        return None


def fallback_action(observation, step_index: int) -> Action:
    invalid_items = [item for item in observation.prescription if not item.valid]
    in_stock_primary = None
    for item in observation.prescription:
        if item.valid and observation.inventory.get(item.medicine.lower(), 0) > 0:
            in_stock_primary = item.medicine.lower()
            break

    if invalid_items:
        # For urgent, explicitly unsafe/invalid orders, immediate rejection is the safest action.
        if step_index == 0 and observation.urgency == "high":
            return Action(action_type="reject", reason="Urgent and unsafe/invalid prescription item present.")
        return Action(action_type="reject", reason="Invalid or high-risk prescription item.")

    if in_stock_primary:
        return Action(action_type="dispense", medicine=in_stock_primary, reason="Valid prescription and stock available.")

    if observation.inventory.get("acetaminophen", 0) > 0:
        return Action(
            action_type="suggest_alternative",
            medicine="acetaminophen",
            reason="Primary medicine unavailable; safe substitute in stock.",
        )

    return Action(action_type="request_info", reason="Unable to safely complete order with current data.")


def query_model(client, observation):
    if client is None:
        return None, "No API credentials provided. Using deterministic fallback policy."

    prompt = (
        "You are evaluating a pharmacy operation. Respond with JSON only matching the Action schema:\n"
        '{"action_type":"dispense|suggest_alternative|reject|request_info","medicine":null,"reason":"..."}\n'
        f"Observation:\n{observation_json(observation)}"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a careful pharmacy operations agent. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = completion.choices[0].message.content or ""
    except Exception as exc:
        return None, f"Model request failed ({exc}). Using deterministic fallback policy."

    action = parse_model_action(text)
    if action is None:
        return None, "Model response was not valid Action JSON. Using deterministic fallback policy."
    return action, None


def run_task(task_name, task_module, grader_module, client):
    env = PharmaSimEnv(task_module)
    observation = env.reset()
    trajectory = []

    print(f"[START] task={task_name} env=PharmaSim model={MODEL_NAME}")
    rewards_list = []
    
    for step_index in range(MAX_STEPS):
        model_action, warning = query_model(client, observation)
        action = model_action or fallback_action(observation, step_index)
        
        observation, reward, done, info = env.step(action)
        transition = {
            "action": serialize_model(action),
            "reward": serialize_model(reward),
            "observation": serialize_model(observation),
            "done": done,
            "info": info,
        }
        trajectory.append(transition)
        rewards_list.append(f"{reward.score:.2f}")
        
        # Format the print string to meet the required hackathon logs
        action_str = action.json() if hasattr(action, 'json') else str(action)
        action_str = action_str.replace('\n', ' ').replace('\r', '')
        error_msg = f'"{warning}"' if warning else "null"
        done_str = "true" if done else "false"
        
        print(f"[STEP] step={step_index + 1} action={action_str} reward={reward.score:.2f} done={done_str} error={error_msg}", flush=True)
        if done:
            break

    score = grader_module.grade(trajectory)
    # Ensure score is strictly between 0 and 1 (hackathon requirement)
    score = max(0.01, min(0.99, score))
    success = "true" if score >= 0.9 else "false"
    
    rewards_str = ",".join(rewards_list)
    print(f"[END] success={success} steps={len(trajectory)} rewards={rewards_str}", flush=True)
    return score


def main():
    client = build_client()
    scores = {}
    for task_name, task_module, grader_module in TASKS:
        scores[task_name] = run_task(task_name, task_module, grader_module, client)

    print("\nFinal Scores")
    for task_name, score in scores.items():
        print(f"{task_name}: {score:.2f}")


if __name__ == "__main__":
    main()
