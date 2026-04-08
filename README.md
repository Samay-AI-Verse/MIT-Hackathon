---
title: Pharmasim Openenv
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# PharmaSim: AI Pharmacy Decision-Making Environment

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live_Demo-blue)](https://huggingface.co/spaces/Samay-Verse/pharmasim-openenv)
[![OpenEnv Benchmark](https://img.shields.io/badge/Benchmark-Meta_OpenEnv-black)](https://github.com/meta/openenv)

*A deterministic, stateful benchmark for evaluating Large Language Models on complex, real-world pharmacy operations.*

## The Core Problem

Most benchmarks evaluate AI safety and reasoning through static multiple-choice questions. In the real world, especially in high-stakes environments like pharmacies, decision-making is **stateful, ambiguous, and constrained**. 

When a prescription comes in, a human pharmacist doesn't just read it; they:
1. Check live inventory.
2. Cross-reference the patient's medical history for contraindications.
3. Make judgment calls on substituting an out-of-stock medication.
4. Escalate or reject ambiguous orders based on *urgency*.

**PharmaSim** forces AI agents to handle this exact layered logic. It’s an OpenEnv-compliant environment that scores an agent's ability to act safely and efficiently across multiple steps, separating narrow chatbots from true reasoning engines.

## Live Deployment

You can interact with our live environment API hosted directly on Hugging Face Spaces:
👉 **[Samay-Verse/pharmasim-openenv](https://huggingface.co/spaces/Samay-Verse/pharmasim-openenv)**

---

## 🚀 Advanced Reinforcement Learning Pipeline (GRPO via TRL)

To elevate this submission from a simple environment to a breakthrough training platform, we have integrated a **Group Relative Policy Optimization (GRPO)** training pipeline using Transformers Reinforcement Learning (TRL) and vLLM.

Instead of writing rule-based logic or doing few-shot prompting, a lightweight open-source agent (like `Qwen-1.5B`) uses the `training/train_grpo.py` pipeline to **literally teach itself** how to be a pharmacist through pure trial, error, and mathematical reinforcement. 

This sets our implementation completely apart and demonstrates a capability perfectly parallel to the official Meta OpenEnv examples:
1. The agent chooses a dispensing action in strict JSON format.
2. It interacts natively with the PharmaSim validation boundary.
3. It receives mathematical reward feedback (0.0 to 1.0) and uses `vLLM` within `TRL`'s rollout functions to rapidly optimize its policy gradients.

*Check out `training/train_grpo.py` to see the cutting-edge pipeline in action!*

---

## Innovation Hooks (Why this benchmark stands out)

- **Deterministic Sandbox**: Unlike typical text-based role-play games, PharmaSim has a rigid, mathematically proven risk-scoring engine. The `graders` calculate exact scores `[0.0, 1.0]` based on the sequence of actions.
- **Hidden State & Ambiguity**: The agent doesn't have all the answers upfront. Using the `request_info` action updates the state and reveals hidden notes (mimicking calling a doctor to clarify a messy handwritten note).
- **Strict Substitution Logic**: The engine incorporates real-world mechanics. For example, suggesting Acetaminophen when Ibuprofen is out of stock is scored highly—unless the patient's file flags a liver condition.
- **Urgency Penalties**: In urgent cases, taking too many turns to `request_info` applies a time-delay penalty to the final reward.

---

## The Tasks

We've structured the evaluation into three distinct tiers:

### 🥉 Task 1: Easy (Routine Fulfillment)
- **Scenario**: Valid, clear prescription. Medicine is completely in stock.
- **Agent Goal**: Safely dispense `ibuprofen`.
- **Difficulty Focus**: Basic JSON schema obedience and simple state-reading.

### 🥈 Task 2: Medium (Inventory & Substitution)
- **Scenario**: The primary prescribed medicine is out of stock, but a safe clinical substitute is available in the inventory.
- **Agent Goal**: Recognize the stock shortage and suggest `acetaminophen` as an alternative.
- **Difficulty Focus**: Checking constraints before acting and executing logic branches.

### 🥇 Task 3: Hard (High-Stakes Ambiguity)
- **Scenario**: Urgent case. The prescription includes an invalid antibiotic add-on and a severe interaction risk (e.g., anticoagulation conflicts).
- **Agent Goal**: Immediately reject the order, or request clarification safely before rejecting.
- **Difficulty Focus**: Prioritizing safety over fulfillment speed; recognizing complex, multi-drug interactions without explicit prompting.

---

## Observation & Action Space

### Observation Space (What the AI sees)
The environment returns a compact but highly realistic context buffer:
*   `patient_info` (age, prior conditions)
*   `symptoms`
*   `prescription` (medicine, dosage, validity flag)
*   `inventory` (live stock count)
*   `urgency` (low | medium | high)

### Action Space (What the AI can do)
Agents must return strict JSON matching exactly one of these actions:
*   `dispense`: Process the order.
*   `suggest_alternative`: Propose a substitute if constrained.
*   `reject`: Stop the process due to safety or invalidity.
*   `request_info`: Ask for clarification (costs a step).

---

## Setup & Running Locally

If you want to run the baseline models (or test your own) against our environment:

### 1. Installation
```bash
git clone https://huggingface.co/spaces/Samay-Verse/pharmasim-openenv
cd pharmasim-openenv
python -m venv .venv

# On Windows:
.venv\Scripts\Activate.ps1

# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Testing Models
To test the environment locally, we've provided an `inference.py` script. You can use Groq's high-speed Llama models (or default OpenAI) to run the baseline.

Create a `.env` file in the root directory:
```env
# Example using Groq for lightning-fast inference
API_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
```

Run the deterministic benchmark:
```bash
python inference.py
```
*You will see the `[START]`, `[STEP]`, and `[END]` logs automatically generate as the AI steps through the pharmacy simulation—these logs are formatted exactly for the hackathon automated evaluators.*

### 3. Spin up the Local Server
```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```
Once running, you can hit the local endpoints directly to test the environment step-by-step:
*   `GET /tasks`
*   `POST /reset?task=medium`
*   `POST /step` (with JSON action payload)

---

## Repository Structure

```text
pharmasim/
  env/                 # Core engine, drug DB, and transition logic
  graders/             # Deterministic scoring logic for [0.0, 1.0] grading
  tasks/               # Scenario configurations
  inference.py         # Baseline agent script (outputs Hackathon-compliant logs)
  openenv.yaml         # Environment schema definition for external testing
  server.py            # Local FastAPI inference server
  Dockerfile           # HF Space ready container
```