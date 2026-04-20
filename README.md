---
title: PharmaSim OpenEnv
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# PharmaSim: AI Pharmacy Decision-Making Environment

[![Hugging Face Space](https://img.shields.io/badge/??%20Space-Live_Demo-blue)](https://huggingface.co/spaces/Samay-Verse/pharmasim-openenv)
[![OpenEnv Benchmark](https://img.shields.io/badge/Benchmark-Meta_OpenEnv-black)](https://github.com/meta/openenv)

Deterministic, stateful benchmark for evaluating language model agents on pharmacy operations where safety, inventory, and urgency interact.

## Why This Environment

Static QA benchmarks do not capture operational decisions. Pharmacy workflows are sequential and constrained:
1. Check inventory before fulfillment.
2. Validate prescription safety and authenticity.
3. Handle substitution under clinical constraints.
4. Prioritize speed in urgent cases without compromising safety.

PharmaSim models these decisions as a strict environment with explicit rewards and failure modes.

## Live Deployment

- Hugging Face Space: https://huggingface.co/spaces/Samay-Verse/pharmasim-openenv

## Core Design

- Deterministic transitions and reward outcomes.
- Hidden-state clarification path via `request_info`.
- Hard safety boundary for contraindications and unsafe combinations.
- Urgency-based delay penalties.
- OpenEnv-compatible task packaging.

## Tasks

### Easy: Routine Fulfillment
- Goal: Dispense `ibuprofen` safely.
- Focus: Schema obedience and straightforward execution.

### Medium: Inventory Substitution
- Goal: Suggest `acetaminophen` when primary medicine is unavailable.
- Focus: Constraint checking and branching decisions.

### Hard: Unsafe Ambiguous Order
- Goal: Reject an urgent unsafe/invalid order.
- Focus: Safety-first behavior under ambiguity and time pressure.

## Observation and Action Space

### Observation
- `patient_info` (age, conditions)
- `symptoms`
- `prescription` (medicine, dosage, validity)
- `inventory`
- `urgency`
- `notes`

### Actions
- `dispense`
- `suggest_alternative`
- `reject`
- `request_info`

All actions must match `env.models.Action` JSON schema.

## Local Setup

```bash
git clone https://huggingface.co/spaces/Samay-Verse/pharmasim-openenv
cd pharmasim-openenv
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\Activate.ps1
```

Mac/Linux:
```bash
source .venv/bin/activate
```

Install deps:
```bash
pip install -r requirements.txt
```

## Run Baseline Inference

Create `.env`:

```env
API_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
```

Run:

```bash
python inference.py
```

Output format is evaluator-friendly:
- `[START]`
- `[STEP]`
- `[END]`

## Run API Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Endpoints:
- `GET /tasks`
- `POST /reset?task=medium`
- `POST /step`
- `GET /state`

## Benchmark Results Template

Fill this table with final measured results before submission.

| Model | Easy | Medium | Hard | Average |
|---|---:|---:|---:|---:|
| Deterministic Fallback | 1.00 | 1.00 | 1.00 | 1.00 |
| Model A | - | - | - | - |
| Model B | - | - | - | - |

## Repository Structure

```text
pharmasim/
  env/                 # Core environment logic and data checks
  graders/             # Deterministic grading functions
  tasks/               # Easy/Medium/Hard scenario configs
  training/            # GRPO training pipeline
  server/app.py        # FastAPI app
  inference.py         # Baseline runner with evaluator-style logs
  openenv.yaml         # OpenEnv metadata and entrypoint
  Dockerfile           # HF Space deployment container
```

## Submission Checklist (Meta Hackathon)

- OpenEnv metadata complete (`openenv.yaml`).
- Deterministic tasks + graders included.
- Local run command documented and reproducible.
- API endpoints available for step-by-step interaction.
- Benchmark table filled with final model scores.
- Short demo video showing easy/medium/hard runs.
