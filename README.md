# Agentic AI Decision-Support System for Emergency Department Operations

This repository contains the implementation and reader-facing replication artifacts for an agentic AI decision-support system that coordinates emergency department (ED) operational recommendations and patient-care escalation.

The system combines LLM-based orchestration with four deterministic analytical adapters for patient risk, ED flow, staffing pressure, and bed capacity. It performs dynamic adapter selection, requires patient-risk screening when active patients are present, applies bounded safety validation, and creates structured follow-up objects.

> **Research use only.** This is a pre-clinical software evaluation. It has not been clinically validated, does not issue medical orders, and does not implement an executable human approval gate. The included benchmark contains synthetic scenarios only and no patient data.

## Repository contents

- `app/`: application, orchestration logic, deterministic adapters, schemas, prompts, and state management.
- `static/`: local user interface.
- `demo_data/`: synthetic example request.
- `evaluation/`: matched-experiment and plotting scripts.
- `results/`: preserved reader-facing outputs from the reported 4,500-record experiment.
- `figures/`: Figures 3(a), 3(b), and 4 generated from the preserved data.
- `tests/`: unit and regression tests.
- `BENCHMARK.md`: rule-derived benchmark and oracle definitions.
- `REPRODUCIBILITY.md`: exact environment, commands, output descriptions, and limitations.

Internal run logs, checkpoints, temporary files, manuscript drafts, and change-history notes are intentionally excluded.

## Installation

Python 3.12 is recommended. For the exact evaluated dependency versions:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.lock
```

Copy `.env.example` to `.env` or set the variables in your shell. Add your own API credential locally and never commit it:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-nano
```

## Run the application

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the local interface or `http://127.0.0.1:8000/docs` for the API documentation. A synthetic request is available at `demo_data/sample_case.json`.

## Quick offline verification

This command checks the experiment pipeline without an API call:

```bash
python evaluation/run_reproducible_experiment.py --offline-smoke-test --runs 2 --count 6 --bootstrap-samples 100 --output-dir evaluation_outputs/smoke
```

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

Two tests require a locally configured API credential; they are skipped when it is absent.

## Reported experiment

The preserved experiment evaluates 180 scenarios, five configurations, and five repeated runs, producing 4,500 scenario records. The main configuration uses `gpt-5-nano`, low reasoning effort for tool selection, medium reasoning effort for recommendation planning, and a maximum of two safety-correction attempts.

See `REPRODUCIBILITY.md` for the full command and a description of every included result file. Re-running the live-model experiment requires a valid API credential and may not produce byte-identical responses because the model alias and hosted service can change.

## Security and privacy

- No real patient records are included.
- No API credentials, access tokens, local environment files, or user-specific paths are included.
- Runtime logs and scenario checkpoints are not included.
- `.gitignore` excludes credentials, local configuration, logs, caches, checkpoints, and generated temporary outputs.

