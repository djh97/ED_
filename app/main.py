from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import EDDecisionResponse, EDRequest, EDStateSnapshot, EDStateUpdate, RawEDTextRequest
from app.orchestrator import EDOrchestrationAgent, evaluate_all_systems
from app.state_manager import EDStateManager

app = FastAPI(title=settings.app_name, version="0.1.0")
agent = EDOrchestrationAgent()
state_manager = EDStateManager()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/sample-case")
def sample_case() -> dict:
    return json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))


@app.post("/evaluate", response_model=EDDecisionResponse)
def evaluate_ed_state(ed_input: EDRequest) -> EDDecisionResponse:
    return agent.decide(ed_input)


@app.post("/state/evaluate", response_model=EDDecisionResponse)
def evaluate_and_store_state(ed_input: EDRequest) -> EDDecisionResponse:
    active_state = state_manager.update_from_full_snapshot(ed_input)
    response = agent.decide(active_state)
    pending = state_manager.merge_follow_up_plan(response.follow_up_plan)
    return response.model_copy(update={"pending_follow_up_count": len(pending), "follow_up_plan": pending})


@app.post("/state/update", response_model=EDDecisionResponse)
def update_ed_state(update: EDStateUpdate) -> EDDecisionResponse:
    active_state = state_manager.apply_update(update)
    response = agent.decide(active_state)
    pending = state_manager.merge_follow_up_plan(response.follow_up_plan)
    return response.model_copy(update={"pending_follow_up_count": len(pending), "follow_up_plan": pending})


@app.get("/state", response_model=EDStateSnapshot)
def get_ed_state() -> EDStateSnapshot:
    return state_manager.snapshot()


@app.post("/state/reset", response_model=EDStateSnapshot)
def reset_ed_state() -> EDStateSnapshot:
    return state_manager.reset()


@app.post("/evaluate-text", response_model=EDDecisionResponse)
def evaluate_ed_text(raw_input: RawEDTextRequest) -> EDDecisionResponse:
    return agent.decide(raw_input.text)


@app.get("/comparison")
def comparison() -> dict:
    return evaluate_all_systems()
