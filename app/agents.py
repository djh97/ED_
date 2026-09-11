from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.models import (
    AgentTraceItem,
    EDRequest,
    FollowUpItem,
    RecommendationItem,
)
from app.prompts import build_input_understanding_prompt

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass(frozen=True)
class InputUnderstandingResult:
    ed_request: EDRequest
    confidence: float
    notes: list[str]
    missing_fields: list[str]


class InputUnderstandingAgent:
    """Normalizes structured JSON or free-text ED notes into the EDRequest schema.

    Structured JSON is validated directly. Free-text ED notes require an LLM
    structured-output extraction step; the agent does not silently fall back to
    deterministic parsing.
    """

    def __init__(self, use_llm: bool = True, client: Any | None = None) -> None:
        self._client = client or (OpenAI(api_key=settings.openai_api_key) if (use_llm and OpenAI and settings.openai_api_key) else None)

    def normalize(self, raw_input: EDRequest | dict[str, Any] | str) -> InputUnderstandingResult:
        if isinstance(raw_input, EDRequest):
            return InputUnderstandingResult(
                ed_request=raw_input,
                confidence=1.0,
                notes=["Input already matched the EDRequest schema."],
                missing_fields=[],
            )

        if isinstance(raw_input, dict):
            request = EDRequest(**raw_input)
            return InputUnderstandingResult(
                ed_request=request,
                confidence=0.95,
                notes=["Structured dictionary was validated against the EDRequest schema."],
                missing_fields=[],
            )

        if not self._client:
            raise RuntimeError(
                "Input Understanding Agent requires OPENAI_API_KEY for free-text ED input. "
                "Provide structured JSON or configure the LLM."
            )

        return self._from_text_llm(raw_input)

    def _from_text_llm(self, text: str) -> InputUnderstandingResult:
        try:
            prompt = build_input_understanding_prompt(text)
            response = self._client.responses.create(
                model=settings.openai_input_model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt)}]}],
                reasoning={"effort": "low"},
            )
            parsed = json.loads(_extract_json_object(response.output_text))
            request = EDRequest(**parsed["ed_request"])
            return InputUnderstandingResult(
                ed_request=request,
                confidence=round(float(parsed.get("confidence", 0.8)), 2),
                notes=list(parsed.get("notes", ["LLM normalized free-text input into EDRequest schema."])),
                missing_fields=list(parsed.get("missing_fields", [])),
            )
        except Exception as exc:
            raise RuntimeError(f"Input Understanding Agent LLM extraction failed: {exc}") from exc

    def build_trace(self, result: InputUnderstandingResult) -> AgentTraceItem:
        evidence = [f"confidence={result.confidence}"]
        if result.missing_fields:
            evidence.append(f"defaulted={', '.join(result.missing_fields)}")
        evidence.extend(result.notes[:2])
        return AgentTraceItem(agent="input_understanding", step="normalized incoming ED data", evidence=evidence)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return text[start : end + 1]


class FollowUpTrackingAgent:
    """Creates trackable follow-up tasks for each recommended ED action."""

    ACTION_OWNERS = {
        "escalate_patient": ("attending_physician", 5),
        "reprioritize_queue": ("charge_nurse", 10),
        "reassign_bed": ("bed_manager", 15),
        "admit_support": ("admitting_team", 20),
        "discharge_support": ("flow_coordinator", 20),
        "staffing_alert": ("charge_nurse", 15),
        "monitor": ("triage_nurse", 30),
    }

    def create_plan(self, recommendations: list[RecommendationItem], system_state: str) -> list[FollowUpItem]:
        plan: list[FollowUpItem] = []
        for index, recommendation in enumerate(recommendations, start=1):
            owner, due = self.ACTION_OWNERS[recommendation.action]
            if system_state == "critical" and recommendation.priority == "urgent":
                due = max(3, due // 2)
            plan.append(
                FollowUpItem(
                    task_id=f"FU-{index:03d}",
                    linked_action=recommendation.action,
                    owner=owner,
                    due_minutes=due,
                    escalation_rule=(
                        f"If {recommendation.action} is not accepted within {due} minutes, "
                        "escalate to ED leadership for human review."
                    ),
                    reason=recommendation.reason,
                )
            )
        return plan

    def build_trace(self, plan: list[FollowUpItem]) -> AgentTraceItem:
        urgent_count = sum(1 for item in plan if item.due_minutes <= 10)
        return AgentTraceItem(
            agent="follow_up_tracking",
            step="created trackable follow-up tasks",
            evidence=[f"tasks={len(plan)}", f"urgent_followups={urgent_count}"],
        )
