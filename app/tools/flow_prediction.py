from __future__ import annotations

from app.enhanced_models import DeterministicFlowAssessment
from app.models import EDRequest, FlowToolResult


_MODEL = DeterministicFlowAssessment()


def run_flow_prediction(ed_input: EDRequest) -> FlowToolResult:
    return _MODEL.predict(ed_input)
