from __future__ import annotations

from app.enhanced_models import DeterministicBedCapacityAssessment
from app.models import BedToolResult, EDRequest


_MODEL = DeterministicBedCapacityAssessment()


def run_bed_management(ed_input: EDRequest) -> BedToolResult:
    return _MODEL.predict(ed_input)
