"""Compatibility imports for the canonical orchestration implementation.

All runtime and evaluation behavior lives in :mod:`app.agentic_system`. Keeping
this module import-only prevents legacy implementations from diverging while
preserving existing API and script imports.
"""

from app.agentic_system import (  # noqa: F401
    CrowdingScoreBaseline,
    DecisionRunner,
    EDOrchestrationAgent,
    ESITriageBaseline,
    EarlyWarningScoreBaseline,
    EvaluationScenario,
    NonAgenticIntegratedBaseline,
    PredictionOnlyBaseline,
    RuleBasedEDBaseline,
    ScenarioScore,
    build_core_evaluation_scenarios,
    build_evaluation_scenarios,
    build_mixed_evaluation_scenarios,
    evaluate_all_systems,
    evaluate_runner,
    score_response,
    summarize_scores,
)

__all__ = [
    "CrowdingScoreBaseline",
    "DecisionRunner",
    "EDOrchestrationAgent",
    "ESITriageBaseline",
    "EarlyWarningScoreBaseline",
    "EvaluationScenario",
    "NonAgenticIntegratedBaseline",
    "PredictionOnlyBaseline",
    "RuleBasedEDBaseline",
    "ScenarioScore",
    "build_core_evaluation_scenarios",
    "build_evaluation_scenarios",
    "build_mixed_evaluation_scenarios",
    "evaluate_all_systems",
    "evaluate_runner",
    "score_response",
    "summarize_scores",
]
