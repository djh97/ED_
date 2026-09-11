from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from app.agents import FollowUpTrackingAgent
from app.agentic_system import (
    DecisionRunner,
    EDOrchestrationAgent,
    NonAgenticIntegratedBaseline,
    PredictionOnlyBaseline,
    RuleBasedEDBaseline,
    build_evaluation_scenarios,
    evaluate_runner,
    score_response,
    summarize_scores,
    _build_action_brief,
    _build_agent_trace,
    _derive_system_state,
    _failed_scenario_score,
    _required_escalation_feedback,
)
from app.models import EDDecisionResponse, EDRequest, EDStateUpdate, RecommendationItem, ToolOutputs
from app.state_manager import EDStateManager
from app.tools.bed_management import run_bed_management
from app.tools.flow_prediction import run_flow_prediction
from app.tools.patient_risk import run_patient_risk
from app.tools.staffing import run_staffing_availability


def evaluate_ablation_baselines() -> dict[str, dict[str, float]]:
    """Runs the component-style ablation baselines on the 180-scenario set."""

    systems: dict[str, DecisionRunner] = {
        "prediction_only_no_action_planner": PredictionOnlyBaseline(),
        "non_agentic_fixed_mapping": NonAgenticIntegratedBaseline(),
        "rule_based_no_llm_reasoning": RuleBasedEDBaseline(),
    }
    return {name: evaluate_runner(name, runner)["summary_metrics"] for name, runner in systems.items()}


@dataclass(frozen=True)
class ComponentContributionConfig:
    name: str
    label: str
    enable_safety_validation: bool
    enable_state_management: bool
    enable_follow_up_tracking: bool
    enable_dynamic_tool_selection: bool = True
    use_llm_orchestration: bool = True


COMPONENT_CONTRIBUTION_CONFIGS = (
    ComponentContributionConfig(
        name="full_framework",
        label="Full framework",
        enable_safety_validation=True,
        enable_state_management=True,
        enable_follow_up_tracking=True,
    ),
    ComponentContributionConfig(
        name="without_safety_validation",
        label="Without safety validation",
        enable_safety_validation=False,
        enable_state_management=True,
        enable_follow_up_tracking=True,
    ),
    ComponentContributionConfig(
        name="without_dynamic_tool_selection",
        label="Without dynamic tool selection (all four tools)",
        enable_safety_validation=True,
        enable_state_management=True,
        enable_follow_up_tracking=True,
        enable_dynamic_tool_selection=False,
    ),
    ComponentContributionConfig(
        name="without_follow_up_tracking",
        label="Without follow-up tracking",
        enable_safety_validation=True,
        enable_state_management=True,
        enable_follow_up_tracking=False,
    ),
    ComponentContributionConfig(
        name="without_llm_orchestration",
        label="Without LLM-based orchestration",
        enable_safety_validation=True,
        enable_state_management=True,
        enable_follow_up_tracking=True,
        enable_dynamic_tool_selection=False,
        use_llm_orchestration=False,
    ),
)


class ComponentReferenceRunner:
    """Framework-equivalent component runner for offline ablation.

    The runner keeps the same four prediction/assessment tools as the full
    system, then toggles only the architectural modules under study.
    """

    def __init__(self, enable_safety_validation: bool, enable_follow_up_tracking: bool) -> None:
        self.enable_safety_validation = enable_safety_validation
        self.enable_follow_up_tracking = enable_follow_up_tracking
        self.follow_up_agent = FollowUpTrackingAgent()

    def decide(self, ed_input: EDRequest) -> EDDecisionResponse:
        tool_outputs = _tool_outputs_for_request(ed_input)
        top_patient = tool_outputs.patient_risk.flagged_patients[0] if tool_outputs.patient_risk.flagged_patients else None
        system_state = _derive_system_state(
            tool_outputs.flow_prediction.bottleneck_level,
            tool_outputs.staffing.staffing_level,
            tool_outputs.bed_management.action_window,
            top_patient,
        )
        recommendations = self._initial_plan(tool_outputs)
        if self.enable_safety_validation:
            recommendations = _apply_safety_validation(recommendations, tool_outputs)
        if not recommendations:
            recommendations = [
                RecommendationItem(
                    action="monitor",
                    priority="medium",
                    reason="No high-risk patient or operational bottleneck was detected.",
                )
            ]
        follow_up_plan = self.follow_up_agent.create_plan(recommendations, system_state) if self.enable_follow_up_tracking else []
        agent_trace = _build_agent_trace(
            tool_outputs,
            recommendations,
            follow_up_plan,
            planning_mode="component_contribution_reference",
        )
        if self.enable_follow_up_tracking:
            agent_trace.append(self.follow_up_agent.build_trace(follow_up_plan))
        return EDDecisionResponse(
            summary=(
                f"Component contribution runner evaluated ED state as {system_state} using the same four "
                "decision-support tools while toggling safety, state, and follow-up modules."
            ),
            action_brief=_build_action_brief(recommendations),
            system_state=system_state,
            active_patient_count=len(ed_input.patients),
            pending_follow_up_count=len(follow_up_plan),
            recommendations=recommendations,
            tool_outputs=tool_outputs,
            agent_trace=agent_trace,
            follow_up_plan=follow_up_plan,
        )

    def _initial_plan(self, tool_outputs: ToolOutputs) -> list[RecommendationItem]:
        recommendations: list[RecommendationItem] = []
        top_patient = tool_outputs.patient_risk.flagged_patients[0] if tool_outputs.patient_risk.flagged_patients else None

        if top_patient and top_patient.risk_level in {"high", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent" if top_patient.risk_level == "critical" else "high",
                    target_id=None,
                    reason=(
                        f"Patient-risk tool identified a {top_patient.risk_level}-risk patient; "
                        "unvalidated initial plan did not enforce an explicit patient target."
                    ),
                )
            )
            if tool_outputs.bed_management.available_beds_now > 0:
                recommendations.append(
                    RecommendationItem(
                        action="admit_support",
                        priority="high",
                        target_id=top_patient.patient_id if self.enable_safety_validation else None,
                        reason="High-risk patient requires admission or high-acuity placement review.",
                    )
                )

        if tool_outputs.flow_prediction.bottleneck_level in {"high", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="reprioritize_queue",
                    priority="urgent" if tool_outputs.flow_prediction.bottleneck_level == "critical" else "high",
                    reason=(
                        f"Flow tool reports {tool_outputs.flow_prediction.bottleneck_level} pressure "
                        f"with predicted wait of {tool_outputs.flow_prediction.predicted_wait_minutes} minutes."
                    ),
                )
            )
        if tool_outputs.staffing.staffing_level in {"strained", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="staffing_alert",
                    priority="urgent" if tool_outputs.staffing.staffing_level == "critical" else "high",
                    reason=(
                        f"Staffing tool reports {tool_outputs.staffing.staffing_level} pressure "
                        f"with score {tool_outputs.staffing.staffing_pressure_score}."
                    ),
                )
            )
        if tool_outputs.bed_management.action_window in {"tight", "critical"}:
            recommendations.append(
                RecommendationItem(
                    action="reassign_bed",
                    priority="urgent" if tool_outputs.bed_management.action_window == "critical" else "high",
                    reason=(
                        f"Bed tool reports {tool_outputs.bed_management.action_window} capacity "
                        f"with {tool_outputs.bed_management.available_beds_now} immediately available beds."
                    ),
                )
            )
        return recommendations


def evaluate_component_contribution(use_real_agentic: bool = False) -> dict[str, Any]:
    """Runs the 180-scenario component contribution benchmark.

    The benchmark, silver-standard labels, and underlying tools stay unchanged.
    Only safety validation, state management, and follow-up tracking are toggled.
    """

    if use_real_agentic and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("use_real_agentic=True requires OPENAI_API_KEY.")

    scenarios = build_evaluation_scenarios()
    results: list[dict[str, Any]] = []
    scenario_results: dict[str, list[dict[str, Any]]] = {}
    for config in COMPONENT_CONTRIBUTION_CONFIGS:
        runner: DecisionRunner
        if use_real_agentic and config.use_llm_orchestration:
            runner = EDOrchestrationAgent(
                use_llm_summary=False,
                use_llm_input=False,
                enable_safety_validation=config.enable_safety_validation,
                enable_follow_up_tracking=config.enable_follow_up_tracking,
                enable_dynamic_tool_selection=config.enable_dynamic_tool_selection,
            )
        else:
            runner = ComponentReferenceRunner(
                enable_safety_validation=config.enable_safety_validation,
                enable_follow_up_tracking=config.enable_follow_up_tracking,
            )
        summary, rows = _evaluate_component_configuration(config, runner, scenarios)
        results.append(summary)
        scenario_results[config.name] = rows

    return {
        "mode": "real_agentic_llm" if use_real_agentic else "offline_component_reference",
        "scenario_count": len(scenarios),
        "configurations": [config.__dict__ for config in COMPONENT_CONTRIBUTION_CONFIGS],
        "summary": results,
        "scenario_results": scenario_results,
    }


def evaluate_stateful_replanning(
    runner: DecisionRunner,
    sequence: list[EDRequest | EDStateUpdate] | None = None,
) -> dict[str, Any]:
    """Evaluates whether sequential ED updates preserve state and trigger re-planning."""

    manager = EDStateManager()
    steps: list[dict[str, Any]] = []

    selected_sequence = sequence or _stateful_sequence()
    for index, payload in enumerate(selected_sequence, start=1):
        if isinstance(payload, EDRequest):
            active_state = manager.update_from_full_snapshot(payload)
        else:
            active_state = manager.apply_update(payload)

        response = runner.decide(active_state)
        pending_follow_up = manager.merge_follow_up_plan(response.follow_up_plan)
        critical_targets = {
            flag.patient_id
            for flag in response.tool_outputs.patient_risk.flagged_patients
            if flag.risk_level in {"high", "critical"}
        }
        escalated_targets = {
            item.target_id
            for item in response.recommendations
            if item.action == "escalate_patient" and item.target_id
        }
        actions = {item.action for item in response.recommendations}

        steps.append(
            {
                "step": index,
                "active_patient_count": len(active_state.patients),
                "critical_targets": sorted(critical_targets),
                "escalated_targets": sorted(escalated_targets),
                "actions": sorted(actions),
                "pending_follow_up_count": len(pending_follow_up),
                "generated_follow_up_tasks": [item.model_dump() for item in response.follow_up_plan],
                "critical_targets_covered": critical_targets.issubset(escalated_targets),
            }
        )

    active_counts = [step["active_patient_count"] for step in steps]
    action_sets = [tuple(step["actions"]) for step in steps]
    critical_target_total = sum(len(step["critical_targets"]) for step in steps)
    critical_target_hits = sum(
        len(set(step["critical_targets"]) & set(step["escalated_targets"]))
        for step in steps
    )
    generated_tasks = [
        task
        for step in steps
        for task in step["generated_follow_up_tasks"]
    ]
    task_ids = [task["task_id"] for task in generated_tasks]

    return {
        "steps": steps,
        "memory_retention_pass": active_counts == sorted(active_counts) and len(active_counts) == len(selected_sequence),
        "replanning_pass": len(set(action_sets)) > 1 and "ED-003" in steps[-1]["critical_targets"],
        "critical_target_coverage": round(critical_target_hits / max(critical_target_total, 1), 3),
        "final_pending_follow_up_count": steps[-1]["pending_follow_up_count"],
        "generated_follow_up_task_count": len(generated_tasks),
        "follow_up_owner_assignment_rate": round(
            sum(1 for task in generated_tasks if task["owner"]) / max(len(generated_tasks), 1), 3
        ),
        "follow_up_unique_task_id_rate": round(len(set(task_ids)) / max(len(task_ids), 1), 3),
        "follow_up_valid_status_rate": round(
            sum(1 for task in generated_tasks if task["status"] in {"pending", "accepted", "completed", "dismissed", "escalated"})
            / max(len(generated_tasks), 1),
            3,
        ),
    }


def evaluate_stateful_trajectory_suite(
    runner_factory: Any,
) -> dict[str, Any]:
    """Runs multiple prespecified operational trajectories with a fresh runner each time."""

    trajectories = _stateful_trajectories()
    results = {
        name: evaluate_stateful_replanning(runner_factory(), sequence)
        for name, sequence in trajectories.items()
    }
    return {
        "trajectory_count": len(results),
        "trajectories": results,
        "memory_retention_rate": round(
            sum(1 for result in results.values() if result["memory_retention_pass"]) / max(len(results), 1), 3
        ),
        "replanning_rate": round(
            sum(1 for result in results.values() if result["replanning_pass"]) / max(len(results), 1), 3
        ),
        "mean_critical_target_coverage": round(
            sum(result["critical_target_coverage"] for result in results.values()) / max(len(results), 1), 3
        ),
    }


def _evaluate_component_configuration(
    config: ComponentContributionConfig,
    runner: DecisionRunner,
    scenarios: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = []
    rows: list[dict[str, Any]] = []
    follow_up_counts: list[int] = []
    for scenario in scenarios:
        payload = _payload_for_component_configuration(scenario.payload, config.enable_state_management)
        start = time.perf_counter()
        try:
            response = runner.decide(payload)
            response_time_ms = (time.perf_counter() - start) * 1000.0
            score = score_response(scenario, response, response_time_ms=response_time_ms)
            status = "completed"
            error_type = None
            error_message = None
            follow_up_count = response.pending_follow_up_count
            tool_selection = response.tool_selection.model_dump() if response.tool_selection else None
            safety_log = [event.model_dump() for event in response.safety_validation_log]
        except Exception as exc:
            response_time_ms = (time.perf_counter() - start) * 1000.0
            score = _failed_scenario_score(scenario, response_time_ms)
            status = "failed"
            error_type = type(exc).__name__
            error_message = str(exc)
            follow_up_count = 0
            last_selection = getattr(runner, "_last_tool_selection", None)
            last_safety_log = getattr(runner, "_last_safety_log", [])
            tool_selection = last_selection.model_dump() if last_selection else None
            safety_log = [event.model_dump() for event in last_safety_log]
        scores.append(score)
        follow_up_counts.append(follow_up_count)
        rows.append(
            {
                "scenario": scenario.name,
                "status": status,
                "error_type": error_type,
                "error_message": error_message,
                "expected_actions": sorted(score.expected_actions),
                "predicted_actions": sorted(score.predicted_actions),
                "true_positives": score.true_positives,
                "false_positives": score.false_positives,
                "false_negatives": score.false_negatives,
                "action_quality": score.action_quality,
                "weighted_error_cost": score.weighted_error_cost,
                "escalation_hit": score.escalation_hit,
                "escalation_target_hit": score.escalation_target_hit,
                "escalation_target_exact_match": score.escalation_target_exact_match,
                "target_precision": score.target_precision,
                "target_recall": score.target_recall,
                "follow_up_count": follow_up_count,
                "tool_selection": tool_selection,
                "safety_validation_log": safety_log,
            }
        )

    metrics = summarize_scores(scores)
    metrics["avg_alert_burden"] = round(sum(follow_up_counts) / max(len(follow_up_counts), 1), 3)
    return {
        "configuration": config.label,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "action_quality": metrics["avg_action_quality"],
        "weighted_error_cost": metrics["avg_weighted_error_cost"],
        "escalation_recall": metrics["escalation_recall"],
        "escalation_target_accuracy": metrics["escalation_target_accuracy"],
        "target_precision": metrics["target_precision"],
        "target_recall": metrics["target_recall"],
        "target_exact_set_accuracy": metrics["target_exact_set_accuracy"],
        "failure_count": sum(1 for row in rows if row["status"] == "failed"),
        "alert_burden": metrics["avg_alert_burden"],
    }, rows


def _payload_for_component_configuration(ed_input: EDRequest, enable_state_management: bool) -> EDRequest:
    """Keeps the current snapshot identical in static component comparisons.

    State retention must be evaluated on matched sequential trajectories. Removing
    patients from a static snapshot changes observable current context and is not a
    valid memory ablation.
    """

    del enable_state_management
    return ed_input.model_copy(deep=True)


def _apply_safety_validation(
    recommendations: list[RecommendationItem],
    tool_outputs: ToolOutputs,
) -> list[RecommendationItem]:
    required_flags = [
        flag
        for flag in tool_outputs.patient_risk.flagged_patients
        if flag.risk_level in {"high", "critical"}
    ]
    if not required_flags:
        return recommendations

    validated = [
        item
        for item in recommendations
        if not (item.action == "escalate_patient" and item.target_id is None)
    ]
    existing_targets = {
        item.target_id
        for item in validated
        if item.action == "escalate_patient" and item.target_id
    }
    for flag in required_flags:
        if flag.patient_id not in existing_targets:
            validated.append(
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent" if flag.risk_level == "critical" else "high",
                    target_id=flag.patient_id,
                    reason=(
                        "Safety validation required explicit escalation for "
                        f"{flag.patient_id} because the patient-risk tool marked the patient as {flag.risk_level} risk."
                    ),
                )
            )
    return validated


def evaluate_safety_validation() -> dict[str, Any]:
    """Tests whether safety validation catches missing high/critical escalations."""

    scenarios = build_evaluation_scenarios()
    required_scenarios = 0
    required_patient_flags = 0
    missing_plan_detected = 0
    valid_plan_passed = 0

    for scenario in scenarios:
        tool_outputs = _tool_outputs_for_request(scenario.payload)
        required_flags = [
            flag
            for flag in tool_outputs.patient_risk.flagged_patients
            if flag.risk_level in {"high", "critical"}
        ]
        if not required_flags:
            continue

        required_scenarios += 1
        required_patient_flags += len(required_flags)
        missing_feedback = _required_escalation_feedback(
            [RecommendationItem(action="monitor", priority="medium", reason="No escalation returned.")],
            tool_outputs,
        )
        if missing_feedback:
            missing_plan_detected += 1

        valid_recommendations = [
            RecommendationItem(
                action="escalate_patient",
                priority="urgent" if flag.risk_level == "critical" else "high",
                target_id=flag.patient_id,
                reason=f"Safety validation recommendation for {flag.patient_id}.",
            )
            for flag in required_flags
        ]
        if not _required_escalation_feedback(valid_recommendations, tool_outputs):
            valid_plan_passed += 1

    return {
        "scenarios": len(scenarios),
        "scenarios_requiring_escalation": required_scenarios,
        "required_patient_flags": required_patient_flags,
        "missing_plan_detection_rate": round(missing_plan_detected / max(required_scenarios, 1), 3),
        "valid_plan_pass_rate": round(valid_plan_passed / max(required_scenarios, 1), 3),
    }


def summarize_agentic_scores_from_totals(totals: dict[str, float]) -> dict[str, float]:
    """Summarizes manually collected chunk totals from the live LLM evaluator."""

    count = int(totals["count"])
    tp = int(totals["tp"])
    fp = int(totals["fp"])
    fn = int(totals["fn"])
    escalation_cases = int(totals["escalation_cases"])
    return {
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "avg_action_quality": round(float(totals["action_quality_sum"]) / max(count, 1), 3),
        "avg_alert_burden": round(float(totals["alert_count_sum"]) / max(count, 1), 3),
        "avg_weighted_error_cost": round(float(totals["weighted_error_cost_sum"]) / max(count, 1), 3),
        "avg_structural_explanation_completeness": round(
            float(totals["structural_explanation_completeness_sum"]) / max(count, 1), 3
        ),
        "avg_response_time_ms": round(float(totals["response_time_ms_sum"]) / max(count, 1), 3),
        "escalation_recall": round(float(totals["escalation_hits"]) / max(escalation_cases, 1), 3),
        "target_precision": round(float(totals["target_tp"]) / max(float(totals["target_tp"]) + float(totals["target_fp"]), 1), 3),
        "target_recall": round(float(totals["target_tp"]) / max(float(totals["target_tp"]) + float(totals["target_fn"]), 1), 3),
        "target_exact_set_accuracy": round(float(totals["escalation_target_exact_matches"]) / max(escalation_cases, 1), 3),
        "failure_rate": round(float(totals.get("failures", 0)) / max(count, 1), 3),
    }


def _tool_outputs_for_request(ed_input: EDRequest) -> ToolOutputs:
    return ToolOutputs(
        flow_prediction=run_flow_prediction(ed_input),
        patient_risk=run_patient_risk(ed_input),
        staffing=run_staffing_availability(ed_input),
        bed_management=run_bed_management(ed_input),
    )


def _stateful_sequence() -> list[EDRequest | EDStateUpdate]:
    initial = EDRequest(
        timestamp="2026-06-17T09:00:00Z",
        current_queue_length=18,
        arrivals_last_hour=21,
        average_wait_minutes=74,
        boarding_patients=6,
        patients=[
            {
                "patient_id": "ED-001",
                "age": 78,
                "sex": "unknown",
                "triage_level": 2,
                "chief_complaint": "Shortness of breath and fever",
                "triage_notes": "Hypotensive, hypoxic, suspected sepsis.",
                "heart_rate": 126,
                "systolic_bp": 86,
                "respiratory_rate": 28,
                "oxygen_saturation": 90,
                "temperature_c": 38.8,
                "has_abnormal_labs": True,
                "suspected_sepsis": True,
                "pain_score": 7,
                "waiting_minutes": 45,
            }
        ],
        staffing={
            "available_nurses": 4,
            "available_physicians": 2,
            "nurse_capacity_per_hour": 4,
            "physician_capacity_per_hour": 6,
            "staff_absence_flag": True,
        },
        beds={
            "total_beds": 24,
            "occupied_beds": 23,
            "discharge_ready_beds": 2,
            "high_acuity_beds_available": 1,
        },
    )
    second = EDStateUpdate(
        timestamp="2026-06-17T09:08:00Z",
        current_queue_length=21,
        arrivals_last_hour=24,
        average_wait_minutes=82,
        boarding_patients=7,
        patients=[
            {
                "patient_id": "ED-002",
                "age": 44,
                "sex": "unknown",
                "triage_level": 3,
                "chief_complaint": "Abdominal pain",
                "triage_notes": "Long wait, stable vitals.",
                "heart_rate": 98,
                "systolic_bp": 118,
                "respiratory_rate": 18,
                "oxygen_saturation": 96,
                "temperature_c": 37.2,
                "has_abnormal_labs": False,
                "suspected_sepsis": False,
                "pain_score": 5,
                "waiting_minutes": 132,
            }
        ],
    )
    third = EDStateUpdate(
        timestamp="2026-06-17T09:15:00Z",
        current_queue_length=27,
        arrivals_last_hour=31,
        average_wait_minutes=103,
        boarding_patients=9,
        patients=[
            {
                "patient_id": "ED-003",
                "age": 69,
                "sex": "unknown",
                "triage_level": 2,
                "chief_complaint": "Chest pain, dyspnea, fever",
                "triage_notes": "New arrival, unstable vitals, abnormal labs, possible sepsis.",
                "heart_rate": 132,
                "systolic_bp": 82,
                "respiratory_rate": 32,
                "oxygen_saturation": 88,
                "temperature_c": 39.1,
                "has_abnormal_labs": True,
                "suspected_sepsis": True,
                "pain_score": 9,
                "waiting_minutes": 6,
            }
        ],
        staffing={
            "available_nurses": 3,
            "available_physicians": 1,
            "nurse_capacity_per_hour": 4,
            "physician_capacity_per_hour": 6,
            "staff_absence_flag": True,
        },
        beds={
            "total_beds": 24,
            "occupied_beds": 24,
            "discharge_ready_beds": 1,
            "high_acuity_beds_available": 0,
        },
    )
    return [initial, second, third]


def _stateful_trajectories() -> dict[str, list[EDRequest | EDStateUpdate]]:
    surge = _stateful_sequence()
    initial, second, third = surge
    staffing_recovery = [
        initial.model_copy(deep=True),
        EDStateUpdate(
            **{
                **second.model_dump(),
                "current_queue_length": 15,
                "arrivals_last_hour": 14,
                "average_wait_minutes": 55,
                "staffing": {
                    "available_nurses": 8,
                    "available_physicians": 4,
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": False,
                },
            }
        ),
        EDStateUpdate(
            **{
                **third.model_dump(),
                "current_queue_length": 17,
                "arrivals_last_hour": 16,
                "average_wait_minutes": 62,
                "staffing": {
                    "available_nurses": 7,
                    "available_physicians": 3,
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": False,
                },
            }
        ),
    ]
    bed_recovery = [
        initial.model_copy(deep=True),
        EDStateUpdate(
            **{
                **second.model_dump(),
                "beds": {
                    "total_beds": 24,
                    "occupied_beds": 19,
                    "discharge_ready_beds": 4,
                    "high_acuity_beds_available": 2,
                },
            }
        ),
        EDStateUpdate(
            **{
                **third.model_dump(),
                "beds": {
                    "total_beds": 24,
                    "occupied_beds": 21,
                    "discharge_ready_beds": 2,
                    "high_acuity_beds_available": 1,
                },
            }
        ),
    ]
    return {
        "arrival_and_capacity_surge": surge,
        "staffing_recovery_with_new_risk": staffing_recovery,
        "bed_recovery_with_new_risk": bed_recovery,
    }
