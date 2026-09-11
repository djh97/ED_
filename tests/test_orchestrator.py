from __future__ import annotations

import json
import os
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.additional_evaluations import _payload_for_component_configuration, evaluate_safety_validation, evaluate_stateful_replanning
from app.agentic_system import (
    EDOrchestrationAgent,
    _build_agentic_reasoning,
    _oracle_urgent_patients,
    _required_escalation_feedback,
    _run_selected_tools,
    score_response,
    evaluate_runner,
)
from app.models import EDRequest, EDStateUpdate, FollowUpItem, RecommendationItem, ToolOutputs
from app.free_text_evaluation import compare_requests
from app.experiment import (
    ExperimentConfig,
    _atomic_write_json,
    _failure_csv,
    _only_checkpoint_writer_changed,
    run_matched_experiment,
)
from app.nhamcs_ed import _build_request, _labels, _temperature_c
from app.orchestrator import EDOrchestrationAgent, RuleBasedEDBaseline, build_evaluation_scenarios, evaluate_all_systems
from app.state_manager import EDStateManager
from app.tools.bed_management import run_bed_management
from app.tools.flow_prediction import run_flow_prediction
from app.tools.patient_risk import run_patient_risk
from app.tools.staffing import run_staffing_availability


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.output_text = json.dumps(payload)


class _FakeResponses:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def create(self, **_: object) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(self.payloads.pop(0))


class _FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.responses = _FakeResponses(payloads)


def _plan(recommendations: list[dict]) -> dict:
    return {
        "reasoning_summary": "Goal -> Plan -> Execute -> Monitor -> Re-plan -> Continue.",
        "goal": "Maintain ED safety.",
        "plan": ["Use selected evidence."],
        "execute": ["Execute selected tools."],
        "monitor_outcomes": ["Monitor unresolved risk."],
        "replan_if_conditions_change": ["Re-plan if validation fails."],
        "continue_until_goal_achieved": "Stop after a validated recommendation or explicit failure.",
        "recommendations": recommendations,
    }


class OrchestratorTests(unittest.TestCase):
    def test_dynamic_decision_executes_only_selected_tools(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)
        client = _FakeClient([
            _plan([
                {
                    "action": "escalate_patient",
                    "priority": "urgent",
                    "target_id": "ED-001",
                    "reason": "Patient-risk tool identified ED-001 as critical and requiring escalation.",
                }
            ])
        ])
        agent = EDOrchestrationAgent(
            use_llm_summary=False,
            use_llm_input=False,
            planner_client=client,
            tool_selector=lambda _: ["patient_risk", "staffing"],
        )

        response = agent.decide(request)

        self.assertTrue(response.tool_outputs.patient_risk.executed)
        self.assertTrue(response.tool_outputs.staffing.executed)
        self.assertFalse(response.tool_outputs.flow_prediction.executed)
        self.assertFalse(response.tool_outputs.bed_management.executed)
        self.assertEqual(response.tool_selection.selected_tools, ["patient_risk", "staffing"])

    def test_safety_replanning_is_bounded_and_logged(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)
        invalid = _plan([
            {"action": "monitor", "priority": "medium", "target_id": None, "reason": "Continue monitoring available evidence."}
        ])
        client = _FakeClient([invalid, invalid])
        agent = EDOrchestrationAgent(
            use_llm_summary=False,
            use_llm_input=False,
            planner_client=client,
            tool_selector=lambda _: ["patient_risk"],
            max_replanning_attempts=1,
        )

        with self.assertRaises(RuntimeError):
            agent.decide(request)

        self.assertEqual(client.responses.calls, 2)
        self.assertEqual(len(agent._last_safety_log), 2)
        self.assertFalse(agent._last_safety_log[-1].passed)

    def test_evaluation_records_runner_failure_without_aborting(self) -> None:
        class FailingRunner:
            def decide(self, _: EDRequest):
                raise ValueError("controlled failure")

        scenarios = build_evaluation_scenarios(count=2)
        result = evaluate_runner("failing", FailingRunner(), scenarios=scenarios)

        self.assertEqual(result["summary_metrics"]["failure_count"], 2)
        self.assertEqual(result["summary_metrics"]["failure_rate"], 1.0)
        self.assertTrue(all(row["status"] == "failed" for row in result["scenario_results"]))

    def test_clustered_bootstrap_is_reproducible(self) -> None:
        scenarios = build_evaluation_scenarios(count=3)
        config = ExperimentConfig(runs=2, bootstrap_samples=25, bootstrap_seed=77)
        factories = {
            "full_dynamic": lambda _: RuleBasedEDBaseline(),
            "all_tools": lambda _: RuleBasedEDBaseline(),
        }

        first = run_matched_experiment(scenarios, factories, config)
        second = run_matched_experiment(scenarios, factories, config)

        for result in (first, second):
            for metrics in result["bootstrap_confidence_intervals"].values():
                metrics.pop("avg_response_time_ms", None)
            for metrics in result["paired_bootstrap_differences"].values():
                metrics.pop("avg_response_time_ms", None)
        self.assertEqual(first["bootstrap_confidence_intervals"], second["bootstrap_confidence_intervals"])
        self.assertEqual(first["paired_bootstrap_differences"], second["paired_bootstrap_differences"])

    def test_checkpoint_resume_does_not_repeat_completed_scenarios(self) -> None:
        scenarios = build_evaluation_scenarios(count=4)
        config = ExperimentConfig(runs=1, bootstrap_samples=5, concurrency=1)
        interrupted_calls = {"count": 0}

        class InterruptingRunner:
            def decide(self, request: EDRequest):
                if interrupted_calls["count"] == 2:
                    raise KeyboardInterrupt()
                interrupted_calls["count"] += 1
                return RuleBasedEDBaseline().decide(request)

        with TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            with self.assertRaises(KeyboardInterrupt):
                run_matched_experiment(
                    scenarios,
                    {"full_dynamic": lambda _: InterruptingRunner()},
                    config,
                    checkpoint_dir=checkpoint_dir,
                )

            index = json.loads((checkpoint_dir / "checkpoint_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["completed_count"], 2)

            resumed_calls = {"count": 0}

            class CountingRunner:
                def decide(self, request: EDRequest):
                    resumed_calls["count"] += 1
                    return RuleBasedEDBaseline().decide(request)

            result = run_matched_experiment(
                scenarios,
                {"full_dynamic": lambda _: CountingRunner()},
                config,
                checkpoint_dir=checkpoint_dir,
                resume=True,
            )

            self.assertEqual(resumed_calls["count"], 2)
            self.assertEqual(len(result["runs"]["full_dynamic"][0]["scenario_results"]), 4)
            final_index = json.loads((checkpoint_dir / "checkpoint_index.json").read_text(encoding="utf-8"))
            self.assertEqual(final_index["completed_count"], 4)
            self.assertEqual(final_index["status"], "scenario_evaluation_complete")

            incompatible = ExperimentConfig(runs=2, bootstrap_samples=5, concurrency=1)
            with self.assertRaises(ValueError):
                run_matched_experiment(
                    scenarios,
                    {"full_dynamic": lambda _: CountingRunner()},
                    incompatible,
                    checkpoint_dir=checkpoint_dir,
                    resume=True,
                )

    def test_checkpoint_writer_upgrade_is_narrowly_scoped(self) -> None:
        previous = {
            "code_commit": "old",
            "source_sha256": "old-hash",
            "models": {"orchestration": "gpt-5-nano"},
            "scenario_sha256": "scenario-hash",
        }
        writer_only = {**previous, "code_commit": "new", "source_sha256": "new-hash"}
        changed_model = {**writer_only, "models": {"orchestration": "different-model"}}

        self.assertTrue(_only_checkpoint_writer_changed(previous, writer_only))
        self.assertFalse(_only_checkpoint_writer_changed(previous, changed_model))

    def test_atomic_json_write_retries_windows_permission_race(self) -> None:
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "checkpoint.json"
            real_replace = os.replace
            attempts = {"count": 0}

            def flaky_replace(source: str | Path, destination: str | Path) -> None:
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise PermissionError("simulated Windows file lock")
                real_replace(source, destination)

            with patch("app.experiment.os.replace", side_effect=flaky_replace):
                _atomic_write_json(target, {"saved": True})

            self.assertEqual(attempts["count"], 3)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"saved": True})

    def test_failure_csv_projects_full_scenario_rows(self) -> None:
        row = {
            "configuration": "full_dynamic",
            "run_index": 0,
            "scenario": "scenario-001",
            "status": "failed",
            "error_type": "RuntimeError",
            "error_message": "controlled failure",
            "metrics": {"failure_rate": 1.0},
            "recommendations": [],
        }

        output = _failure_csv([row])

        self.assertIn("full_dynamic,0,scenario-001,failed,RuntimeError,controlled failure", output)
        self.assertNotIn("metrics", output.splitlines()[0])

    def test_dynamic_selector_adds_mandatory_patient_risk(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)
        agent = EDOrchestrationAgent(
            use_llm_summary=False,
            use_llm_planning=False,
            use_llm_input=False,
            tool_selector=lambda _: ["staffing"],
        )

        decision = agent._select_tools(request)

        self.assertEqual(decision.selected_tools, ["patient_risk", "staffing"])
        self.assertIn("flow_prediction", decision.skipped_tools)
        self.assertIn("bed_management", decision.skipped_tools)

    def test_selected_tool_execution_skips_unselected_tools(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)

        flow, risk, staffing, bed = _run_selected_tools(request, {"patient_risk", "staffing"})

        self.assertFalse(flow.executed)
        self.assertTrue(risk.executed)
        self.assertTrue(staffing.executed)
        self.assertFalse(bed.executed)

    def test_mixed_scenario_oracle_and_scoring_cover_all_urgent_targets(self) -> None:
        scenario = next(item for item in build_evaluation_scenarios() if len(item.urgent_targets) > 1)
        response = RuleBasedEDBaseline().decide(scenario.payload)

        score = score_response(scenario, response)

        self.assertGreater(len(score.expected_targets), 1)
        self.assertEqual(score.target_true_positives + score.target_false_negatives, len(score.expected_targets))
        self.assertEqual(score.escalation_target_hit, score.target_false_negatives == 0)

    def test_benchmark_emits_discharge_support_when_rule_is_satisfied(self) -> None:
        scenarios = build_evaluation_scenarios()

        self.assertTrue(any("discharge_support" in scenario.expected_actions for scenario in scenarios))

    def test_final_benchmark_action_counts_match_figure_3(self) -> None:
        scenarios = build_evaluation_scenarios(count=180, seed=2026)
        counts = Counter(
            action
            for scenario in scenarios
            for action in scenario.expected_actions
        )

        self.assertEqual(
            counts,
            Counter(
                {
                    "escalate_patient": 132,
                    "admit_support": 121,
                    "reprioritize_queue": 90,
                    "staffing_alert": 90,
                    "reassign_bed": 59,
                    "discharge_support": 59,
                    "monitor": 10,
                }
            ),
        )

    def test_oracle_and_validator_targets_agree_on_final_benchmark(self) -> None:
        scenarios = build_evaluation_scenarios(count=180, seed=2026)

        for scenario in scenarios:
            oracle_targets = _oracle_urgent_patients(scenario.payload)
            adapter = run_patient_risk(scenario.payload)
            validator_targets = {
                flag.patient_id
                for flag in adapter.flagged_patients
                if flag.risk_level in {"high", "critical"}
            }
            self.assertEqual(oracle_targets, validator_targets, scenario.name)

    def test_state_ablation_preserves_complete_current_snapshot(self) -> None:
        scenario = build_evaluation_scenarios()[0]

        with_state = _payload_for_component_configuration(scenario.payload, True)
        without_state = _payload_for_component_configuration(scenario.payload, False)

        self.assertEqual(with_state.model_dump(), without_state.model_dump())
        self.assertEqual(len(without_state.patients), len(scenario.payload.patients))

    def test_free_text_field_comparison_reports_changed_field(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        expected = EDRequest(**payload)
        predicted = expected.model_copy(update={"current_queue_length": expected.current_queue_length + 1})

        results = compare_requests(expected, predicted)
        queue = next(item for item in results if item["field"] == "current_queue_length")

        self.assertFalse(queue["correct"])

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is required for real LLM orchestration tests.")
    def test_high_risk_patient_triggers_escalation(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)

        result = EDOrchestrationAgent(use_llm_summary=False).decide(request)

        actions = {item.action for item in result.recommendations}
        self.assertIn("escalate_patient", actions)
        self.assertIn("staffing_alert", actions)
        self.assertIn("reassign_bed", actions)
        self.assertEqual(result.system_state, "critical")
        self.assertGreaterEqual(len(result.follow_up_plan), len(result.recommendations))
        self.assertIn("input_understanding", {item.agent for item in result.agent_trace})
        self.assertIn("follow_up_tracking", {item.agent for item in result.agent_trace})
        orchestration_trace = [item for item in result.agent_trace if item.agent == "orchestration"][0]
        self.assertIn("planning_mode=llm_prompted", orchestration_trace.evidence)
        self.assertIn("LLM plan", {item.step for item in result.agent_trace})
        self.assertIn("LLM monitor outcomes", {item.step for item in result.agent_trace})

    def test_evaluation_includes_all_comparison_systems(self) -> None:
        comparison = evaluate_all_systems(agentic_runner=RuleBasedEDBaseline())

        self.assertEqual(
            set(comparison),
            {
                "esi_triage_baseline",
                "news2_qsofa_baseline",
                "nedocs_edwin_crowding_baseline",
                "prediction_only_baseline",
                "rule_based_baseline",
                "non_agentic_integrated_baseline",
                "agentic_orchestration",
            },
        )
        agentic_metrics = comparison["agentic_orchestration"]["summary_metrics"]
        rule_metrics = comparison["rule_based_baseline"]["summary_metrics"]
        prediction_metrics = comparison["prediction_only_baseline"]["summary_metrics"]

        self.assertGreaterEqual(agentic_metrics["avg_action_quality"], rule_metrics["avg_action_quality"])
        self.assertGreater(agentic_metrics["avg_action_quality"], prediction_metrics["avg_action_quality"])
        self.assertIn("avg_response_time_ms", agentic_metrics)
        self.assertIn("avg_structural_explanation_completeness", agentic_metrics)
        self.assertNotIn("avg_explanation_quality", agentic_metrics)
        self.assertNotIn("avg_recommendation_delay", agentic_metrics)

    def test_default_evaluation_uses_180_scenarios(self) -> None:
        self.assertEqual(len(build_evaluation_scenarios()), 180)

    def test_state_manager_keeps_previous_patient_when_new_patient_arrives(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        initial = EDRequest(**payload)
        manager = EDStateManager()
        manager.update_from_full_snapshot(initial)

        new_patient = {
            "patient_id": "ED-NEW",
            "age": 63,
            "triage_level": 2,
            "heart_rate": 118,
            "systolic_bp": 94,
            "respiratory_rate": 26,
            "oxygen_saturation": 91,
            "temperature_c": 38.1,
            "has_abnormal_labs": True,
            "suspected_sepsis": False,
            "pain_score": 8,
            "waiting_minutes": 5,
        }
        updated = manager.apply_update(
            EDStateUpdate(
                timestamp="2026-05-19T10:12:00Z",
                current_queue_length=19,
                patients=[new_patient],
            )
        )

        patient_ids = {patient.patient_id for patient in updated.patients}
        self.assertIn("ED-001", patient_ids)
        self.assertIn("ED-NEW", patient_ids)
        self.assertEqual(len(updated.patients), 4)

    def test_state_manager_removes_discharged_patient(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        manager = EDStateManager()
        manager.update_from_full_snapshot(EDRequest(**payload))

        updated = manager.apply_update(EDStateUpdate(discharged_patient_ids=["ED-002"]))
        patient_ids = {patient.patient_id for patient in updated.patients}

        self.assertNotIn("ED-002", patient_ids)
        self.assertEqual(len(updated.patients), 2)

    def test_state_manager_deduplicates_operational_followups(self) -> None:
        manager = EDStateManager()
        first = FollowUpItem(
            task_id="FU-001",
            linked_action="staffing_alert",
            owner="charge_nurse",
            due_minutes=10,
            escalation_rule="Escalate if not accepted.",
            reason="Staffing tool shows critical pressure.",
        )
        second = FollowUpItem(
            task_id="FU-001",
            linked_action="staffing_alert",
            owner="charge_nurse",
            due_minutes=10,
            escalation_rule="Escalate if not accepted.",
            reason="Staffing tool still shows critical pressure after another patient arrived.",
        )

        manager.merge_follow_up_plan([first])
        pending = manager.merge_follow_up_plan([second])

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].task_id, "SFU-001")
        self.assertIn("still shows critical pressure", pending[0].reason)

    def test_required_escalation_validator_requires_targeted_high_risk_patient(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)
        tool_outputs = ToolOutputs(
            flow_prediction=run_flow_prediction(request),
            patient_risk=run_patient_risk(request),
            staffing=run_staffing_availability(request),
            bed_management=run_bed_management(request),
        )

        missing = _required_escalation_feedback(
            [RecommendationItem(action="monitor", priority="medium", reason="Continue monitoring the ED state.")],
            tool_outputs,
        )
        satisfied = _required_escalation_feedback(
            [
                RecommendationItem(
                    action="escalate_patient",
                    priority="urgent",
                    target_id="ED-001",
                    reason="Patient risk tool flagged ED-001 as critical.",
                )
            ],
            tool_outputs,
        )

        self.assertTrue(any("ED-001" in item for item in missing))
        self.assertEqual(satisfied, [])

    def test_agentic_reasoning_formats_planning_trace(self) -> None:
        reasoning = _build_agentic_reasoning(
            {
                "reasoning_summary": "Goal -> Plan -> Execute -> Monitor -> Re-plan -> Continue.",
                "goal": "Reduce ED risk while maintaining throughput.",
                "plan": ["Use all active patients and tools."],
                "execute": ["Run risk, flow, staffing, and bed tools."],
                "monitor_outcomes": ["Watch unresolved escalations."],
                "replan_if_conditions_change": ["Escalate priority if staffing worsens."],
                "continue_until_goal_achieved": "Continue until escalated or safely monitored.",
            }
        )

        self.assertIsNotNone(reasoning)
        self.assertEqual(reasoning.goal, "Reduce ED risk while maintaining throughput.")
        self.assertIn("Run risk", reasoning.execute[0])

    def test_llm_trace_accepts_mapping_evidence(self) -> None:
        from app.agentic_system import _build_llm_cycle_trace

        trace = _build_llm_cycle_trace({"plan": {"first": "assess risk", "second": "allocate bed"}})

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].evidence, ["first: assess risk", "second: allocate bed"])

    def test_nhamcs_row_maps_to_ed_request_and_labels(self) -> None:
        row = {
            "year": "2022",
            "visit_month": "5",
            "day_of_week": "2",
            "arrival_time": "1430",
            "age": "77",
            "sex": "Female",
            "temp": "101.3",
            "heart_rate": "132",
            "resp_rate": "28",
            "sys_bp": "84",
            "spo2": "89",
            "pain_score": "7",
            "target_triage_acuity": "2",
            "wait_time_minutes": "95",
            "ems_arrival": "Yes",
            "seen_last_72h": "1",
            "chief_complaint_text": "fever. shortness of breath",
        }

        request = _build_request(row, "NHAMCS-TEST", {"2022-5-2-14": 6})
        labels = _labels(row)

        self.assertEqual(request.patients[0].patient_id, "NHAMCS-TEST")
        self.assertEqual(request.patients[0].triage_level, 2)
        self.assertEqual(request.patients[0].sex, "female")
        self.assertTrue(request.patients[0].suspected_sepsis)
        self.assertTrue(labels["high_acuity"])
        self.assertTrue(labels["critical_vitals"])
        self.assertTrue(labels["prolonged_wait"])
        self.assertTrue(labels["revisit_72h"])
        self.assertAlmostEqual(_temperature_c("101.3"), 38.5)

    def test_safety_validation_evaluation_catches_missing_escalations(self) -> None:
        metrics = evaluate_safety_validation()

        self.assertEqual(metrics["scenarios"], 180)
        self.assertGreater(metrics["scenarios_requiring_escalation"], 0)
        self.assertEqual(metrics["missing_plan_detection_rate"], 1.0)
        self.assertEqual(metrics["valid_plan_pass_rate"], 1.0)

    def test_stateful_replanning_evaluation_tracks_active_patients(self) -> None:
        metrics = evaluate_stateful_replanning(RuleBasedEDBaseline())

        self.assertTrue(metrics["memory_retention_pass"])
        self.assertTrue(metrics["replanning_pass"])
        self.assertEqual([step["active_patient_count"] for step in metrics["steps"]], [1, 2, 3])
        self.assertIn("ED-003", metrics["steps"][-1]["critical_targets"])

    def test_agentic_planner_requires_llm(self) -> None:
        payload = json.loads(Path("demo_data/sample_case.json").read_text(encoding="utf-8"))
        request = EDRequest(**payload)

        with self.assertRaises(RuntimeError):
            EDOrchestrationAgent(use_llm_summary=False, use_llm_planning=False, use_llm_input=False).decide(request)

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is required for real LLM free-text extraction tests.")
    def test_free_text_input_uses_llm_extractor(self) -> None:
        text = (
            "Queue 18, arrivals 22, average wait 95, boarding 6. "
            "Nurses 3, physicians 1, total beds 24, occupied beds 23. "
            "Patient age 77, triage 2, HR 128, SBP 86, RR 30, SpO2 89, "
            "temp 38.9, shortness of breath, fever, sepsis."
        )
        result = EDOrchestrationAgent(use_llm_summary=False).decide(text)
        actions = {item.action for item in result.recommendations}
        self.assertIn("escalate_patient", actions)
        self.assertIn("input_understanding", {item.agent for item in result.agent_trace})


if __name__ == "__main__":
    unittest.main()
