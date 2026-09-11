from __future__ import annotations

from typing import Any

from app.models import EDRequest, ToolOutputs


AVAILABLE_TOOLS = {
    "patient_risk": "Assess every active patient for high or critical escalation risk. Mandatory when patients are present.",
    "flow_prediction": "Assess queue, arrivals, waiting time, boarding, and congestion pressure.",
    "staffing": "Assess workload relative to nurse and physician capacity.",
    "bed_management": "Assess occupancy, immediate capacity, discharge-ready beds, and high-acuity placement constraints.",
}


ALLOWED_ACTIONS = [
    "escalate_patient",
    "reprioritize_queue",
    "reassign_bed",
    "admit_support",
    "discharge_support",
    "staffing_alert",
    "monitor",
]

ALLOWED_PRIORITIES = ["low", "medium", "high", "urgent"]


def build_tool_selection_prompt(ed_input: EDRequest) -> dict[str, Any]:
    """Prompt for selecting only the tools needed for the current snapshot."""

    return {
        "agent": "ED Tool Selection Controller",
        "task": (
            "Select the smallest safe subset of decision-support tools required for this ED snapshot. "
            "patient_risk is mandatory whenever patients are present. Select additional tools only when their "
            "evidence can materially affect an operational recommendation."
        ),
        "available_tools": AVAILABLE_TOOLS,
        "rules": [
            "Always select patient_risk when the snapshot contains one or more patients.",
            "Select flow_prediction for material queue, arrival, waiting-time, or boarding questions.",
            "Select staffing when staffing capacity or absence may constrain operations.",
            "Select bed_management when occupancy or placement capacity may constrain care.",
            "Return JSON only and do not generate recommendations in this step.",
        ],
        "input_snapshot": ed_input.model_dump(),
        "required_output": {
            "selected_tools": ["tool names from available_tools"],
            "rationale": ["one concise reason per selected tool"],
        },
    }


def build_input_understanding_prompt(raw_text: str) -> dict[str, Any]:
    """Prompt for converting messy ED text into the EDRequest schema."""

    return {
        "agent": "Input Understanding Agent",
        "role": (
            "You convert emergency department free-text notes into a structured ED snapshot "
            "for downstream decision-support tools."
        ),
        "safety_rules": [
            "Do not diagnose, treat, or make clinical recommendations.",
            "Only extract or conservatively normalize information from the input text.",
            "If a required field is missing, use the provided conservative default and list the field in missing_fields.",
            "Return JSON only. Do not use markdown.",
        ],
        "required_output": {
            "ed_request": {
                "timestamp": "ISO-8601 timestamp. Use current_or_unknown if absent.",
                "current_queue_length": "integer >= 0; default 8 if absent",
                "arrivals_last_hour": "integer >= 0; default 8 if absent",
                "average_wait_minutes": "integer >= 0; default 35 if absent",
                "boarding_patients": "integer >= 0; default 1 if absent",
                "patients": [
                    {
                        "patient_id": "string; default TEXT-001",
                        "age": "integer 0-120; default 55 if absent",
                        "sex": "female, male, other, or unknown",
                        "triage_level": "integer 1-5; default 3 if absent",
                        "chief_complaint": "short text",
                        "triage_notes": "short text copied/summarized from source",
                        "heart_rate": "integer 20-250; default 92 if absent",
                        "systolic_bp": "integer 40-300; default 118 if absent",
                        "respiratory_rate": "integer 5-80; default 18 if absent",
                        "oxygen_saturation": "integer 50-100; default 96 if absent",
                        "temperature_c": "float 30-45; default 37.0 if absent",
                        "has_abnormal_labs": "boolean",
                        "suspected_sepsis": "boolean",
                        "pain_score": "integer 0-10; default 0 if absent",
                        "waiting_minutes": "integer >= 0; default 30 if absent",
                    }
                ],
                "staffing": {
                    "available_nurses": "integer >= 0; default 5 if absent",
                    "available_physicians": "integer >= 0; default 2 if absent",
                    "nurse_capacity_per_hour": 4,
                    "physician_capacity_per_hour": 6,
                    "staff_absence_flag": "boolean",
                },
                "beds": {
                    "total_beds": "integer >= 1; default 24 if absent",
                    "occupied_beds": "integer >= 0; default 18 if absent",
                    "discharge_ready_beds": "integer >= 0; default 2 if absent",
                    "high_acuity_beds_available": "integer >= 0; default 1 if absent",
                },
            },
            "confidence": "float between 0 and 1",
            "notes": ["short normalization note"],
            "missing_fields": ["field names defaulted because missing"],
        },
        "raw_text": raw_text,
    }


def build_orchestration_prompt(
    ed_input: EDRequest,
    tool_outputs: ToolOutputs,
    validation_feedback: list[str] | None = None,
) -> dict[str, Any]:
    """Prompt for selecting ED operational recommendations from tool outputs."""

    return {
        "agent": "ED Orchestration Agent",
        "role": (
            "You are a human-supervised emergency department operations decision-support agent. "
            "You are the central controller: the user gives input to you, and you coordinate the input-normalization agent, "
            "ED state manager, decision tools, and follow-up agent. "
            "Your job is to coordinate the ED workflow using a Level 4 agentic loop: "
            "Goal -> Plan -> Execute -> Monitor outcomes -> Re-plan if conditions change -> Continue until goal achieved."
        ),
        "agentic_cycle": {
            "goal": [
                "Keep patients safe.",
                "Reduce waiting-room and queue risk.",
                "Coordinate crowding, staffing, and bed/capacity constraints.",
                "Escalate urgent cases under human oversight.",
            ],
            "plan": [
                "Decide which agents/tools are needed for the current ED state.",
                "Anticipate conflicts, such as high patient risk with limited beds or high crowding with low staffing.",
                "Decide what success means for this snapshot: resolved, escalated, admitted, discharged, or safely monitored.",
            ],
            "execute": [
                "Use the ED State Manager to reason over all active patients and unresolved follow-up tasks.",
                "Use the Input Understanding Agent output as the structured ED snapshot.",
                "Use only the tools marked executed=true; executed=false means the tool was not selected and produced no evidence.",
                "Treat Patient Risk Tool output as mandatory safety evidence whenever active patients are present.",
            ],
            "monitor_outcomes": [
                "Check patient risk, flow pressure, staffing pressure, bed capacity, active patients, and follow-up needs together.",
                "Identify unresolved risk, delayed action, missing ownership, or operational blockage.",
                "Decide which recommendation needs follow-up tracking.",
            ],
            "replan_if_conditions_change": [
                "If risk increases, escalate patient priority.",
                "If beds become unavailable, re-plan bed/capacity actions.",
                "If staffing worsens, re-plan staffing and queue actions.",
                "If follow-up is delayed or dismissed, escalate to the next human owner.",
            ],
            "continue_until_goal_achieved": [
                "Continue the loop until the situation is resolved, escalated, admitted, discharged, or safely monitored.",
                "Do not claim completion unless the available information supports it.",
                "Return a follow-up need for any unresolved goal.",
            ],
            "output": [
                "Return prioritized recommendations only from the allowed action list.",
                "Return a short reasoning summary that explicitly mentions Goal, Plan, Execute, Monitor, Re-plan, and Continue.",
                "Return JSON only.",
            ],
        },
        "safety_rules": [
            "Do not provide autonomous medical orders.",
            "Do not invent facts beyond the input snapshot and tool outputs.",
            "Recommendations are decision support for clinicians and operations leaders.",
            "Prefer clear, prioritized, actionable workflow recommendations.",
            "Return JSON only. Do not use markdown.",
        ],
        "allowed_actions": ALLOWED_ACTIONS,
        "allowed_priorities": ALLOWED_PRIORITIES,
        "priority_guidance": {
            "urgent": "unstable patient, critical operational bottleneck, or immediate escalation needed",
            "high": "important action needed soon, but not immediate life-threatening escalation",
            "medium": "watch/monitor or routine operational follow-up",
            "low": "minor optimization only",
        },
        "coordination_rules": [
            "For every patient whose risk_level is high or critical in patient_risk.flagged_patients, include a separate escalate_patient recommendation with that patient_id as target_id.",
            "Do not downgrade a high-risk or critical-risk patient to monitor-only.",
            "If high-risk patient placement is feasible, include admit_support.",
            "If the flow tool was executed and flow/crowding is high or critical, include reprioritize_queue.",
            "If the staffing tool was executed and staffing is strained or critical, include staffing_alert.",
            "If the bed tool was executed and beds are tight or critical and capacity blocks care, include reassign_bed.",
            "If the bed tool was executed, occupancy is at least 85%, and discharge-ready beds are available, include discharge_support.",
            "If no action is needed, return one monitor recommendation.",
            "Avoid duplicate actions. Each recommendation should explain which tool evidence caused it.",
        ],
        "validation_feedback": validation_feedback or [],
        "input_snapshot": ed_input.model_dump(),
        "tool_outputs": tool_outputs.model_dump(),
        "required_output": {
            "reasoning_summary": "brief explanation using Goal -> Plan -> Execute -> Monitor -> Re-plan -> Continue wording",
            "goal": "short statement of the ED operational goal for this case",
            "plan": ["1-3 short planning steps"],
            "execute": ["which agents/tools were used and why"],
            "monitor_outcomes": ["key outcomes or unresolved risks to monitor"],
            "replan_if_conditions_change": ["what should change if ED conditions worsen or follow-up is delayed"],
            "continue_until_goal_achieved": "short completion/follow-up condition",
            "recommendations": [
                {
                    "action": "one allowed action",
                    "priority": "one allowed priority",
                    "target_id": "patient id if patient-specific, otherwise null",
                    "reason": "one concise sentence citing tool evidence",
                }
            ],
        },
    }


def build_summary_prompt(
    ed_input: EDRequest,
    tool_outputs: ToolOutputs,
    recommendations: list[dict[str, Any]],
    system_state: str,
) -> dict[str, Any]:
    """Prompt for optional narrative summary only."""

    return {
        "agent": "Narrative Summary Agent",
        "system_state": system_state,
        "input_snapshot": ed_input.model_dump(),
        "tool_outputs": tool_outputs.model_dump(),
        "recommendations": recommendations,
        "task": "Write a concise ED decision-support summary. Do not invent facts.",
    }
