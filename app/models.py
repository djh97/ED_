from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    patient_id: str
    age: int = Field(ge=0, le=120)
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    triage_level: int = Field(ge=1, le=5, description="1 is highest acuity, 5 is lowest")
    chief_complaint: str = ""
    triage_notes: str = ""
    heart_rate: int = Field(ge=20, le=250)
    systolic_bp: int = Field(ge=40, le=300)
    respiratory_rate: int = Field(ge=5, le=80)
    oxygen_saturation: int = Field(ge=50, le=100)
    temperature_c: float = Field(ge=30.0, le=45.0)
    has_abnormal_labs: bool = False
    suspected_sepsis: bool = False
    pain_score: int = Field(default=0, ge=0, le=10)
    waiting_minutes: int = Field(default=0, ge=0)


class StaffingInput(BaseModel):
    available_nurses: int = Field(ge=0)
    available_physicians: int = Field(ge=0)
    nurse_capacity_per_hour: int = Field(default=4, ge=1)
    physician_capacity_per_hour: int = Field(default=6, ge=1)
    staff_absence_flag: bool = False


class BedInput(BaseModel):
    total_beds: int = Field(ge=1)
    occupied_beds: int = Field(ge=0)
    discharge_ready_beds: int = Field(default=0, ge=0)
    high_acuity_beds_available: int = Field(default=0, ge=0)


class EDRequest(BaseModel):
    timestamp: str
    current_queue_length: int = Field(ge=0)
    arrivals_last_hour: int = Field(ge=0)
    average_wait_minutes: int = Field(ge=0)
    boarding_patients: int = Field(ge=0)
    patients: list[PatientInput]
    staffing: StaffingInput
    beds: BedInput


class EDStateUpdate(BaseModel):
    timestamp: str | None = None
    current_queue_length: int | None = Field(default=None, ge=0)
    arrivals_last_hour: int | None = Field(default=None, ge=0)
    average_wait_minutes: int | None = Field(default=None, ge=0)
    boarding_patients: int | None = Field(default=None, ge=0)
    patients: list[PatientInput] = Field(default_factory=list)
    discharged_patient_ids: list[str] = Field(default_factory=list)
    completed_follow_up_task_ids: list[str] = Field(default_factory=list)
    staffing: StaffingInput | None = None
    beds: BedInput | None = None


class FlowToolResult(BaseModel):
    model_name: str = "Deterministic flow assessment"
    executed: bool = True
    congestion_score: float = Field(ge=0, le=1)
    predicted_wait_minutes: int = Field(ge=0)
    bottleneck_level: Literal["low", "moderate", "high", "critical"]
    recommended_adjustment: str | None = None
    reasoning: list[str] = Field(default_factory=list)
    rationale: str


class RiskFlag(BaseModel):
    patient_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: Literal["low", "moderate", "high", "critical"]
    escalation_needed: bool
    top_risk_factors: list[str] = Field(default_factory=list)
    rationale: str


class PatientRiskToolResult(BaseModel):
    model_name: str = "Deterministic patient-risk assessment"
    executed: bool = True
    highest_risk_patient_id: str | None = None
    flagged_patients: list[RiskFlag]
    recommended_adjustment: str | None = None
    rationale: str


class StaffingToolResult(BaseModel):
    model_name: str = "Deterministic staffing assessment"
    executed: bool = True
    staffing_pressure_score: float = Field(ge=0, le=1)
    staffing_level: Literal["adequate", "strained", "critical"]
    estimated_hourly_capacity: int = Field(ge=0)
    recommended_adjustment: str | None = None
    reasoning: list[str] = Field(default_factory=list)
    rationale: str


class BedToolResult(BaseModel):
    model_name: str = "Deterministic bed-capacity assessment"
    executed: bool = True
    occupancy_rate: float = Field(ge=0, le=1.5)
    available_beds_now: int
    action_window: Literal["open", "tight", "critical"]
    recommended_adjustment: str | None = None
    reasoning: list[str] = Field(default_factory=list)
    rationale: str


class RecommendationItem(BaseModel):
    action: Literal[
        "escalate_patient",
        "reprioritize_queue",
        "reassign_bed",
        "admit_support",
        "discharge_support",
        "staffing_alert",
        "monitor"
    ]
    priority: Literal["low", "medium", "high", "urgent"]
    target_id: str | None = None
    reason: str


class AgentTraceItem(BaseModel):
    agent: Literal[
        "input_understanding",
        "orchestration",
        "flow_tool",
        "patient_risk_tool",
        "staffing_tool",
        "bed_tool",
        "follow_up_tracking",
    ]
    step: str
    evidence: list[str] = Field(default_factory=list)


class FollowUpItem(BaseModel):
    task_id: str
    linked_action: Literal[
        "escalate_patient",
        "reprioritize_queue",
        "reassign_bed",
        "admit_support",
        "discharge_support",
        "staffing_alert",
        "monitor",
    ]
    owner: str
    due_minutes: int = Field(ge=0)
    status: Literal["pending", "accepted", "completed", "dismissed", "escalated"] = "pending"
    escalation_rule: str
    reason: str


class AgenticReasoning(BaseModel):
    reasoning_summary: str = ""
    goal: str = ""
    plan: list[str] = Field(default_factory=list)
    execute: list[str] = Field(default_factory=list)
    monitor_outcomes: list[str] = Field(default_factory=list)
    replan_if_conditions_change: list[str] = Field(default_factory=list)
    continue_until_goal_achieved: str = ""


class ToolSelectionDecision(BaseModel):
    selected_tools: list[Literal["patient_risk", "flow_prediction", "staffing", "bed_management"]]
    skipped_tools: list[Literal["patient_risk", "flow_prediction", "staffing", "bed_management"]] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    selection_rounds: int = Field(default=1, ge=1)
    fallback_used: bool = False
    error: str | None = None


class SafetyValidationEvent(BaseModel):
    attempt: int = Field(ge=0)
    passed: bool
    feedback: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)


class ToolOutputs(BaseModel):
    flow_prediction: FlowToolResult
    patient_risk: PatientRiskToolResult
    staffing: StaffingToolResult
    bed_management: BedToolResult


class EDDecisionResponse(BaseModel):
    summary: str
    action_brief: str
    system_state: Literal["stable", "watch", "strained", "critical"]
    active_patient_count: int = 0
    pending_follow_up_count: int = 0
    agentic_reasoning: AgenticReasoning | None = None
    tool_selection: ToolSelectionDecision | None = None
    safety_validation_log: list[SafetyValidationEvent] = Field(default_factory=list)
    recommendations: list[RecommendationItem]
    tool_outputs: ToolOutputs
    agent_trace: list[AgentTraceItem] = Field(default_factory=list)
    follow_up_plan: list[FollowUpItem] = Field(default_factory=list)


class EDStateSnapshot(BaseModel):
    active_state: EDRequest | None = None
    active_patient_count: int = 0
    pending_follow_up_plan: list[FollowUpItem] = Field(default_factory=list)


class RawEDTextRequest(BaseModel):
    text: str = Field(min_length=1)
