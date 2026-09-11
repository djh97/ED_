from __future__ import annotations

from datetime import datetime, timezone

from app.models import EDRequest, EDStateSnapshot, EDStateUpdate, FollowUpItem, PatientInput


class EDStateManager:
    """Maintains active ED context across sequential updates.

    This is an in-memory research state store. A production version should
    replace it with an EHR/event-stream backed store.
    """

    def __init__(self) -> None:
        self._active_state: EDRequest | None = None
        self._patients_by_id: dict[str, PatientInput] = {}
        self._pending_follow_up: list[FollowUpItem] = []

    def reset(self) -> EDStateSnapshot:
        self._active_state = None
        self._patients_by_id = {}
        self._pending_follow_up = []
        return self.snapshot()

    def snapshot(self) -> EDStateSnapshot:
        return EDStateSnapshot(
            active_state=self._active_state,
            active_patient_count=len(self._patients_by_id),
            pending_follow_up_plan=self._pending_follow_up,
        )

    def update_from_full_snapshot(self, ed_input: EDRequest) -> EDRequest:
        self._active_state = ed_input
        self._patients_by_id = {patient.patient_id: patient for patient in ed_input.patients}
        return ed_input

    def apply_update(self, update: EDStateUpdate) -> EDRequest:
        if self._active_state is None:
            if not update.patients:
                raise ValueError("First ED state update must include at least one patient or use /evaluate with a full EDRequest.")
            self._active_state = self._build_initial_state(update)

        for patient_id in update.discharged_patient_ids:
            self._patients_by_id.pop(patient_id, None)

        for patient in update.patients:
            self._patients_by_id[patient.patient_id] = patient

        if update.completed_follow_up_task_ids:
            completed = set(update.completed_follow_up_task_ids)
            self._pending_follow_up = [
                task.model_copy(update={"status": "completed"}) if task.task_id in completed else task
                for task in self._pending_follow_up
            ]

        current = self._active_state
        self._active_state = EDRequest(
            timestamp=update.timestamp or current.timestamp,
            current_queue_length=update.current_queue_length if update.current_queue_length is not None else current.current_queue_length,
            arrivals_last_hour=update.arrivals_last_hour if update.arrivals_last_hour is not None else current.arrivals_last_hour,
            average_wait_minutes=update.average_wait_minutes if update.average_wait_minutes is not None else current.average_wait_minutes,
            boarding_patients=update.boarding_patients if update.boarding_patients is not None else current.boarding_patients,
            patients=list(self._patients_by_id.values()),
            staffing=update.staffing or current.staffing,
            beds=update.beds or current.beds,
        )
        return self._active_state

    def merge_follow_up_plan(self, follow_up_plan: list[FollowUpItem]) -> list[FollowUpItem]:
        existing = {task.task_id: task for task in self._pending_follow_up if task.status not in {"completed", "dismissed"}}
        next_index = len(existing) + 1
        for task in follow_up_plan:
            key = self._follow_up_key(task)
            matched_id = None
            for existing_id, existing_task in existing.items():
                if self._follow_up_key(existing_task) == key:
                    matched_id = existing_id
                    break
            if matched_id:
                existing[matched_id] = task.model_copy(update={"task_id": matched_id})
            else:
                state_task_id = f"SFU-{next_index:03d}"
                existing[state_task_id] = task.model_copy(update={"task_id": state_task_id})
                next_index += 1
        self._pending_follow_up = list(existing.values())
        return self._pending_follow_up

    def _build_initial_state(self, update: EDStateUpdate) -> EDRequest:
        self._patients_by_id = {patient.patient_id: patient for patient in update.patients}
        return EDRequest(
            timestamp=update.timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            current_queue_length=update.current_queue_length or len(update.patients),
            arrivals_last_hour=update.arrivals_last_hour or len(update.patients),
            average_wait_minutes=update.average_wait_minutes or 30,
            boarding_patients=update.boarding_patients or 0,
            patients=list(self._patients_by_id.values()),
            staffing=update.staffing or {
                "available_nurses": 5,
                "available_physicians": 2,
                "nurse_capacity_per_hour": 4,
                "physician_capacity_per_hour": 6,
                "staff_absence_flag": False,
            },
            beds=update.beds or {
                "total_beds": 24,
                "occupied_beds": 18,
                "discharge_ready_beds": 2,
                "high_acuity_beds_available": 1,
            },
        )

    def _follow_up_key(self, task: FollowUpItem) -> tuple[str, str | None, str]:
        operational_actions = {
            "reprioritize_queue",
            "reassign_bed",
            "discharge_support",
            "staffing_alert",
        }
        if task.linked_action in operational_actions:
            return (task.linked_action, None, task.owner)
        return (task.linked_action, self._target_from_reason(task.reason), task.owner)

    def _target_from_reason(self, reason: str) -> str | None:
        for patient_id in self._patients_by_id:
            if patient_id in reason:
                return patient_id
        return None
