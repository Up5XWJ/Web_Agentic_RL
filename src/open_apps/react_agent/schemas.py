"""Typed, serializable state exchanged by the ReAct memory agent.

The model is deliberately not given the benchmark's structured ``subtasks``.
It must create ``Subgoal`` entries from the immutable natural-language goal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SubgoalStatus = Literal["pending", "in_progress", "verified", "blocked"]
AssessmentStatus = Literal["not_applicable", "success", "failed", "uncertain"]


class SchemaError(ValueError):
    """Raised when a model response does not satisfy the agent protocol."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{field_name} must be a JSON object, got {type(value).__name__}.")
    return value


def _string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field_name} must be a string.")
    value = value.strip()
    if not value and not allow_empty:
        raise SchemaError(f"{field_name} must not be empty.")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaError(f"{field_name} must be a list of strings.")
    return [item.strip() for item in value if item.strip()]


@dataclass
class Subgoal:
    id: str
    description: str
    status: SubgoalStatus = "pending"
    app: str | None = None
    evidence: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subgoal":
        data = _mapping(data, "subgoal")
        status = data.get("status", "pending")
        if status not in {"pending", "in_progress", "verified", "blocked"}:
            raise SchemaError(f"Invalid subgoal status {status!r}.")
        app = data.get("app")
        if app is not None:
            app = _string(app, "subgoal.app")
        return cls(
            id=_string(data.get("id"), "subgoal.id"),
            description=_string(data.get("description"), "subgoal.description"),
            status=status,
            app=app,
            evidence=_string(data.get("evidence", ""), "subgoal.evidence", allow_empty=True),
        )


@dataclass
class VerifiedFact:
    key: str
    value: str
    evidence: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedFact":
        data = _mapping(data, "verified_fact")
        return cls(
            key=_string(data.get("key"), "verified_fact.key"),
            value=_string(data.get("value"), "verified_fact.value", allow_empty=True),
            evidence=_string(data.get("evidence"), "verified_fact.evidence"),
        )


@dataclass
class FailureRecord:
    action: str
    reason: str
    retry_count: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureRecord":
        data = _mapping(data, "failure")
        retry_count = data.get("retry_count", 1)
        if not isinstance(retry_count, int) or retry_count < 1:
            raise SchemaError("failure.retry_count must be a positive integer.")
        return cls(
            action=_string(data.get("action"), "failure.action"),
            reason=_string(data.get("reason"), "failure.reason"),
            retry_count=retry_count,
        )


@dataclass
class TransientPageState:
    url: str = ""
    focused_bid: str = ""
    bids: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TransientPageState":
        if data is None:
            return cls()
        data = _mapping(data, "transient_page_state")
        bids = data.get("bids", {})
        if not isinstance(bids, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in bids.items()
        ):
            raise SchemaError("transient_page_state.bids must map semantic names to BID strings.")
        return cls(
            url=_string(data.get("url", ""), "transient_page_state.url", allow_empty=True),
            focused_bid=_string(
                data.get("focused_bid", ""),
                "transient_page_state.focused_bid",
                allow_empty=True,
            ),
            bids={key.strip(): value.strip() for key, value in bids.items() if key.strip() and value.strip()},
        )


@dataclass
class MemoryBlock:
    plan: list[Subgoal] = field(default_factory=list)
    current_subgoal_id: str | None = None
    verified_facts: list[VerifiedFact] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    transient_page_state: TransientPageState = field(default_factory=TransientPageState)

    @classmethod
    def empty(cls) -> "MemoryBlock":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryBlock":
        data = _mapping(data, "memory_next")
        plan_data = data.get("plan", [])
        facts_data = data.get("verified_facts", [])
        failures_data = data.get("failures", [])
        if not isinstance(plan_data, list):
            raise SchemaError("memory_next.plan must be a list.")
        if not isinstance(facts_data, list):
            raise SchemaError("memory_next.verified_facts must be a list.")
        if not isinstance(failures_data, list):
            raise SchemaError("memory_next.failures must be a list.")

        current = data.get("current_subgoal_id")
        if current is not None:
            current = _string(current, "memory_next.current_subgoal_id")

        return cls(
            plan=[Subgoal.from_dict(item) for item in plan_data],
            current_subgoal_id=current,
            verified_facts=[VerifiedFact.from_dict(item) for item in facts_data],
            pending=_string_list(data.get("pending", []), "memory_next.pending"),
            failures=[FailureRecord.from_dict(item) for item in failures_data],
            uncertainties=_string_list(
                data.get("uncertainties", []), "memory_next.uncertainties"
            ),
            transient_page_state=TransientPageState.from_dict(
                data.get("transient_page_state")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionAssessment:
    status: AssessmentStatus = "not_applicable"
    evidence: str = "No previous action to assess."
    page_changed: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ActionAssessment":
        if data is None:
            return cls()
        data = _mapping(data, "previous_action_assessment")
        status = data.get("status", "uncertain")
        if status not in {"not_applicable", "success", "failed", "uncertain"}:
            raise SchemaError(f"Invalid action assessment status {status!r}.")
        page_changed = data.get("page_changed", False)
        if not isinstance(page_changed, bool):
            raise SchemaError("previous_action_assessment.page_changed must be a boolean.")
        return cls(
            status=status,
            evidence=_string(
                data.get("evidence", ""),
                "previous_action_assessment.evidence",
                allow_empty=True,
            ),
            page_changed=page_changed,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReactDecision:
    observation: str
    current_subgoal: str
    reason: str
    expected_result: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReactDecision":
        data = _mapping(data, "decision")
        return cls(
            observation=_string(data.get("observation"), "decision.observation"),
            current_subgoal=_string(
                data.get("current_subgoal"), "decision.current_subgoal"
            ),
            reason=_string(data.get("reason"), "decision.reason"),
            expected_result=_string(
                data.get("expected_result"), "decision.expected_result"
            ),
        )


@dataclass
class ActionCall:
    name: str
    bid: str | None = None
    value: str | None = None
    options: str | list[str] | None = None
    key: str | None = None
    text: str | None = None
    coordinate: list[float] | None = None
    delta: list[float] | None = None
    button: str = "left"
    wait_ms: float = 1000.0
    reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionCall":
        data = _mapping(data, "action")
        coordinate = data.get("coordinate")
        delta = data.get("delta")
        for value, field_name in ((coordinate, "coordinate"), (delta, "delta")):
            if value is not None and not (
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(item, (int, float)) for item in value)
            ):
                raise SchemaError(f"action.{field_name} must be a two-number list.")

        options = data.get("options")
        if options is not None and not (
            isinstance(options, str)
            or (isinstance(options, list) and all(isinstance(item, str) for item in options))
        ):
            raise SchemaError("action.options must be a string or list of strings.")

        wait_ms = data.get("wait_ms", 1000.0)
        if not isinstance(wait_ms, (int, float)):
            raise SchemaError("action.wait_ms must be numeric.")

        def optional_string(name: str) -> str | None:
            value = data.get(name)
            if value is None:
                return None
            return _string(value, f"action.{name}", allow_empty=name in {"value", "text"})

        return cls(
            name=_string(data.get("name"), "action.name"),
            bid=optional_string("bid"),
            value=optional_string("value"),
            options=options,
            key=optional_string("key"),
            text=optional_string("text"),
            coordinate=coordinate,
            delta=delta,
            button=str(data.get("button", "left")),
            wait_ms=float(wait_ms),
            reason=optional_string("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass
class AgentResponse:
    memory_next: MemoryBlock
    previous_action_assessment: ActionAssessment
    decision: ReactDecision
    action: ActionCall

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResponse":
        data = _mapping(data, "agent response")
        return cls(
            memory_next=MemoryBlock.from_dict(data.get("memory_next")),
            previous_action_assessment=ActionAssessment.from_dict(
                data.get("previous_action_assessment")
            ),
            decision=ReactDecision.from_dict(data.get("decision")),
            action=ActionCall.from_dict(data.get("action")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
