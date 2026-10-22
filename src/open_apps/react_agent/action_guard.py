"""Pre-execution validation for model-authored actions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentlab.llm.llm_utils import ParseError

from .schemas import ActionCall


BID_ACTIONS = {"click", "fill", "focus", "select_option"}
COORDINATE_ACTIONS = {"mouse_click", "mouse_dblclick"}
SUPPORTED_ACTIONS = BID_ACTIONS | {
    "keyboard_press",
    "keyboard_type",
    "scroll",
    "noop",
    "report_infeasible",
} | COORDINATE_ACTIONS


@dataclass
class ActionGuardConfig:
    coordinate_space: int = 1000
    allow_unfocused_keyboard_type: bool = False
    max_wait_ms: float = 10_000
    allowed_actions: set[str] = field(default_factory=lambda: set(SUPPORTED_ACTIONS))


class ActionGuard:
    def __init__(self, config: ActionGuardConfig | None = None):
        self.config = config or ActionGuardConfig()

    def validate(self, action: ActionCall, obs: dict[str, Any]) -> None:
        if action.name not in self.config.allowed_actions:
            raise ParseError(
                f"Unsupported action {action.name!r}. Allowed: {sorted(self.config.allowed_actions)}."
            )

        valid_bids = extract_valid_bids(obs)
        if action.name in BID_ACTIONS:
            if not action.bid:
                raise ParseError(f"Action {action.name!r} requires a bid.")
            if action.bid not in valid_bids:
                raise ParseError(
                    f"BID {action.bid!r} is stale or absent from the current observation. "
                    "Choose a BID visible in the current HTML/AXTree."
                )
            props = obs.get("extra_element_properties", {}).get(action.bid, {})
            visibility = props.get("visibility")
            if isinstance(visibility, (int, float)) and visibility < 0.5:
                raise ParseError(f"BID {action.bid!r} is not currently visible.")

        if action.name == "fill" and action.value is None:
            raise ParseError("fill requires action.value, which may be an empty string.")
        if action.name == "select_option" and action.options is None:
            raise ParseError("select_option requires action.options.")
        if action.name == "keyboard_press" and not action.key:
            raise ParseError("keyboard_press requires action.key.")
        if action.name == "keyboard_type":
            if action.text is None:
                raise ParseError("keyboard_type requires action.text.")
            if (
                not self.config.allow_unfocused_keyboard_type
                and not str(obs.get("focused_element_bid", "") or "")
            ):
                raise ParseError(
                    "keyboard_type requires a focused element. Prefer fill(bid, value), "
                    "or focus/click the field first."
                )

        if action.name in COORDINATE_ACTIONS:
            self._validate_pair(action.coordinate, "coordinate", bounded=True)
        if action.name == "scroll":
            self._validate_pair(action.delta, "delta", bounded=False)
        if action.button not in {"left", "middle", "right"}:
            raise ParseError("action.button must be left, middle, or right.")
        if action.name == "noop" and not (0 <= action.wait_ms <= self.config.max_wait_ms):
            raise ParseError(
                f"noop.wait_ms must be between 0 and {self.config.max_wait_ms}."
            )
        if action.name == "report_infeasible" and not action.reason:
            raise ParseError("report_infeasible requires a concrete reason.")

    def _validate_pair(self, value, name: str, *, bounded: bool) -> None:
        if value is None:
            raise ParseError(f"action.{name} is required.")
        if bounded and any(not 0 <= item <= self.config.coordinate_space for item in value):
            raise ParseError(
                f"action.{name} must be inside 0..{self.config.coordinate_space}."
            )


def extract_valid_bids(obs: dict[str, Any]) -> set[str]:
    bids = {str(bid) for bid in obs.get("extra_element_properties", {})}
    html = str(obs.get("pruned_html", obs.get("dom_txt", "")))
    axtree = str(obs.get("axtree_txt", ""))
    bids.update(re.findall(r"\bbid\s*=\s*['\"]([A-Za-z0-9]+)['\"]", html))
    bids.update(re.findall(r"^\s*\[([A-Za-z0-9]+)\]", axtree, re.MULTILINE))
    return bids
