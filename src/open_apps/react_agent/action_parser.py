"""Strict JSON response parsing and BrowserGym action serialization."""

from __future__ import annotations

import json
import re
from typing import Any

from agentlab.llm.llm_utils import ParseError

from .schemas import ActionCall, AgentResponse, SchemaError


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class ReactResponseParser:
    def parse(self, response: str) -> AgentResponse:
        response = (response or "").strip()
        if not response:
            raise ParseError("Empty response. Return exactly one JSON object.")
        try:
            payload = self._extract_json(response)
            return AgentResponse.from_dict(payload)
        except (json.JSONDecodeError, SchemaError, TypeError, ValueError) as exc:
            raise ParseError(
                f"Invalid ReAct response: {exc}. Return one JSON object matching the required schema."
            ) from exc

    @staticmethod
    def _extract_json(response: str) -> dict[str, Any]:
        fenced = _FENCED_JSON_RE.search(response)
        candidate = fenced.group(1) if fenced else response
        start = candidate.find("{")
        if start < 0:
            raise json.JSONDecodeError("No JSON object found", candidate, 0)
        payload, end = json.JSONDecoder().raw_decode(candidate[start:])
        if not isinstance(payload, dict):
            raise SchemaError("Top-level response must be a JSON object.")
        trailing = candidate[start + end :].strip()
        if trailing and not trailing.startswith("```"):
            raise SchemaError("Do not output text after the JSON object.")
        return payload


class BrowserGymActionSerializer:
    """Convert a validated action into BrowserGym high-level syntax."""

    def __init__(self, coordinate_space: int = 1000):
        if coordinate_space <= 0:
            raise ValueError("coordinate_space must be positive.")
        self.coordinate_space = coordinate_space

    def serialize(self, action: ActionCall, viewport: tuple[int, int]) -> str:
        name = action.name
        if name == "click":
            return f"click({action.bid!r}, button={action.button!r})"
        if name == "fill":
            return f"fill({action.bid!r}, {action.value!r})"
        if name == "focus":
            return f"focus({action.bid!r})"
        if name == "select_option":
            return f"select_option({action.bid!r}, {action.options!r})"
        if name == "keyboard_press":
            return f"keyboard_press({action.key!r})"
        if name == "keyboard_type":
            return f"keyboard_type({action.text!r})"
        if name in {"mouse_click", "mouse_dblclick"}:
            x, y = self._rescale(action.coordinate, viewport)
            return f"{name}({x}, {y}, button={action.button!r})"
        if name == "scroll":
            dx, dy = self._rescale(action.delta, viewport)
            return f"scroll({dx}, {dy})"
        if name == "noop":
            return f"noop(wait_ms={action.wait_ms!r})"
        if name == "report_infeasible":
            return f"report_infeasible({action.reason!r})"
        raise ValueError(f"Unsupported action {name!r}.")

    def _rescale(
        self, values: list[float] | None, viewport: tuple[int, int]
    ) -> tuple[int, int]:
        if values is None:
            raise ValueError("Coordinate action is missing a two-number list.")
        width, height = viewport
        return (
            int(round(float(values[0]) * width / self.coordinate_space)),
            int(round(float(values[1]) * height / self.coordinate_space)),
        )
