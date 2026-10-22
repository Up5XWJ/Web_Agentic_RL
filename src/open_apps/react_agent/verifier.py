"""Deterministic feedback about the action executed on the previous turn."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any

from .schemas import ActionAssessment, ActionCall


def observation_signature(obs: dict[str, Any] | None) -> str:
    if not obs:
        return ""
    parts = (
        str(obs.get("url", "")),
        str(obs.get("pruned_html", obs.get("dom_txt", ""))),
        str(obs.get("axtree_txt", "")),
        str(obs.get("focused_element_bid", "")),
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8", errors="replace")).hexdigest()


class OutcomeVerifier:
    """Assess action-level effects without access to benchmark ground truth."""

    def assess(
        self,
        previous_obs: dict[str, Any] | None,
        current_obs: dict[str, Any],
        previous_action: ActionCall | None,
    ) -> ActionAssessment:
        if previous_action is None or previous_obs is None:
            return ActionAssessment()

        before = observation_signature(previous_obs)
        after = observation_signature(current_obs)
        changed = before != after
        error = str(current_obs.get("last_action_error", "") or "").strip()
        if error:
            return ActionAssessment(
                status="failed",
                evidence=f"BrowserGym reported an action error: {error[:500]}",
                page_changed=changed,
            )

        action = previous_action
        focused_bid = str(current_obs.get("focused_element_bid", "") or "")
        current_text = "\n".join(
            (
                str(current_obs.get("pruned_html", current_obs.get("dom_txt", ""))),
                str(current_obs.get("axtree_txt", "")),
            )
        )

        if action.name == "focus":
            success = focused_bid == action.bid
            return ActionAssessment(
                status="success" if success else "failed",
                evidence=(
                    f"Focused element is bid={focused_bid!r}."
                    if success
                    else f"Expected focus on bid={action.bid!r}, current focus is {focused_bid!r}."
                ),
                page_changed=changed,
            )

        if action.name == "fill":
            found = self._value_near_bid(current_text, action.bid or "", action.value or "")
            return ActionAssessment(
                status="success" if found else ("uncertain" if changed else "failed"),
                evidence=(
                    f"Current observation shows value {action.value!r} near bid={action.bid}."
                    if found
                    else f"Could not confirm value {action.value!r} near bid={action.bid}."
                ),
                page_changed=changed,
            )

        if action.name == "select_option":
            options = action.options if isinstance(action.options, list) else [action.options]
            options = [option for option in options if option is not None]
            found = any(self._value_near_bid(current_text, action.bid or "", option) for option in options)
            return ActionAssessment(
                status="success" if found else ("uncertain" if changed else "failed"),
                evidence=(
                    f"Current observation reflects selected option near bid={action.bid}."
                    if found
                    else f"Could not confirm selected option near bid={action.bid}."
                ),
                page_changed=changed,
            )

        if changed:
            return ActionAssessment(
                status="success",
                evidence="The URL, page text, element state, or focus changed after the action.",
                page_changed=True,
            )

        return ActionAssessment(
            status="uncertain",
            evidence="BrowserGym reported no error, but the textual page state did not change.",
            page_changed=False,
        )

    @staticmethod
    def _value_near_bid(text: str, bid: str, value: str) -> bool:
        if not bid:
            return False
        markers = (f'bid="{bid}"', f"bid='{bid}'", f"[{bid}]")
        lowered = html.unescape(text).casefold()
        target = html.unescape(str(value)).casefold()
        for marker in markers:
            start = lowered.find(marker.casefold())
            if start < 0:
                continue
            # HTML attributes occur after bid, while AXTree values occur on the
            # same line. A local window avoids matching an unrelated field.
            window = lowered[max(0, start - 120) : start + 700]
            if target and target in window:
                return True
            # Handle common escaped/whitespace-normalized value attributes.
            normalized_window = re.sub(r"\s+", " ", window)
            normalized_target = re.sub(r"\s+", " ", target)
            if normalized_target and normalized_target in normalized_window:
                return True
            if not target and re.search(r"\bvalue\s*=\s*(['\"])\1", window):
                return True
        return False
