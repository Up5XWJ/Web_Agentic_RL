"""Compact spatial observations that help the model decide whether to scroll."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from browsergym.core.constants import BROWSERGYM_ID_ATTRIBUTE as BID_ATTR

from .schemas import ActionAssessment, ActionCall


INTERACTIVE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "listbox",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
    "treeitem",
}


def _viewport_size(obs: dict[str, Any]) -> tuple[int, int]:
    screenshot = obs.get("screenshot")
    if screenshot is not None and hasattr(screenshot, "shape"):
        return int(screenshot.shape[1]), int(screenshot.shape[0])
    return 1000, 1000


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else round(value, 2)


def extract_scroll_state(obs: dict[str, Any]) -> dict[str, Any]:
    """Derive main-document scroll facts from the CDP DOMSnapshot observation."""
    viewport_width, viewport_height = _viewport_size(obs)
    state: dict[str, Any] = {
        "available": False,
        "source": "main_document_dom_snapshot",
        "scroll_x": 0,
        "scroll_y": 0,
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "document_width": viewport_width,
        "document_height": viewport_height,
        "max_scroll_x": 0,
        "max_scroll_y": 0,
        "can_scroll_left": False,
        "can_scroll_right": False,
        "can_scroll_up": False,
        "can_scroll_down": False,
        "at_top": True,
        "at_bottom": True,
    }

    dom_object = obs.get("dom_object")
    if not isinstance(dom_object, dict):
        return state
    documents = dom_object.get("documents")
    if not isinstance(documents, list) or not documents:
        return state
    document = documents[0]
    if not isinstance(document, dict):
        return state

    def number(name: str, fallback: float) -> float:
        value = document.get(name, fallback)
        return float(value) if isinstance(value, (int, float)) else float(fallback)

    scroll_x = max(0.0, number("scrollOffsetX", 0))
    scroll_y = max(0.0, number("scrollOffsetY", 0))
    document_width = max(float(viewport_width), number("contentWidth", viewport_width))
    document_height = max(float(viewport_height), number("contentHeight", viewport_height))
    max_scroll_x = max(0.0, document_width - viewport_width)
    max_scroll_y = max(0.0, document_height - viewport_height)
    tolerance = 2.0

    state.update(
        {
            "available": True,
            "scroll_x": _clean_number(scroll_x),
            "scroll_y": _clean_number(scroll_y),
            "document_width": _clean_number(document_width),
            "document_height": _clean_number(document_height),
            "max_scroll_x": _clean_number(max_scroll_x),
            "max_scroll_y": _clean_number(max_scroll_y),
            "can_scroll_left": scroll_x > tolerance,
            "can_scroll_right": scroll_x < max_scroll_x - tolerance,
            "can_scroll_up": scroll_y > tolerance,
            "can_scroll_down": scroll_y < max_scroll_y - tolerance,
            "at_top": scroll_y <= tolerance,
            "at_bottom": scroll_y >= max_scroll_y - tolerance,
        }
    )
    return state


def _decode_string(strings: list[Any], index: Any) -> str:
    if isinstance(index, int) and 0 <= index < len(strings):
        value = strings[index]
        return value if isinstance(value, str) else str(value)
    return ""


def _dom_element_metadata(obs: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Recover compact names and attributes for BID elements from DOMSnapshot."""
    dom_object = obs.get("dom_object")
    if not isinstance(dom_object, dict):
        return {}
    strings = dom_object.get("strings")
    documents = dom_object.get("documents")
    if not isinstance(strings, list) or not isinstance(documents, list):
        return {}

    result: dict[str, dict[str, str]] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        nodes = document.get("nodes")
        if not isinstance(nodes, dict):
            continue
        attributes = nodes.get("attributes", [])
        node_names = nodes.get("nodeName", [])
        node_types = nodes.get("nodeType", [])
        node_values = nodes.get("nodeValue", [])
        parents = nodes.get("parentIndex", [])
        if not all(isinstance(value, list) for value in (attributes, node_names, node_types)):
            continue

        children: dict[int, list[int]] = {}
        if isinstance(parents, list):
            for child_index, parent_index in enumerate(parents):
                if isinstance(parent_index, int) and parent_index >= 0:
                    children.setdefault(parent_index, []).append(child_index)

        text_cache: dict[int, str] = {}

        def subtree_text(node_index: int, budget: int = 20) -> str:
            if node_index in text_cache:
                return text_cache[node_index]
            stack = [node_index]
            chunks: list[str] = []
            visited = 0
            while stack and visited < budget:
                current = stack.pop()
                visited += 1
                node_type = node_types[current] if current < len(node_types) else None
                if node_type == 3 and current < len(node_values):
                    text = _decode_string(strings, node_values[current]).strip()
                    if text:
                        chunks.append(text)
                stack.extend(reversed(children.get(current, [])))
            text_cache[node_index] = " ".join(" ".join(chunks).split())[:160]
            return text_cache[node_index]

        for node_index, raw_attributes in enumerate(attributes):
            if not isinstance(raw_attributes, list):
                continue
            attrs = {
                _decode_string(strings, raw_attributes[index]): _decode_string(
                    strings, raw_attributes[index + 1]
                )
                for index in range(0, len(raw_attributes) - 1, 2)
            }
            bid = attrs.get(BID_ATTR)
            if not bid:
                continue
            tag = (
                _decode_string(strings, node_names[node_index]).lower()
                if node_index < len(node_names)
                else ""
            )
            own_text = subtree_text(node_index)
            parent_index = parents[node_index] if node_index < len(parents) else -1
            context = subtree_text(parent_index) if isinstance(parent_index, int) and parent_index >= 0 else ""
            result[str(bid)] = {
                "tag": tag,
                "type": attrs.get("type", ""),
                "name": attrs.get("aria-label", "")
                or attrs.get("title", "")
                or attrs.get("placeholder", "")
                or own_text
                or attrs.get("name", "")
                or attrs.get("id", "")
                or context,
                "context": context,
                "href": attrs.get("href", ""),
                "hx_target": attrs.get("hx-target", ""),
            }
    return result


def _axtree_element_metadata(obs: dict[str, Any]) -> dict[str, dict[str, str]]:
    axtree = obs.get("axtree_object")
    if not isinstance(axtree, dict) or not isinstance(axtree.get("nodes"), list):
        return {}
    result: dict[str, dict[str, str]] = {}
    for node in axtree["nodes"]:
        if not isinstance(node, dict) or node.get("ignored"):
            continue
        bid = node.get("browsergym_id")
        if bid is None:
            continue
        role = node.get("role", {})
        name = node.get("name", {})
        result[str(bid)] = {
            "role": str(role.get("value", "")) if isinstance(role, dict) else "",
            "name": str(name.get("value", "")) if isinstance(name, dict) else "",
        }
    return result


def _role_from_dom(metadata: dict[str, str]) -> str:
    tag = metadata.get("tag", "")
    input_type = metadata.get("type", "").lower()
    if tag == "a":
        return "link"
    if tag == "button":
        return "button"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag == "input":
        if input_type in {"checkbox", "radio", "button", "submit"}:
            return {"submit": "button"}.get(input_type, input_type)
        return "textbox"
    return "clickable"


def extract_offscreen_interactive_elements(
    obs: dict[str, Any],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return a bounded spatial index of interactive elements above/below the viewport."""
    if limit <= 0:
        return []
    _, viewport_height = _viewport_size(obs)
    extra_properties = obs.get("extra_element_properties")
    if not isinstance(extra_properties, dict):
        return []

    dom_metadata = _dom_element_metadata(obs)
    ax_metadata = _axtree_element_metadata(obs)
    candidates: list[dict[str, Any]] = []
    for raw_bid, properties in extra_properties.items():
        if not isinstance(properties, dict):
            continue
        visibility = properties.get("visibility")
        bbox = properties.get("bbox")
        if not isinstance(visibility, (int, float)) or visibility >= 0.5:
            continue
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        x, y, width, height = (float(value) for value in bbox)
        if y + height <= 0:
            direction = "above"
            distance = -(y + height)
        elif y >= viewport_height:
            direction = "below"
            distance = y - viewport_height
        else:
            # The element intersects the viewport but less than 50% is visible. It is
            # already represented visually and is not a truly offscreen candidate.
            continue

        bid = str(raw_bid)
        dom = dom_metadata.get(bid, {})
        ax = ax_metadata.get(bid, {})
        role = ax.get("role") or _role_from_dom(dom)
        if role not in INTERACTIVE_ROLES and not properties.get("clickable"):
            continue
        name = (ax.get("name") or dom.get("name") or dom.get("context") or "").strip()
        item: dict[str, Any] = {
            "bid": bid,
            "role": role,
            "name": " ".join(name.split())[:120],
            "direction": direction,
            "distance_px": _clean_number(max(0.0, distance)),
        }
        if dom.get("href"):
            item["href"] = dom["href"][:160]
        if dom.get("hx_target"):
            item["hx_target"] = dom["hx_target"][:120]
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            float(item["distance_px"]),
            0 if item["role"] in {"link", "button"} else 1,
            item["name"],
        )
    )
    return candidates[:limit]


def _merge_ranges(ranges: list[list[float]], new_range: list[float]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted([*ranges, new_range]):
        if not merged or start > merged[-1][1] + 2:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged[-6:]


@dataclass
class PageExplorationTracker:
    """Deterministic spatial memory; it informs but never chooses an action."""

    url: str = ""
    top_visited: bool = False
    bottom_visited: bool = False
    min_scroll_y_seen: float | None = None
    max_scroll_y_seen: float | None = None
    max_scroll_y_available: float = 0
    visited_scroll_ranges: list[list[float]] = field(default_factory=list)
    last_scroll_direction: str | None = None
    last_scroll_changed_position: bool | None = None
    no_progress_turns: int = 0

    def reset(self) -> None:
        self.__dict__.update(PageExplorationTracker().__dict__)

    def update(
        self,
        obs: dict[str, Any],
        *,
        previous_obs: dict[str, Any] | None = None,
        previous_action: ActionCall | None = None,
        assessment: ActionAssessment | None = None,
    ) -> dict[str, Any]:
        scroll_state = extract_scroll_state(obs)
        url = str(obs.get("url", "") or "")
        if url != self.url:
            self.reset()
            self.url = url

        scroll_y = float(scroll_state["scroll_y"])
        viewport_height = float(scroll_state["viewport_height"])
        document_height = float(scroll_state["document_height"])
        max_scroll_y = float(scroll_state["max_scroll_y"])
        if max_scroll_y > self.max_scroll_y_available + 2:
            self.bottom_visited = False
        self.max_scroll_y_available = max_scroll_y
        self.min_scroll_y_seen = (
            scroll_y if self.min_scroll_y_seen is None else min(self.min_scroll_y_seen, scroll_y)
        )
        self.max_scroll_y_seen = (
            scroll_y if self.max_scroll_y_seen is None else max(self.max_scroll_y_seen, scroll_y)
        )
        self.top_visited = self.top_visited or bool(scroll_state["at_top"])
        self.bottom_visited = self.bottom_visited or bool(scroll_state["at_bottom"])
        self.visited_scroll_ranges = _merge_ranges(
            self.visited_scroll_ranges,
            [scroll_y, min(document_height, scroll_y + viewport_height)],
        )

        if previous_action is not None:
            made_progress = bool(assessment and assessment.page_changed)
            if previous_action.name == "scroll":
                delta_y = float((previous_action.delta or [0, 0])[1])
                self.last_scroll_direction = "down" if delta_y > 0 else "up" if delta_y < 0 else None
                previous_scroll_y = float(extract_scroll_state(previous_obs or {})["scroll_y"])
                self.last_scroll_changed_position = abs(previous_scroll_y - scroll_y) > 2
                made_progress = made_progress or self.last_scroll_changed_position
            self.no_progress_turns = 0 if made_progress else self.no_progress_turns + 1

        unexplored_directions: list[str] = []
        if scroll_state["can_scroll_up"] and not self.top_visited:
            unexplored_directions.append("up")
        if scroll_state["can_scroll_down"] and not self.bottom_visited:
            unexplored_directions.append("down")

        return {
            "url": self.url,
            "top_visited": self.top_visited,
            "bottom_visited": self.bottom_visited,
            "min_scroll_y_seen": _clean_number(float(self.min_scroll_y_seen or 0)),
            "max_scroll_y_seen": _clean_number(float(self.max_scroll_y_seen or 0)),
            "visited_scroll_ranges": [
                [_clean_number(float(start)), _clean_number(float(end))]
                for start, end in self.visited_scroll_ranges
            ],
            "unexplored_directions": unexplored_directions,
            "last_scroll_direction": self.last_scroll_direction,
            "last_scroll_changed_position": self.last_scroll_changed_position,
            "no_progress_turns": self.no_progress_turns,
        }
