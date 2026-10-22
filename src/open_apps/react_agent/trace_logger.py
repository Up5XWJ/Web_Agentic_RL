"""Human-readable per-turn traces for debugging long-horizon failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .schemas import ActionAssessment, AgentResponse, MemoryBlock


class TraceLogger:
    def __init__(self, save_dir: str | Path | None):
        self.root = Path(save_dir) / "react_agent_trace" if save_dir else None

    def log_turn(
        self,
        *,
        turn: int,
        obs: dict[str, Any],
        memory_before: MemoryBlock,
        automatic_assessment: ActionAssessment,
        prompt_text: str,
        raw_response: str,
        parsed_response: AgentResponse | None,
        browsergym_action: str | None,
        spatial_state: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self.root is None:
            return
        turn_dir = self.root / f"turn_{turn:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(turn_dir / "memory_before.json", memory_before.to_dict())
        self._write_json(turn_dir / "automatic_assessment.json", automatic_assessment.to_dict())
        if spatial_state is not None:
            self._write_json(turn_dir / "spatial_state.json", spatial_state)
        (turn_dir / "html.txt").write_text(
            str(obs.get("pruned_html", obs.get("dom_txt", ""))), encoding="utf-8"
        )
        (turn_dir / "axtree.txt").write_text(str(obs.get("axtree_txt", "")), encoding="utf-8")
        (turn_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        (turn_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
        self._write_json(
            turn_dir / "result.json",
            {
                "parsed_response": parsed_response.to_dict() if parsed_response else None,
                "browsergym_action": browsergym_action,
                "error": error,
            },
        )
        screenshot = obs.get("screenshot")
        if isinstance(screenshot, np.ndarray):
            Image.fromarray(screenshot).save(turn_dir / "screenshot.png")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
