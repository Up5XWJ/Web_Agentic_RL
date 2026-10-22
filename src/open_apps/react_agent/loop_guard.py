"""Detect repeated ineffective actions without retaining full prompt history."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentlab.llm.llm_utils import ParseError

from .schemas import ActionAssessment, ActionCall


@dataclass
class OutcomeRecord:
    action_signature: str
    before_signature: str
    after_signature: str
    assessment: str


class LoopGuard:
    def __init__(self, repeat_limit: int = 2, history_limit: int = 6):
        if repeat_limit < 1:
            raise ValueError("repeat_limit must be at least 1.")
        self.repeat_limit = repeat_limit
        self.history_limit = max(history_limit, repeat_limit)
        self.records: list[OutcomeRecord] = []

    def reset(self) -> None:
        self.records.clear()

    def record(
        self,
        action: ActionCall,
        before_signature: str,
        after_signature: str,
        assessment: ActionAssessment,
    ) -> None:
        self.records.append(
            OutcomeRecord(
                action_signature=self.action_signature(action),
                before_signature=before_signature,
                after_signature=after_signature,
                assessment=assessment.status,
            )
        )
        self.records = self.records[-self.history_limit :]

    def validate_candidate(self, action: ActionCall) -> None:
        signature = self.action_signature(action)
        repeated = 0
        for record in reversed(self.records):
            ineffective = (
                record.before_signature == record.after_signature
                or record.assessment in {"failed", "uncertain"}
            )
            if record.action_signature != signature or not ineffective:
                break
            repeated += 1
        if repeated >= self.repeat_limit:
            raise ParseError(
                "Loop detected: this same action has already been ineffective "
                f"{repeated} consecutive times. Choose a different BID/action or re-plan."
            )

    def prompt_context(self) -> str:
        if not self.records:
            return "No loop warning."
        last = self.records[-1]
        unchanged_run = 0
        for record in reversed(self.records):
            if record.before_signature == record.after_signature:
                unchanged_run += 1
            else:
                break
        if unchanged_run:
            return (
                f"Warning: the textual page state was unchanged for {unchanged_run} "
                "consecutive action(s). Do not blindly repeat the same action."
            )
        if last.assessment == "failed":
            return "Warning: the previous action failed. Re-observe and choose a corrected action."
        return "No loop warning."

    @staticmethod
    def action_signature(action: ActionCall) -> str:
        return json.dumps(action.to_dict(), ensure_ascii=False, sort_keys=True)
