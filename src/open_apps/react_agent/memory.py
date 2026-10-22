"""Fixed-size overwrite memory with deterministic safety checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .schemas import ActionAssessment, MemoryBlock, SchemaError


@dataclass
class MemoryConfig:
    max_tokens: int = 1024
    max_subgoals: int = 12
    max_verified_facts: int = 20
    max_failures: int = 3
    max_uncertainties: int = 5


class MemoryManager:
    """Validate a model-authored full replacement for the previous memory.

    Token counting uses a conservative four-characters-per-token estimate. It
    is provider-independent and is only a hard guard; actual API usage remains
    the source of truth in experiment statistics.
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()

    def initialize(self) -> MemoryBlock:
        return MemoryBlock.empty()

    def prepare_update(
        self,
        previous: MemoryBlock,
        candidate: MemoryBlock,
        automatic_assessment: ActionAssessment,
        *,
        current_url: str,
        focused_bid: str,
        valid_bids: Iterable[str],
    ) -> MemoryBlock:
        """Return a safe, bounded candidate or raise ``SchemaError``."""
        self._validate_plan(previous, candidate, automatic_assessment)
        self._deduplicate(candidate)

        config = self.config
        candidate.failures = candidate.failures[-config.max_failures :]
        candidate.uncertainties = candidate.uncertainties[-config.max_uncertainties :]
        candidate.verified_facts = candidate.verified_facts[-config.max_verified_facts :]

        valid_bid_set = {str(bid) for bid in valid_bids}
        candidate.transient_page_state.url = current_url
        candidate.transient_page_state.focused_bid = focused_bid or ""
        candidate.transient_page_state.bids = {
            name: bid
            for name, bid in candidate.transient_page_state.bids.items()
            if bid in valid_bid_set
        }

        estimated_tokens = self.estimate_tokens(candidate)
        if estimated_tokens > config.max_tokens:
            raise SchemaError(
                "memory_next is too large: approximately "
                f"{estimated_tokens} tokens, limit is {config.max_tokens}. "
                "Rewrite it more compactly without deleting plan steps or verified facts."
            )
        return candidate

    def estimate_tokens(self, memory: MemoryBlock) -> int:
        payload = json.dumps(memory.to_dict(), ensure_ascii=False, separators=(",", ":"))
        return max(1, (len(payload) + 3) // 4)

    def _validate_plan(
        self,
        previous: MemoryBlock,
        candidate: MemoryBlock,
        assessment: ActionAssessment,
    ) -> None:
        if not candidate.plan:
            raise SchemaError(
                "memory_next.plan must contain the subgoals inferred from the user goal."
            )
        if len(candidate.plan) > self.config.max_subgoals:
            raise SchemaError(
                f"memory_next.plan has more than {self.config.max_subgoals} subgoals."
            )

        ids = [subgoal.id for subgoal in candidate.plan]
        if len(ids) != len(set(ids)):
            raise SchemaError("memory_next.plan contains duplicate subgoal ids.")

        old_by_id = {subgoal.id: subgoal for subgoal in previous.plan}
        new_by_id = {subgoal.id: subgoal for subgoal in candidate.plan}
        missing_ids = set(old_by_id) - set(new_by_id)
        if missing_ids:
            raise SchemaError(
                "Overwrite memory may compress details but must not delete plan steps. "
                f"Missing: {sorted(missing_ids)}."
            )

        for subgoal in candidate.plan:
            if subgoal.status == "verified" and not subgoal.evidence:
                raise SchemaError(
                    f"Verified subgoal {subgoal.id!r} must include observation evidence."
                )

        # A failed browser action cannot by itself verify the subgoal that was
        # active before that action. This catches the most damaging form of
        # optimistic memory update while still allowing other page evidence.
        if assessment.status == "failed" and previous.current_subgoal_id:
            old = old_by_id.get(previous.current_subgoal_id)
            new = new_by_id.get(previous.current_subgoal_id)
            if old and new and old.status != "verified" and new.status == "verified":
                raise SchemaError(
                    "The previous action failed, so the active subgoal cannot be marked "
                    "verified from that action. Keep it pending/in_progress and re-plan."
                )

        if candidate.current_subgoal_id is not None and candidate.current_subgoal_id not in new_by_id:
            raise SchemaError("memory_next.current_subgoal_id must reference a plan entry.")

        if candidate.current_subgoal_id is None and any(
            subgoal.status in {"pending", "in_progress"} for subgoal in candidate.plan
        ):
            raise SchemaError(
                "memory_next.current_subgoal_id is required while unfinished subgoals remain."
            )

    @staticmethod
    def _deduplicate(memory: MemoryBlock) -> None:
        facts_by_key = {}
        for fact in memory.verified_facts:
            facts_by_key[fact.key] = fact
        memory.verified_facts = list(facts_by_key.values())
        memory.pending = list(dict.fromkeys(memory.pending))
        memory.uncertainties = list(dict.fromkeys(memory.uncertainties))
