"""Episode-level GRPO advantages and exact, non-cumulative model-call rows.

No GPU packages are imported here. A row is one actual model invocation, including
unsuccessful parse/retry attempts; it is never a reconstructed full conversation.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any

from .types import Episode
from .vision import encode_prompt, has_images, object_array, pack_prompts


@dataclass
class TrainingRow:
    episode_id: str
    group_id: str
    policy_version: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    old_log_probs: list[float]
    advantage: float
    reward: float
    step_index: int
    attempt_index: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    input_encoding: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    rows: list[TrainingRow] = field(default_factory=list)
    episode_advantages: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def prepare_training_batch(
    episodes: list[Episode],
    *,
    group_size: int,
    policy_version: str,
    normalize_std: bool = False,
    epsilon: float = 1e-6,
) -> TrainingBatch:
    """Compute advantages before flattening: retries must not reweight a group.

    One infrastructure failure invalidates its entire comparison group. A policy
    failure (invalid JSON, exhausted browser budget, etc.) remains a valid sample.
    """
    if group_size < 2 or epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("GRPO requires group_size >= 2 and a finite positive epsilon")
    groups: dict[str, list[Episode]] = defaultdict(list)
    seen: set[str] = set()
    for episode in episodes:
        if not episode.group_id or episode.episode_id in seen:
            raise ValueError("Each episode needs a group id and a unique episode id")
        seen.add(episode.episode_id)
        groups[episode.group_id].append(episode)
    result = TrainingBatch()
    skipped = zero_variance = valid = successes = submissions = incorrect_finishes = 0
    rewards: list[float] = []
    for group_id, group in groups.items():
        if len(group) != group_size:
            raise ValueError(f"Incomplete group {group_id}: {len(group)} != {group_size}")
        if any(not e.trainable or e.infrastructure_error for e in group):
            skipped += 1
            continue
        identities = {(e.task_id, e.seed, e.initial_state_hash) for e in group}
        if len(identities) != 1 or not group[0].initial_state_hash:
            raise ValueError(f"Group {group_id} was not sampled from one verified initial state")
        if any(e.policy_version != policy_version for e in group):
            raise ValueError(f"Group {group_id} contains stale/mixed policy versions")
        if any(not math.isfinite(e.reward) for e in group):
            raise ValueError(f"Non-finite reward in {group_id}")
        mean = math.fsum(e.reward for e in group) / len(group)
        # Sample std follows veRL GRPO; the default disables std normalization.
        variance = math.fsum((e.reward - mean) ** 2 for e in group) / (len(group) - 1)
        std = math.sqrt(variance)
        zero_variance += int(std <= epsilon)
        denominator = std + epsilon if normalize_std else 1.0
        valid += 1
        for episode in group:
            advantage = (episode.reward - mean) / denominator
            result.episode_advantages[episode.episode_id] = advantage
            successes += int(episode.success)
            submissions += int(episode.submitted)
            incorrect_finishes += int(episode.terminal_reason == "incorrect_finish")
            rewards.append(episode.reward)
            for attempt in episode.attempts:
                generation = attempt.generation
                metadata = generation.metadata
                if (metadata.get("trainable") is False or metadata.get("synthetic_token_ids")
                        or metadata.get("backend") in {"scripted", "synthetic", "mock"}):
                    raise ValueError("Synthetic/scripted generations cannot enter a real policy update")
                if generation.policy_version != policy_version:
                    raise ValueError("A model call has a stale policy version")
                if not generation.prompt_token_ids or not generation.response_token_ids:
                    raise ValueError("Cannot train an empty or unrecorded model invocation")
                if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in generation.prompt_token_ids + generation.response_token_ids):
                    raise ValueError("Recorded token IDs must be nonnegative integers")
                if len(generation.old_log_probs) != len(generation.response_token_ids):
                    raise ValueError("Every generated token needs a real old-policy log probability")
                if not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                           and math.isfinite(v) and v <= 1e-6 for v in generation.old_log_probs):
                    raise ValueError("Invalid old-policy log probability")
                encoding = metadata.get("input_encoding", {})
                if has_images(attempt.messages) and not encoding:
                    raise ValueError("Screenshot model calls require recorded input encoding provenance")
                result.rows.append(TrainingRow(
                    episode_id=episode.episode_id,
                    group_id=group_id,
                    policy_version=policy_version,
                    prompt_token_ids=list(generation.prompt_token_ids),
                    response_token_ids=list(generation.response_token_ids),
                    old_log_probs=list(generation.old_log_probs),
                    advantage=advantage,
                    reward=episode.reward,
                    step_index=attempt.step_index,
                    attempt_index=attempt.attempt_index,
                    messages=deepcopy(attempt.messages) if encoding else [],
                    input_encoding=deepcopy(encoding),
                ))
    result.metrics = {
        "train/valid_groups": float(valid),
        "train/skipped_groups": float(skipped),
        "train/zero_variance_groups": float(zero_variance),
        "train/episodes": float(len(rewards)),
        "train/model_calls": float(len(result.rows)),
        "train/success_rate": successes / len(rewards) if rewards else 0.0,
        "train/submission_rate": submissions / len(rewards) if rewards else 0.0,
        "train/incorrect_finish_rate": incorrect_finishes / len(rewards) if rewards else 0.0,
        "train/reward_mean": math.fsum(rewards) / len(rewards) if rewards else 0.0,
    }
    return result


def tensor_batch(
    rows: list[TrainingRow], *, pad_token_id: int, world_size: int,
    max_prompt_tokens: int, max_response_tokens: int, temperature: float,
    tokenizer: Any = None, processor: Any = None,
) -> Any:
    """Make a veRL DataProto lazily; padding contributes exactly zero loss.

    Attention padding and loss padding differ: dummy rows retain valid inputs so
    every FSDP rank runs the same number of forwards. Their response_mask is zero,
    and the global loss denominator counts only original generated tokens.
    """
    import numpy as np
    import torch
    from verl import DataProto

    if not rows or world_size < 1:
        raise ValueError("At least one row and one worker are required")
    if any(len(r.prompt_token_ids) > max_prompt_tokens for r in rows):
        raise ValueError("Prompt exceeds budget; refusing silent training-time truncation")
    if any(len(r.response_token_ids) > max_response_tokens for r in rows):
        raise ValueError("Response exceeds the configured generation budget")
    real_count = len(rows)
    padded_count = math.ceil(real_count / world_size) * world_size
    padded = rows + [rows[0]] * (padded_count - real_count)
    prompt_len = max(len(r.prompt_token_ids) for r in rows)
    response_len = max(len(r.response_token_ids) for r in rows)
    prompts = torch.full((padded_count, prompt_len), pad_token_id, dtype=torch.long)
    responses = torch.full((padded_count, response_len), pad_token_id, dtype=torch.long)
    attention = torch.zeros((padded_count, prompt_len + response_len), dtype=torch.long)
    mask = torch.zeros((padded_count, response_len), dtype=torch.long)
    old = torch.zeros((padded_count, response_len), dtype=torch.float32)
    advantages = torch.zeros_like(old)
    for i, row in enumerate(padded):
        p, r = len(row.prompt_token_ids), len(row.response_token_ids)
        prompts[i, -p:] = torch.tensor(row.prompt_token_ids)
        responses[i, :r] = torch.tensor(row.response_token_ids)
        attention[i, prompt_len - p:prompt_len + r] = 1
        old[i, :r] = torch.tensor(row.old_log_probs)
        if i < real_count:
            mask[i, :r] = 1
            advantages[i, :r] = row.advantage
    positions = (attention.cumsum(-1) - 1).clamp_min(0) * attention
    non_tensors = {"episode_id": np.array([r.episode_id for r in padded], dtype=object)}
    if processor is not None or any(row.input_encoding or has_images(row.messages) for row in rows):
        if processor is None or tokenizer is None:
            raise ValueError("Recorded multimodal inputs require their original processor and tokenizer")
        encoded = []
        for row in rows:
            if not row.input_encoding or not row.messages:
                raise ValueError("Missing multimodal prompt provenance; cannot recreate policy inputs")
            item = encode_prompt(row.messages, tokenizer, processor,
                                 chat_template_kwargs=row.input_encoding.get("chat_template_kwargs", {}))
            if item.token_ids != row.prompt_token_ids or item.provenance != row.input_encoding:
                raise ValueError("Recorded multimodal input changed: pixels, token IDs or MRoPE positions differ")
            encoded.append(item)
        encoded += [encoded[0]] * (padded_count - real_count)
        packed, _ = pack_prompts(encoded, pad_token_id)
        if not torch.equal(packed["input_ids"], prompts):
            raise ValueError("Multimodal prompt padding differs from recorded tokens")
        # Exactly match upstream synchronous rollout's response position extension.
        delta = torch.arange(1, response_len + 1).view(1, 1, -1)
        response_positions = packed["position_ids"][..., -1:] + delta
        positions = torch.cat((packed["position_ids"], response_positions), dim=-1)
        non_tensors["multi_modal_inputs"] = object_array([item.image_inputs for item in encoded])
    return DataProto.from_dict(
        tensors={
            "prompts": prompts, "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention, "position_ids": positions,
            "response_mask": mask, "old_log_probs": old,
            "advantages": advantages,
        },
        non_tensors=non_tensors,
        meta_info={
            "temperature": temperature,
            "global_token_num": attention.sum(-1).tolist(),
            "global_response_tokens": int(mask.sum()),
            "real_model_calls": real_count,
        },
    )
