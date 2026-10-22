"""Bounded, asynchronous Harness requests with a single batched model caller.

Environment loops run independently in CPU threads/processes. Only this
dispatcher calls the shared veRL worker, so generation/log-prob collectives
cannot interleave. Exiting the scheduler is a barrier before actor updates.
"""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeout
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
from queue import Empty, Full, Queue
import threading
import time
from typing import Any

from .types import BatchPolicy, Generation, GenerationRequest, InfrastructureError


@dataclass(frozen=True)
class SchedulerConfig:
    max_concurrent_episodes: int = 8
    inference_batch_size: int = 2
    batch_wait_ms: float = 10.0
    request_timeout_seconds: float = 600.0

    def __post_init__(self):
        for name in ("max_concurrent_episodes", "inference_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"rollout_scheduler.{name} must be a positive integer")
        for name in ("batch_wait_ms", "request_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"rollout_scheduler.{name} must be finite and nonnegative")
        if self.request_timeout_seconds <= self.batch_wait_ms / 1000:
            raise ValueError("request_timeout_seconds must exceed the batching wait")


def request_seed(base_seed: int, policy_version: str, request_id: str) -> int:
    """Stable across batch composition, thread completion order and resume."""
    payload = f"{base_seed}\0{policy_version}\0{request_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass
class _Pending:
    request: GenerationRequest
    future: Future = field(default_factory=Future)
    submitted_at: float = field(default_factory=time.monotonic)


class PolicyProxy:
    """Per-episode state; the existing synchronous Harness API stays unchanged."""

    def __init__(self, scheduler: RolloutScheduler, episode_id: str):
        self.scheduler = scheduler
        self.episode_id = episode_id
        self.call_index = 0

    def generate(self, messages: list[dict[str, Any]]) -> Generation:
        request_id = f"{self.episode_id}/call-{self.call_index}"
        self.call_index += 1  # Includes every format/action repair attempt.
        version = self.scheduler.policy_version
        request = GenerationRequest(
            request_id, deepcopy(messages), request_seed(self.scheduler.seed, version, request_id), version,
        )
        return self.scheduler.generate(request)


class RolloutScheduler:
    def __init__(self, policy: BatchPolicy, config: SchedulerConfig, *, seed: int = 0):
        if not callable(getattr(policy, "generate_batch", None)):
            raise TypeError("Concurrent rollout requires policy.generate_batch(requests)")
        self.policy = policy
        self.config = config
        self.seed = seed
        self.policy_version = policy.policy_version
        self._queue: Queue[_Pending] = Queue(maxsize=config.max_concurrent_episodes)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: dict[str, _Pending] = {}
        self._seen: set[str] = set()
        self._error: InfrastructureError | None = None
        # A failed/hung RPC must never allow another optimizer update. The
        # caller aborts training and shuts down Ray; Python cannot kill a thread.
        self._thread = threading.Thread(target=self._serve, name="rollout-inference", daemon=True)
        self._requests = self._batches = self._tokens = self._max_batch = 0
        self._queue_seconds = self._inference_seconds = 0.0

    def __enter__(self):
        self._thread.start()
        return self

    def policy_for(self, episode_id: str) -> PolicyProxy:
        return PolicyProxy(self, episode_id)

    def generate(self, request: GenerationRequest) -> Generation:
        pending = _Pending(request)
        with self._lock:
            if self._stop.is_set() or not self._thread.is_alive():
                raise self._error or InfrastructureError("Rollout scheduler is not running")
            if request.policy_version != self.policy_version:
                raise InfrastructureError("Request has a different policy version")
            if request.request_id in self._seen:
                raise InfrastructureError(f"Duplicate inference request: {request.request_id}")
            self._seen.add(request.request_id)
            self._pending[request.request_id] = pending
            self._requests += 1
        try:
            self._queue.put(pending, timeout=self.config.request_timeout_seconds)
            remaining = self.config.request_timeout_seconds - (time.monotonic() - pending.submitted_at)
            return pending.future.result(timeout=max(0, remaining))
        except (Full, FutureTimeout) as exc:
            if pending.future.done():
                return pending.future.result()
            self.abort(f"Inference request timed out: {request.request_id}")
            raise self._error from exc

    def abort(self, reason: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = InfrastructureError(reason)
            self._stop.set()
            for pending in self._pending.values():
                if not pending.future.done():
                    pending.future.set_exception(self._error)
            self._pending.clear()

    def _serve(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    first = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                batch = [first]
                deadline = time.monotonic() + self.config.batch_wait_ms / 1000
                while len(batch) < self.config.inference_batch_size:
                    remaining = deadline - time.monotonic()
                    try:
                        batch.append(self._queue.get(timeout=max(0, remaining)))
                    except Empty:
                        break
                if self._stop.is_set():
                    break
                if self.policy.policy_version != self.policy_version:
                    raise InfrastructureError("Policy changed while rollout requests were active")
                started = time.monotonic()
                results = self.policy.generate_batch([item.request for item in batch])
                elapsed = time.monotonic() - started
                if self.policy.policy_version != self.policy_version:
                    raise InfrastructureError("Policy changed during batched inference")
                if len(results) != len(batch):
                    raise InfrastructureError("Inference batch returned a different number of results")
                # Validate correspondence before releasing ANY result.
                for pending, result in zip(batch, results, strict=True):
                    if isinstance(result, Exception):
                        continue  # A row-local context error disqualifies its own group only.
                    request = pending.request
                    if (not isinstance(result, Generation)
                            or result.policy_version != request.policy_version
                            or result.metadata.get("request_id") != request.request_id
                            or result.metadata.get("sample_seed") != request.sample_seed):
                        raise InfrastructureError("Inference result identity/seed/policy version mismatch")
                with self._lock:
                    self._batches += 1
                    self._max_batch = max(self._max_batch, len(batch))
                    self._inference_seconds += elapsed
                    self._queue_seconds += sum(started - item.submitted_at for item in batch)
                    for pending, result in zip(batch, results, strict=True):
                        self._pending.pop(pending.request.request_id, None)
                        if pending.future.done():
                            continue
                        if isinstance(result, Exception):
                            pending.future.set_exception(result)
                        else:
                            self._tokens += len(result.response_token_ids)
                            pending.future.set_result(result)
        except BaseException as exc:
            self.abort(f"Batched inference failed: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        with self._lock:
            unfinished = bool(self._pending)
        if unfinished:
            self.abort("Scheduler closed with outstanding requests")
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            self.abort("Inference worker did not drain; refusing further policy updates")
        if self._error:
            raise self._error

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.abort(f"Rollout sampling interrupted: {exc}")
        self.close()

    def metrics(self) -> dict[str, float]:
        with self._lock:
            return {
                "rollout/inference_requests": float(self._requests),
                "rollout/inference_batches": float(self._batches),
                "rollout/mean_inference_batch_size": self._requests / max(self._batches, 1),
                "rollout/max_inference_batch_size": float(self._max_batch),
                "rollout/mean_queue_wait_seconds": self._queue_seconds / max(self._requests, 1),
                "rollout/inference_seconds": self._inference_seconds,
                "rollout/generated_tokens": float(self._tokens),
            }
