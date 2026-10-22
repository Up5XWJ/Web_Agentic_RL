"""Online environment -> grouped trajectories -> GRPO update -> fresh rollout.

Configuration checks and checkpoint-state inspection do not import CUDA, Ray or
veRL. The runnable backend is loaded only after configuration validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable

from .scheduler import SchedulerConfig
from .training_data import prepare_training_batch
from .types import Episode, TaskSpec


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    result = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(result, dict):
        raise ValueError("Training configuration must be a mapping")
    return result


def _positive(config: dict[str, Any], name: str, minimum: int = 1) -> None:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config: dict[str, Any], check_paths: bool = False) -> dict[str, Any]:
    """Validate concurrent environments with synchronous batched veRL calls."""
    if config.get("backend") != "verl":
        raise ValueError("This training entrypoint requires backend: verl")
    for key in ("epochs", "max_updates", "tasks_per_update"):
        _positive(config, key)
    _positive(config, "group_size", 2)
    scheduler = SchedulerConfig(**config.get("rollout_scheduler", {}))
    for key in ("checkpoint_every", "eval_every"):
        _positive(config, key, 0)
    for key in ("manifest", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} is required")
    if check_paths and not Path(config["manifest"]).is_file():
        raise ValueError(f"Missing task manifest: {config['manifest']}")
    if config.get("eval_every") and not config.get("eval_manifest"):
        raise ValueError("eval_every requires an eval_manifest")
    if check_paths and config.get("eval_manifest") and not Path(config["eval_manifest"]).is_file():
        raise ValueError(f"Missing evaluation manifest: {config['eval_manifest']}")
    grpo = config.get("grpo", {})
    epsilon = grpo.get("epsilon", 1e-6)
    if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("grpo.epsilon must be finite and positive")
    if not isinstance(grpo.get("normalize_std", False), bool):
        raise ValueError("grpo.normalize_std must be a boolean")
    if not isinstance(config.get("harness", {}).get("use_screenshot", False), bool):
        raise ValueError("harness.use_screenshot must be a boolean")
    verl = config.get("verl", {})
    model = verl.get("actor_rollout_ref", {}).get("model", {})
    actor = verl.get("actor_rollout_ref", {}).get("actor", {})
    rollout = verl.get("actor_rollout_ref", {}).get("rollout", {})
    if not model.get("path"):
        raise ValueError("verl.actor_rollout_ref.model.path is required")
    if model.get("lora_adapter_path"):
        raise ValueError("Resume sharded checkpoints with resume_from; a preloaded LoRA would change reference-KL semantics")
    if actor.get("strategy", "fsdp") not in ("fsdp", "fsdp2"):
        raise ValueError("The online backend supports FSDP/FSDP2 only")
    if actor.get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("Sequence parallelism is not supported by the initial global-loss adapter")
    if actor.get("ppo_epochs", 1) != 1:
        raise ValueError("Use exactly one optimizer step per online batch (ppo_epochs: 1)")
    if actor.get("policy_loss", {}).get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("This backend implements the clipped GRPO/PPO vanilla objective")
    if actor.get("loss_agg_mode", "token-mean") != "token-mean":
        raise ValueError("This backend uses a global generated-token mean objective")
    for key in ("entropy_coeff", "kl_loss_coef"):
        value = actor.get(key, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"actor.{key} must be finite and nonnegative")
    if verl.get("algorithm", {}).get("use_kl_in_reward", False):
        raise ValueError("Use actor.use_kl_loss; do not also penalize reward with KL")
    if rollout.get("name") != "vllm" or rollout.get("mode") != "sync" or rollout.get("n", 1) != 1:
        raise ValueError("Use synchronous vllm with n=1; the harness owns episode grouping")
    if rollout.get("multi_turn", {}).get("enable", False):
        raise ValueError("veRL's default multi-turn concatenation must be disabled")
    if rollout.get("top_p", 1) != 1 or rollout.get("top_k", -1) != -1 or rollout.get("repetition_penalty", 1) != 1:
        raise ValueError("Use top_p=1, top_k=-1 and repetition_penalty=1 for consistent policy likelihoods")
    for name in ("min_p", "presence_penalty", "frequency_penalty", "min_tokens"):
        if rollout.get(name, 0) != 0:
            raise ValueError(f"rollout.{name} changes the sampled distribution and must be zero")
    for name in ("logit_bias", "logits_processors", "allowed_token_ids", "bad_words", "stop", "stop_token_ids"):
        if rollout.get(name):
            raise ValueError(f"rollout.{name} is unsupported by the exact-likelihood adapter")
    unsupported_sampling = {
        "repetition_penalty", "min_p", "presence_penalty", "frequency_penalty", "min_tokens",
        "logit_bias", "logits_processors", "allowed_token_ids", "bad_words", "stop", "stop_token_ids",
    }
    supplied_unsupported = unsupported_sampling.intersection(rollout)
    if supplied_unsupported:
        raise ValueError("Remove sampling fields absent from upstream veRL 0.6.1 RolloutConfig: "
                         + ", ".join(sorted(supplied_unsupported)))
    if not isinstance(rollout.get("temperature"), (int, float)) or not math.isfinite(rollout["temperature"]) or rollout["temperature"] <= 0:
        raise ValueError("Training requires a finite positive sampling temperature")
    for name in ("prompt_length", "response_length"):
        _positive(rollout, name)
    total = rollout["prompt_length"] + rollout["response_length"]
    if rollout.get("max_model_len", total) < total:
        raise ValueError("max_model_len must cover both prompt and full response budget")
    if rollout.get("enable_chunked_prefill", True) and rollout.get("max_num_batched_tokens", 8192) < rollout.get("max_model_len", total):
        raise ValueError("veRL v0.6.1 chunked prefill requires max_num_batched_tokens >= max_model_len")
    trainer = verl.get("trainer", {})
    for name in ("nnodes", "n_gpus_per_node"):
        _positive(trainer, name)
    world = trainer["nnodes"] * trainer["n_gpus_per_node"]
    tp = rollout.get("tensor_model_parallel_size", 1)
    if not isinstance(tp, int) or tp < 1 or world % tp:
        raise ValueError("Rollout tensor parallel size must divide the GPU count")
    if actor.get("ppo_mini_batch_size", world) < world:
        raise ValueError("veRL worker initialization requires ppo_mini_batch_size >= GPU count")
    if check_paths:
        from .environment import BrowserGymEnvironment
        from .harness import HarnessConfig
        from .rewards import RewardConfig
        from .tasks import load_manifest

        HarnessConfig(**config.get("harness", {}))
        RewardConfig(**config.get("reward", {}))
        environment = config.get("environment", {})
        timeout = environment.get("timeout_seconds", 120)
        viewport = environment.get("viewport", [1024, 640])
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("environment.timeout_seconds must be finite and positive")
        if (not isinstance(viewport, (list, tuple)) or len(viewport) != 2
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in viewport)):
            raise ValueError("environment.viewport requires two positive integer dimensions")
        if not isinstance(environment.get("headless", True), bool):
            raise ValueError("environment.headless must be boolean")
        # Constructor validates arguments but does not launch a process/browser.
        BrowserGymEnvironment(**environment).close()
        tasks = load_manifest(config["manifest"])
        if not tasks:
            raise ValueError("The training manifest is empty")
        if config.get("eval_manifest"):
            eval_tasks = load_manifest(config["eval_manifest"])
            if not eval_tasks:
                raise ValueError("The evaluation manifest is empty")
            overlap = {t.family_id for t in tasks} & {t.family_id for t in eval_tasks}
            if overlap:
                raise ValueError("Train/evaluation manifests overlap in underlying task families")
    return {"backend": "verl", "verl_version": "0.6.1", "world_size": world,
            "model": model["path"], "group_size": config["group_size"],
            "tasks_per_update": config["tasks_per_update"],
            "episodes_per_batch": config["tasks_per_update"] * config["group_size"],
            "rollout_scheduler": asdict(scheduler),
            "max_prompt_tokens": rollout["prompt_length"], "max_response_tokens": rollout["response_length"],
            "loss": "global mean over generated memory/action/retry tokens",
            "checked_gpu_runtime": False}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    # Operational paths and evaluation/save cadence do not define the policy.
    ignored = {"resume_from", "output_dir", "checkpoint_every", "eval_every", "eval_manifest"}
    return _hash({key: value for key, value in config.items() if key not in ignored})


def _harness_fingerprint(config: dict[str, Any]) -> str:
    from .harness import HarnessConfig, MemoryHarness
    return MemoryHarness(HarnessConfig(**config.get("harness", {}))).metadata()["harness_fingerprint"]


def _pipeline_fingerprint() -> str:
    """Guard resume against a changed reward, environment or training algorithm."""
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in ("train.py", "training_data.py", "verl_backend.py", "rollout.py", "scheduler.py",
                 "rewards.py", "environment.py", "types.py", "vision.py", "../mcp/appserver.py"):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


@dataclass
class TaskCursor:
    task_count: int
    max_epochs: int
    seed: int
    epoch: int = 0
    index: int = 0
    order: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if not self.order:
            self.order = list(range(self.task_count))
            self.rng.shuffle(self.order)

    def take(self, count: int) -> list[int]:
        indices: list[int] = []
        while len(indices) < count and self.epoch < self.max_epochs:
            if self.index == self.task_count:
                self.epoch += 1
                self.index = 0
                if self.epoch >= self.max_epochs:
                    break
                self.rng.shuffle(self.order)
            indices.append(self.order[self.index])
            self.index += 1
        return indices

    def state_dict(self) -> dict[str, Any]:
        return asdict(self) | {"rng_state": self.rng.getstate()}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TaskCursor:
        values = dict(state)
        rng_state = values.pop("rng_state")
        cursor = cls(**values)
        cursor.rng.setstate(_tuple_tree(rng_state))
        return cursor


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_training_checkpoint(
    backend: Any, config: dict[str, Any], tasks: list[TaskSpec], cursor: TaskCursor, batch_count: int,
) -> Path:
    state = backend.state_dict()
    path = Path(config["output_dir"]) / "checkpoints" / f"update-{state['update_count']:06d}-batch-{batch_count:06d}"
    state_file = path / "training_state.json"
    if state_file.exists():
        existing = json.loads(state_file.read_text())
        if existing["backend"] == state and existing["batch_count"] == batch_count:
            return path
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    backend.save_checkpoint(path)
    _write_json(path / "training_state.json", {
        "format_version": 1, "backend": state, "cursor": cursor.state_dict(),
        "batch_count": batch_count, "config_hash": _config_hash(config),
        "tasks_hash": _hash([asdict(t) for t in tasks]),
        "harness_fingerprint": _harness_fingerprint(config),
        "pipeline_fingerprint": _pipeline_fingerprint(),
    })
    _write_json(path / "run_config.json", config)
    _write_json(Path(config["output_dir"]) / "latest_checkpoint.json", {"path": str(path.resolve())})
    return path


def _resolve_checkpoint(path: str | Path) -> Path:
    directory = Path(path)
    if (directory / "latest_checkpoint.json").is_file():
        directory = Path(json.loads((directory / "latest_checkpoint.json").read_text())["path"])
    if not (directory / "training_state.json").is_file():
        raise ValueError(f"No complete training checkpoint at {directory}")
    return directory


def run_online_loop(
    config: dict[str, Any], tasks: list[TaskSpec], backend: Any,
    environment_factory: Callable[[], Any], *, harness_config: Any = None,
    reward_fn: Callable | None = None, sample_group_fn: Callable | None = None,
    evaluation_fn: Callable | None = None,
) -> dict[str, Any]:
    """Testable orchestration used unchanged by the GPU entrypoint."""
    from .rollout import RolloutSamplingError, sample_batch
    scheduler_config = SchedulerConfig(**config.get("rollout_scheduler", {}))
    if not tasks:
        raise ValueError("The training manifest contains no tasks")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output_dir is nonempty; choose a fresh directory, including when resuming")
    output.mkdir(parents=True, exist_ok=True)
    cursor = TaskCursor(len(tasks), config["epochs"], config.get("seed", 0))
    batch_count = 0
    if config.get("resume_from"):
        checkpoint = _resolve_checkpoint(config["resume_from"])
        state = json.loads((checkpoint / "training_state.json").read_text())
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint state version")
        if state["config_hash"] != _config_hash(config) or state["tasks_hash"] != _hash([asdict(t) for t in tasks]):
            raise ValueError("Resume requires the same training config and manifest content")
        if (state.get("harness_fingerprint") != _harness_fingerprint(config)
                or state.get("pipeline_fingerprint") != _pipeline_fingerprint()):
            raise ValueError("Resume requires the same Harness and training pipeline source fingerprints")
        backend.load_checkpoint(checkpoint, state["backend"])
        cursor = TaskCursor.from_state(state["cursor"])
        batch_count = int(state["batch_count"])
    last_metrics: dict[str, float] = {}
    while backend.state_dict()["update_count"] < config["max_updates"]:
        selected = cursor.take(config["tasks_per_update"])
        if not selected:
            break
        batch_count += 1
        groups = [(f"batch-{batch_count}:slot-{slot}:task-{task_index}", tasks[task_index])
                  for slot, task_index in enumerate(selected)]
        sampling_metrics: dict[str, float] = {}
        sampling_error = None
        if sample_group_fn is None:
            try:
                sampled = sample_batch(
                    groups, backend.policy, environment_factory, group_size=config["group_size"],
                    scheduler_config=scheduler_config, seed=config.get("seed", 0),
                    harness_config=harness_config, reward_fn=reward_fn,
                )
            except RolloutSamplingError as exc:
                sampled, sampling_error = exc.batch, exc
            episodes = sampled.episodes
            sampling_metrics = sampled.metrics
        else:
            # Explicit test/custom sampler hook; the production path above
            # always uses the shared bounded dispatcher and batch barrier.
            episodes: list[Episode] = []
            for group_id, task in groups:
                episodes.extend(sample_group_fn(
                    task, backend.policy, environment_factory,
                    group_size=config["group_size"], group_id=group_id,
                    harness_config=harness_config, reward_fn=reward_fn,
                ))
        # Persist rejected generations and infrastructure failures too.
        trace = output / "episodes" / f"batch-{batch_count:06d}.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        from .artifacts import write_episodes
        write_episodes(trace, episodes)
        if sampling_error is not None:
            _write_json(trace.with_suffix(".failure.json"), {
                "batch": batch_count, "error": str(sampling_error), **sampling_metrics,
            })
            raise sampling_error
        grpo = config.get("grpo", {})
        packed = prepare_training_batch(
            episodes, group_size=config["group_size"], policy_version=backend.policy.policy_version,
            normalize_std=grpo.get("normalize_std", False), epsilon=grpo.get("epsilon", 1e-6),
        )
        last_metrics = dict(packed.metrics) | sampling_metrics
        informative = any(abs(row.advantage) > 0 for row in packed.rows)
        update_started = time.monotonic()
        if packed.rows and informative:
            last_metrics.update(backend.update(packed.rows))
            last_metrics["train/update_applied"] = 1.0
        else:
            last_metrics["train/update_applied"] = 0.0
        last_metrics["train/update_seconds"] = time.monotonic() - update_started
        update_count = backend.state_dict()["update_count"]
        record = {"batch": batch_count, "update": update_count,
                  "policy_version": backend.policy.policy_version, **last_metrics}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if last_metrics["train/update_applied"]:
            if config["eval_every"] and update_count % config["eval_every"] == 0 and evaluation_fn:
                # All rollout sampling uses explicit per-call seeds. Preserve
                # the training counter around evaluation for reproducible resume.
                counter = backend.policy.sample_counter
                try:
                    evaluation_fn(backend.policy, update_count)
                finally:
                    backend.policy.sample_counter = counter
            if config["checkpoint_every"] and update_count % config["checkpoint_every"] == 0:
                save_training_checkpoint(backend, config, tasks, cursor, batch_count)
    # A zero-update run still has a complete, explicitly labeled checkpoint.
    checkpoint = save_training_checkpoint(backend, config, tasks, cursor, batch_count)
    return {"updates": backend.state_dict()["update_count"], "batches": batch_count,
            "checkpoint": str(checkpoint), "last_metrics": last_metrics}


def run(config: dict[str, Any], *, evaluate_only: bool = False) -> dict[str, Any]:
    validate_config(config, check_paths=True)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output_dir is nonempty; choose a fresh directory, including when resuming/evaluating")
    from .artifacts import summarize, write_episodes
    from .environment import BrowserGymEnvironment
    from .harness import HarnessConfig
    from .rewards import RewardConfig, score_episode
    from .rollout import RolloutJob, sample_episodes
    from .tasks import load_manifest
    from .verl_backend import VerlBackend

    tasks = load_manifest(config["manifest"])
    eval_tasks = load_manifest(config["eval_manifest"]) if config.get("eval_manifest") else []
    if evaluate_only and not eval_tasks:
        raise ValueError("Evaluation requires an eval_manifest")
    harness = HarnessConfig(**config.get("harness", {}))
    reward_config = RewardConfig(**config.get("reward", {}))
    reward_fn = lambda ep: score_episode(ep, reward_config)
    environment_factory = lambda: BrowserGymEnvironment(**config.get("environment", {}))

    def evaluate(policy, update):
        sampled = sample_episodes(
            [RolloutJob(task, f"eval-{update}:task-{i}", f"eval-{update}-{i}") for i, task in enumerate(eval_tasks)],
            policy, environment_factory, scheduler_config=SchedulerConfig(**config.get("rollout_scheduler", {})),
            seed=config.get("seed", 0), harness_config=harness, reward_fn=reward_fn,
        )
        episodes = sampled.episodes
        result = {"update": update, "policy_version": policy.policy_version,
                  **summarize(episodes), **sampled.metrics}
        destination = Path(config["output_dir"]) / "eval" / f"update-{update:06d}"
        destination.mkdir(parents=True, exist_ok=False)
        _write_json(destination / "metrics.json", result)
        write_episodes(destination / "episodes.jsonl", episodes)
        print(json.dumps({"evaluation": result}), flush=True)
        return result

    backend = VerlBackend(config)
    try:
        if evaluate_only:
            if config.get("resume_from"):
                checkpoint = _resolve_checkpoint(config["resume_from"])
                state = json.loads((checkpoint / "training_state.json").read_text())
                backend.load_checkpoint(checkpoint, state["backend"])
            return evaluate(backend.policy, backend.update_count)
        return run_online_loop(config, tasks, backend, environment_factory,
                               harness_config=harness, reward_fn=reward_fn, evaluation_fn=evaluate)
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/agentic_rl/verl.yaml")
    parser.add_argument("--set", "--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--manifest")
    parser.add_argument("--model")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    for value, key in ((args.manifest, "manifest"), (args.resume, "resume_from")):
        if value:
            config[key] = value
    if args.model:
        config["verl"]["actor_rollout_ref"]["model"]["path"] = args.model
    result = validate_config(config) if args.check_config else run(config, evaluate_only=args.evaluate)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
