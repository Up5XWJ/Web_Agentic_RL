"""Batched online-policy adapter for upstream veRL **0.6.1**.

GPU dependencies are deliberately lazy. The harness calls ``generate`` once per
actual prompt; it never enters veRL's cumulative-conversation rollout machinery.
The upstream worker still owns FSDP, weight transfer to vLLM, optimizer state,
checkpointing and learning-rate scheduling.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any

from .training_data import TrainingRow, tensor_batch
from .types import ContextLimitError, Generation, GenerationRequest, InfrastructureError
from .vision import EncodedPrompt, encode_prompt, has_images, pack_prompts


VERL_VERSION = "0.6.1"


@contextmanager
def _per_request_sampling(rollout: Any, seeds: list[int], expected_prompt_ids: list[list[int]] | None = None):
    """Adapt the pinned sync rollout's single SamplingParams to a per-row list.

    vLLM 0.11 accepts one SamplingParams per prompt. Keep upstream's weight
    transfer, LoRA selection and token packing; intercept only its engine call.
    The dispatcher serializes worker calls, and the original method is always
    restored, including on failure. Seeds have already followed veRL DP slicing
    as non-tensor batch fields; all TP ranks receive the same local seed list.
    """
    engine = rollout.inference_engine
    original = engine.generate
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        prompts = kwargs.get("prompts")
        if args or prompts is None or len(prompts) != len(seeds) or calls != 1:
            raise RuntimeError("Unexpected veRL/vLLM batch call; check the pinned backend versions")
        template = kwargs["sampling_params"]
        parameters = []
        for seed in seeds:
            parameter = deepcopy(template)
            parameter.seed = int(seed)
            parameters.append(parameter)
        outputs = original(**{**kwargs, "sampling_params": parameters})
        if expected_prompt_ids is not None:
            if len(outputs) != len(expected_prompt_ids) or any(
                output.prompt_token_ids != expected for output, expected in zip(outputs, expected_prompt_ids)
            ):
                raise RuntimeError("vLLM/HF multimodal prompt token expansion differs; refusing mismatched RL scores")
        return outputs

    engine.generate = generate
    try:
        yield
        if calls != 1:
            raise RuntimeError("veRL did not invoke the expected synchronous vLLM engine")
    finally:
        engine.generate = original


def _online_update_policy(actor: Any, data: Any) -> dict[str, float]:
    """One globally normalized optimizer step, including variable-length retries.

    Official veRL splits a fixed mini-batch into local micro-batch means. Here the
    number of model calls varies per episode. Normalize all token sums by the
    *global* count, so DP dummy rows and micro-batch lengths do not change weights.
    Keep an equal number of forwards on every FSDP rank, including zero-loss pads.
    """
    import torch
    import torch.distributed as dist
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, kl_penalty
    from verl.utils.device import get_device_id

    actor.actor_module.train()
    actor.actor_optimizer.zero_grad()
    world = dist.get_world_size()
    total_tokens = int(data.meta_info["global_response_tokens"])
    if total_tokens <= 0:
        raise ValueError("No generated tokens to optimize")
    scale = world / total_tokens  # FSDP averages gradients across ranks.
    # pg, reference KL, entropy, old-policy approx KL, clip fraction, tokens
    sums = torch.zeros(6, dtype=torch.float64, device=get_device_id())
    for micro in data.split(1):
        micro = micro.to(get_device_id())
        inputs = {**micro.batch, **micro.non_tensor_batch}
        mask = inputs["response_mask"]
        entropy, log_probs = actor._forward_micro_batch(
            inputs, temperature=float(data.meta_info["temperature"]), calculate_entropy=True,
        )
        # This aggregation mode is sum/R; undo R before globally normalizing.
        pg, metrics = compute_policy_loss_vanilla(
            old_log_prob=inputs["old_log_probs"], log_prob=log_probs,
            advantages=inputs["advantages"], response_mask=mask,
            loss_agg_mode="seq-mean-token-sum-norm", config=actor.config,
        )
        pg_sum = pg * mask.shape[-1]
        kl_sum = log_probs.sum() * 0.0
        if actor.config.use_kl_loss:
            kl = kl_penalty(log_probs, inputs["ref_log_prob"], actor.config.kl_loss_type)
            kl_sum = (kl * mask).sum()
        entropy_sum = (entropy * mask).sum()
        loss = pg_sum
        if actor.config.use_kl_loss:
            loss = loss + actor.config.kl_loss_coef * kl_sum
        loss = loss - actor.config.entropy_coeff * entropy_sum
        (loss * scale).backward()
        tokens = mask.sum().detach()
        sums += torch.stack((
            pg_sum.detach(), kl_sum.detach(), entropy_sum.detach(),
            tokens * metrics["actor/ppo_kl"],
            tokens * metrics["actor/pg_clipfrac"], tokens,
        )).to(torch.float64)
    norm = actor._optimizer_step()
    actor.actor_optimizer.zero_grad()
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    denominator = float(sums[-1])
    values = (sums[:5] / max(denominator, 1)).cpu().tolist()
    grad_norm = float(norm.detach().cpu()) if hasattr(norm, "detach") else float(norm)
    return dict(zip(
        ("actor/pg_loss", "actor/ref_kl", "actor/entropy", "actor/ppo_kl", "actor/pg_clipfrac"), values,
    )) | {"actor/grad_norm": grad_norm, "actor/generated_tokens": denominator}


def _make_worker_class() -> type:
    """Imported only inside the GPU entrypoint, and serialized by Ray."""
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    class OnlineActorRolloutRefWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            result = super().init_model()
            if self._is_actor:
                self.actor.update_policy = MethodType(_online_update_policy, self.actor)
            return result

        @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
        def generate_sequences(self, prompts):
            if not hasattr(self.rollout, "sampling_params"):
                raise RuntimeError("Expected veRL 0.6.1 synchronous vLLM rollout")
            seeds = prompts.non_tensor_batch["sample_seeds"].tolist()
            expected = prompts.non_tensor_batch.get("expected_prompt_ids")
            with _per_request_sampling(self.rollout, seeds, None if expected is None else expected.tolist()):
                result = super().generate_sequences(prompts)
                eos = self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id
                result.meta_info["eos_token_ids"] = eos if isinstance(eos, list) else [eos]
                return result

    return OnlineActorRolloutRefWorker


def compose_verl_config(config: dict[str, Any]) -> Any:
    """Compose upstream defaults instead of depending on any local veRL fork."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    import verl

    actual = importlib.metadata.version("verl")
    if actual != VERL_VERSION:
        raise RuntimeError(f"This backend requires upstream verl=={VERL_VERSION}; installed {actual}")
    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(config_name="ppo_trainer")
    with open_dict(base):
        merged = OmegaConf.merge(base, OmegaConf.create(config["verl"]))
    with open_dict(merged):
        # No offline QA dataset or veRL-default conversation loop is used.
        merged.algorithm.adv_estimator = "grpo"
        merged.algorithm.use_kl_in_reward = False
        merged.actor_rollout_ref.rollout.n = 1
        merged.actor_rollout_ref.rollout.mode = "sync"
        merged.actor_rollout_ref.actor.optim.total_training_steps = int(config["max_updates"])
        merged.trainer.total_training_steps = int(config["max_updates"])
        merged.trainer.default_local_dir = str(Path(config["output_dir"]).resolve())
        merged.trainer.logger = ["console"]
    OmegaConf.resolve(merged)
    return merged


class VerlPolicy:
    """Batch real Harness calls, preserving each exact prompt/response/score."""

    def __init__(self, worker: Any, tokenizer: Any, config: dict[str, Any], *,
                 processor: Any = None, policy_version: str = "update-0"):
        self.worker = worker
        self.tokenizer = tokenizer
        self.processor = processor
        self.use_screenshot = config.get("harness", {}).get("use_screenshot", False)
        if self.use_screenshot and processor is None:
            raise ValueError("use_screenshot=true requires a Qwen3-VL model and processor")
        self.config = config
        self.policy_version = policy_version
        self.sample_counter = 0
        self.seed = int(config.get("seed", 0))
        self.chat_template_kwargs = config.get("chat_template_kwargs", {})
        rollout = config["verl"]["actor_rollout_ref"]["rollout"]
        self.max_prompt_tokens = int(rollout["prompt_length"])
        self.max_response_tokens = int(rollout["response_length"])
        self.temperature = float(rollout["temperature"])

    def generate(self, messages: list[dict[str, Any]]) -> Generation:
        request = GenerationRequest(
            f"serial-{self.sample_counter}", messages,
            (self.seed + self.sample_counter) % (2**31 - 1), self.policy_version,
        )
        result = self.generate_batch([request])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def _encode(self, request: GenerationRequest) -> EncodedPrompt:
        if request.policy_version != self.policy_version:
            raise InfrastructureError("Refusing to generate with a different policy version")
        if has_images(request.messages) != self.use_screenshot:
            raise InfrastructureError("Prompt images disagree with harness.use_screenshot; refusing silent mode change")
        encoded = encode_prompt(
            request.messages, self.tokenizer, self.processor, chat_template_kwargs=self.chat_template_kwargs,
        )
        if self.use_screenshot and len(encoded.images) != 1:
            raise InfrastructureError("Expected exactly one current screenshot per model call")
        if not encoded.token_ids or len(encoded.token_ids) > self.max_prompt_tokens:
            raise ContextLimitError(f"Prompt length {len(encoded.token_ids)} exceeds budget {self.max_prompt_tokens}")
        return encoded

    def generate_batch(self, requests: list[GenerationRequest]) -> list[Generation | Exception]:
        # An oversized context affects its own trajectory/group, not unrelated
        # tasks that happened to share the same inference batch.
        results: dict[int, Generation | Exception] = {}
        valid: list[tuple[int, GenerationRequest, EncodedPrompt]] = []
        for index, request in enumerate(requests):
            try:
                valid.append((index, request, self._encode(request)))
            except (ContextLimitError, InfrastructureError) as exc:
                results[index] = exc
        if not valid:
            return [results[index] for index in range(len(requests))]
        import numpy as np
        import torch
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        batch_size = len(valid)
        tensors, non_tensors = pack_prompts([e for _, _, e in valid], self.tokenizer.pad_token_id)
        input_ids = tensors["input_ids"]
        self.sample_counter += batch_size
        request = DataProto.from_dict(
            tensors=tensors,
            non_tensors={**non_tensors,
                         "sample_seeds": np.array([item.sample_seed for _, item, _ in valid], dtype=np.int64),
                         "request_ids": np.array([item.request_id for _, item, _ in valid], dtype=object)},
            meta_info={"do_sample": True, "temperature": self.temperature},
        )
        # Padding here is *only for generation*. Duplicated trajectories are
        # discarded and never enter the environment or a training loss.
        padded, padding = pad_dataproto_to_divisor(request, self.worker.world_size)
        generated_padded = self.worker.generate_sequences(padded)
        if self.processor is not None and "multi_modal_inputs" not in generated_padded.non_tensor_batch:
            raise InfrastructureError("Rollout lost image inputs before old-policy scoring")
        # Score before the next policy update, using the exact tokenized input.
        old_padded = self.worker.compute_log_prob(generated_padded)
        generated = unpad_dataproto(generated_padded, padding)
        old = unpad_dataproto(old_padded, padding)
        if (len(generated) != batch_size or len(old) != batch_size
                or list(generated.non_tensor_batch["request_ids"]) != [item.request_id for _, item, _ in valid]
                or list(generated.non_tensor_batch["sample_seeds"]) != [item.sample_seed for _, item, _ in valid]
                or not torch.equal(generated.batch["prompts"], input_ids)):
            raise InfrastructureError("veRL changed the inference batch order or exact prompt tokens")
        response_width = generated.batch["responses"].shape[-1]
        eos = self.tokenizer.eos_token_id
        eos_ids = generated.meta_info.get("eos_token_ids", eos if isinstance(eos, list) else [eos])
        for row, (index, item, encoded) in enumerate(valid):
            try:
                length = int(generated.batch["attention_mask"][row, -response_width:].sum())
                response_ids = generated.batch["responses"][row, :length].tolist()
                if not response_ids or length > self.max_response_tokens:
                    raise InfrastructureError("Rollout returned an invalid generated-token count")
                log_probs = old.batch["old_log_probs"][row, :length].tolist()
                if len(log_probs) != length or not all(math.isfinite(value) and value <= 1e-6 for value in log_probs):
                    raise InfrastructureError("Rollout returned invalid old-policy log probabilities")
                entropy = float(old.batch["entropys"][row, :length].mean())
                if not math.isfinite(entropy):
                    raise InfrastructureError("Rollout returned non-finite entropy")
                reason = "stop" if length < self.max_response_tokens or response_ids[-1] in eos_ids else "length"
                metadata = {"request_id": item.request_id, "sample_seed": item.sample_seed,
                            "temperature": self.temperature, "log_prob_source": "unchanged_actor_recompute",
                            "mean_entropy": entropy}
                if encoded.provenance:
                    metadata["input_encoding"] = encoded.provenance
                if "rollout_log_probs" in generated.batch:
                    metadata["rollout_log_probs"] = generated.batch["rollout_log_probs"][row, :length].tolist()
                results[index] = Generation(
                    text=self.tokenizer.decode(response_ids, skip_special_tokens=True),
                    prompt_token_ids=encoded.token_ids, response_token_ids=response_ids,
                    old_log_probs=log_probs, policy_version=item.policy_version,
                    finish_reason=reason, metadata=metadata,
                )
            except InfrastructureError as exc:
                results[index] = exc
        return [results[index] for index in range(len(requests))]


class VerlBackend:
    """Policy generation, real GRPO actor updates, and sharded checkpoint I/O."""

    def __init__(self, config: dict[str, Any]):
        import ray
        import torch
        from omegaconf import OmegaConf, open_dict
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
        from verl.trainer.ppo.utils import Role
        from verl.utils import hf_tokenizer
        from transformers import AutoConfig, AutoProcessor

        self.config = deepcopy(config)
        self.verl_config = compose_verl_config(config)
        self.started_ray = not ray.is_initialized()
        if not torch.cuda.is_available():
            raise RuntimeError("The veRL backend requires CUDA GPUs; use --check-config on a laptop")
        if self.started_ray:
            runtime = get_ppo_ray_runtime_env()
            runtime = OmegaConf.to_container(runtime) if OmegaConf.is_config(runtime) else dict(runtime)
            source_root = str(Path(__file__).resolve().parents[2])
            runtime.setdefault("env_vars", {})["PYTHONPATH"] = os.pathsep.join(
                value for value in (source_root, os.environ.get("PYTHONPATH", "")) if value
            )
            ray.init(runtime_env=runtime)
        torch.manual_seed(int(config.get("seed", 0)))
        worker_cls = ray.remote(_make_worker_class())
        roles = {Role.ActorRollout: worker_cls}
        mappings = {Role.ActorRollout: "actors"}
        if self.verl_config.actor_rollout_ref.actor.use_kl_loss:
            roles[Role.RefPolicy] = worker_cls
            mappings[Role.RefPolicy] = "actors"
        resource_pool = ResourcePoolManager(
            resource_pool_spec={"actors": [int(self.verl_config.trainer.n_gpus_per_node)] * int(self.verl_config.trainer.nnodes)},
            mapping=mappings,
        )

        class OnlineWorkerOwner(RayPPOTrainer):
            def _create_dataloader(owner, *args, **kwargs):
                owner.total_training_steps = int(config["max_updates"])

        tokenizer = hf_tokenizer(
            self.verl_config.actor_rollout_ref.model.path,
            trust_remote_code=self.verl_config.actor_rollout_ref.model.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model_path = self.verl_config.actor_rollout_ref.model.path
        trust = self.verl_config.actor_rollout_ref.model.trust_remote_code
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust)
        processor = None
        if model_config.model_type == "qwen3_vl":
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust)
            # One image per turn, no video. Avoid allocating a video/image
            # profiling budget for modalities this experiment never supplies.
            with open_dict(self.verl_config):
                rollout = self.verl_config.actor_rollout_ref.rollout
                rollout.limit_images = None
                rollout.engine_kwargs.setdefault("vllm", {})["limit_mm_per_prompt"] = {
                    "image": int(config.get("harness", {}).get("use_screenshot", False)), "video": 0,
                }
        elif config.get("harness", {}).get("use_screenshot", False) or hasattr(model_config, "vision_config"):
            raise ValueError("The screenshot RL path currently supports Qwen3-VL dense models only")
        # Even a text-only ablation on Qwen3-VL needs its four-track MRoPE inputs.
        self.owner = OnlineWorkerOwner(
            config=self.verl_config, tokenizer=tokenizer, processor=processor, role_worker_mapping=roles,
            resource_pool_manager=resource_pool, ray_worker_group_cls=RayWorkerGroup,
        )
        self.owner.init_workers()
        self.worker = self.owner.actor_rollout_wg
        # Upstream uses this also to offload after initialization.
        self.worker.load_checkpoint(None)
        self.policy = VerlPolicy(self.worker, tokenizer, config, processor=processor)
        self.update_count = 0

    def update(self, rows: list[TrainingRow]) -> dict[str, float]:
        if any(row.policy_version != self.policy.policy_version for row in rows):
            raise ValueError("Refusing an update with stale rollout rows")
        batch = tensor_batch(
            rows, pad_token_id=self.policy.tokenizer.pad_token_id,
            world_size=self.worker.world_size,
            max_prompt_tokens=self.policy.max_prompt_tokens,
            max_response_tokens=self.policy.max_response_tokens,
            temperature=self.policy.temperature,
            tokenizer=self.policy.tokenizer, processor=self.policy.processor,
        )
        if self.verl_config.actor_rollout_ref.actor.use_kl_loss:
            ref_worker = self.worker if self.owner.ref_in_actor else self.owner.ref_policy_wg
            reference = ref_worker.compute_ref_log_prob(batch)
            batch.batch["ref_log_prob"] = reference.batch["ref_log_prob"]
        output = self.worker.update_actor(batch)
        # Returned metrics can be scalars or lists collected from Ray ranks.
        metrics: dict[str, float] = {}
        for name, value in output.meta_info["metrics"].items():
            if isinstance(value, (tuple, list)):
                value = sum(float(x) for x in value) / len(value)
            metrics[name] = float(value)
        if not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError("Non-finite actor update; stop and inspect the last checkpoint")
        self.update_count += 1
        self.policy.policy_version = f"update-{self.update_count}"
        # The next generate_sequences enters upstream rollout_mode(), which
        # transfers the updated actor/LoRA weights to vLLM before generation.
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {"policy_version": self.policy.policy_version,
                "sample_counter": self.policy.sample_counter, "update_count": self.update_count}

    def save_checkpoint(self, directory: str | Path) -> None:
        path = Path(directory).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.worker.save_checkpoint(str(path / "actor"), None, self.update_count, max_ckpt_to_keep=None)

    def load_checkpoint(self, directory: str | Path, state: dict[str, Any]) -> None:
        actor = Path(directory).resolve() / "actor"
        if not actor.is_dir():
            raise ValueError(f"Missing veRL actor checkpoint: {actor}")
        self.worker.load_checkpoint(str(actor), del_local_after_load=False)
        self.update_count = int(state["update_count"])
        self.policy.policy_version = str(state["policy_version"])
        self.policy.sample_counter = int(state["sample_counter"])
        if self.policy.policy_version != f"update-{self.update_count}":
            raise ValueError("Checkpoint policy version does not match update counter")

    def close(self) -> None:
        if self.started_ray:
            import ray
            ray.shutdown()
