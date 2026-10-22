"""CPU tests for online RL bookkeeping and loss math, not GPU performance claims."""

from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from open_apps.agentic_rl.train import (
    TaskCursor, load_config, run_online_loop, save_training_checkpoint, validate_config,
)
from open_apps.agentic_rl.training_data import prepare_training_batch, tensor_batch
from open_apps.agentic_rl.types import AttemptRecord, ContextLimitError, Episode, Generation, GenerationRequest, InfrastructureError, TaskSpec
from open_apps.agentic_rl.verl_backend import VerlPolicy, _online_update_policy, _per_request_sampling


ROOT = Path(__file__).resolve().parents[2]


def episode(index, reward, attempts=1, *, version="update-0", group="g"):
    calls = [AttemptRecord(
        step_index=0, attempt_index=i, messages=[{"role": "user", "content": f"prompt-{i}"}],
        generation=Generation(f"raw-{i}", [10 + i, 20], [30 + i, 40 + i], [-0.2, -0.3], version),
        accepted=i == attempts - 1, error_type="parse" if i < attempts - 1 else None,
    ) for i in range(attempts)]
    return Episode(f"{group}:{index}", group, "task", 7, policy_version=version,
                   initial_state_hash="same-state", attempts=calls, reward=reward,
                   success=reward == 1, terminal_reason="success" if reward == 1 else "max_steps")


def test_advantages_are_computed_per_episode_before_retry_flattening():
    episodes = [episode(0, 0, attempts=3), episode(1, 1)]
    packed = prepare_training_batch(episodes, group_size=2, policy_version="update-0")
    assert [r.advantage for r in packed.rows] == [-0.5, -0.5, -0.5, 0.5]
    assert [r.response_token_ids for r in packed.rows[:3]] == [[30, 40], [31, 41], [32, 42]]
    assert packed.rows[1].prompt_token_ids == [11, 20]
    assert packed.metrics["train/model_calls"] == 4


def test_infrastructure_failure_drops_whole_group_not_policy_failure():
    episodes = [episode(0, 0), episode(1, 1)]
    episodes[0].infrastructure_error = "browser unavailable"
    packed = prepare_training_batch(episodes, group_size=2, policy_version="update-0")
    assert not packed.rows
    assert packed.metrics["train/skipped_groups"] == 1


@pytest.mark.parametrize("mutation", [
    lambda ep: setattr(ep, "initial_state_hash", "different-state"),
    lambda ep: setattr(ep, "policy_version", "update-1"),
    lambda ep: setattr(ep.attempts[0].generation, "old_log_probs", [0.2, -0.2]),
    lambda ep: setattr(ep.attempts[0].generation, "prompt_token_ids", [True, 20]),
    lambda ep: setattr(ep.attempts[0].generation, "old_log_probs", [-0.2]),
    lambda ep: ep.attempts[0].generation.metadata.update(synthetic_token_ids=True),
    lambda ep: ep.attempts[0].generation.metadata.update(trainable=False),
    lambda ep: ep.attempts[0].generation.metadata.update(backend="scripted"),
])
def test_training_boundary_rejects_corrupt_or_synthetic_data(mutation):
    episodes = [episode(0, 0), episode(1, 1)]
    mutation(episodes[0])
    with pytest.raises(ValueError):
        prepare_training_batch(episodes, group_size=2, policy_version="update-0")


def test_zero_variance_is_finite_and_no_singleton_group_is_accepted():
    packed = prepare_training_batch([episode(0, 0), episode(1, 0)], group_size=2,
                                    policy_version="update-0", normalize_std=True)
    assert all(row.advantage == 0 for row in packed.rows)
    with pytest.raises(ValueError, match="Incomplete"):
        prepare_training_batch([episode(0, 0)], group_size=2, policy_version="update-0")


class FakeDataProto:
    def __init__(self, tensors=None, non_tensors=None, meta_info=None):
        self.batch = tensors or {}
        self.non_tensor_batch = non_tensors or {}
        self.meta_info = meta_info or {}

    @classmethod
    def from_dict(cls, tensors=None, non_tensors=None, meta_info=None, **kwargs):
        return cls(tensors, non_tensors, meta_info)

    def __len__(self):
        return len(next(iter(self.batch.values())))

    def split(self, size):
        return [FakeDataProto({k: v[i:i + size] for k, v in self.batch.items()},
                              {k: v[i:i + size] for k, v in self.non_tensor_batch.items()},
                              dict(self.meta_info)) for i in range(0, len(self), size)]

    def to(self, device):
        self.batch = {k: v.to(device) for k, v in self.batch.items()}
        return self


@pytest.fixture
def fake_verl(monkeypatch):
    module = ModuleType("verl")
    module.DataProto = FakeDataProto
    monkeypatch.setitem(sys.modules, "verl", module)
    return module


def test_tensor_pack_preserves_every_generated_token_and_masks_dp_padding(fake_verl):
    torch = pytest.importorskip("torch")
    episodes = [episode(0, 0, 2), episode(1, 1)]
    rows = prepare_training_batch(episodes, group_size=2, policy_version="update-0").rows
    rows[1].prompt_token_ids = [91]
    rows[2].response_token_ids = [92]
    rows[2].old_log_probs = [-0.8]
    data = tensor_batch(rows, pad_token_id=0, world_size=2, max_prompt_tokens=10,
                        max_response_tokens=10, temperature=0.7)
    assert data.batch["response_mask"].tolist() == [[1, 1], [1, 1], [1, 0], [0, 0]]
    assert data.batch["prompts"][1].tolist() == [0, 91]
    assert data.batch["input_ids"][1].tolist() == [0, 91, 31, 41]
    assert data.batch["position_ids"][1].tolist() == [0, 0, 1, 2]
    assert data.meta_info["global_response_tokens"] == 5
    assert data.meta_info["temperature"] == 0.7
    assert torch.equal(data.batch["advantages"][3], torch.zeros(2))
    with pytest.raises(ValueError, match="truncation"):
        tensor_batch(rows, pad_token_id=0, world_size=2, max_prompt_tokens=1,
                     max_response_tokens=10, temperature=1)


def test_real_torch_gradient_is_invariant_to_zero_loss_padding(fake_verl, monkeypatch):
    torch = pytest.importorskip("torch")
    core = ModuleType("verl.trainer.ppo.core_algos")

    def clipped_objective(*, old_log_prob, log_prob, advantages, response_mask, **kwargs):
        ratio = (log_prob - old_log_prob).exp()
        token_loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
        return (token_loss * response_mask).sum() / response_mask.shape[-1], {
            "actor/ppo_kl": 0.0, "actor/pg_clipfrac": 0.0,
        }

    core.compute_policy_loss_vanilla = clipped_objective
    core.kl_penalty = lambda current, reference, kind: (current - reference).square()
    device = ModuleType("verl.utils.device")
    device.get_device_id = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", core)
    monkeypatch.setitem(sys.modules, "verl.utils.device", device)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    class Actor:
        def __init__(self):
            self.actor_module = torch.nn.Linear(1, 1, bias=False)
            self.parameter = self.actor_module.weight
            with torch.no_grad():
                self.parameter.fill_(0.2)
            self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
            self.config = SimpleNamespace(use_kl_loss=False, entropy_coeff=0.0)

        def _forward_micro_batch(self, inputs, **kwargs):
            p = self.parameter.reshape(())
            log_prob = torch.nn.functional.logsigmoid(p).expand_as(inputs["old_log_probs"])
            entropy = (-(p.sigmoid() * torch.nn.functional.logsigmoid(p)
                         + (-p).sigmoid() * torch.nn.functional.logsigmoid(-p))).expand_as(log_prob)
            return entropy, log_prob

        def _optimizer_step(self):
            norm = self.parameter.grad.norm().clone()
            self.actor_optimizer.step()
            return norm

    episodes = [episode(0, 0, 2), episode(1, 1)]
    rows = prepare_training_batch(episodes, group_size=2, policy_version="update-0").rows
    old = float(torch.nn.functional.logsigmoid(torch.tensor(0.2)))
    for row in rows:
        row.old_log_probs = [old] * len(row.response_token_ids)
    parameters = []
    for divisor in (1, 4, 8):
        data = tensor_batch(rows, pad_token_id=0, world_size=divisor, max_prompt_tokens=10,
                            max_response_tokens=10, temperature=1)
        actor = Actor()
        metrics = _online_update_policy(actor, data)
        assert metrics["actor/generated_tokens"] == 6
        assert metrics["actor/entropy"] > 0
        parameters.append(float(actor.parameter))
    # Independent analytic expectation for globally averaged Bernoulli policy
    # gradient: dL/dp = -mean(A) * (1-sigmoid(p)).
    expected = 0.2 + 0.1 * (-1 / 6) * (1 - torch.sigmoid(torch.tensor(0.2)).item())
    assert parameters == pytest.approx([expected] * 3, abs=1e-7)


def test_generation_uses_worker_tokens_and_actual_old_logprobs(fake_verl, monkeypatch):
    torch = pytest.importorskip("torch")
    protocol = ModuleType("verl.protocol")

    def pad(data, divisor):
        import numpy as np
        assert divisor == 2
        return FakeDataProto({k: v.repeat((2,) + (1,) * (v.ndim - 1)) for k, v in data.batch.items()},
                             {k: np.concatenate([v, v]) for k, v in data.non_tensor_batch.items()},
                             meta_info=data.meta_info), 1

    def unpad(data, count):
        return FakeDataProto({k: v[:-count] for k, v in data.batch.items()},
                             {k: v[:-count] for k, v in data.non_tensor_batch.items()}, meta_info=data.meta_info)

    protocol.pad_dataproto_to_divisor = pad
    protocol.unpad_dataproto = unpad
    monkeypatch.setitem(sys.modules, "verl.protocol", protocol)

    class Worker:
        world_size = 2
        def generate_sequences(self, request):
            assert request.non_tensor_batch["sample_seeds"].tolist() == [17, 17]
            assert request.meta_info["temperature"] == 1.0
            return FakeDataProto({"responses": torch.tensor([[50, 77], [60, 77]]),
                                  "prompts": request.batch["input_ids"],
                                  "attention_mask": torch.ones((2, 5), dtype=torch.long)},
                                 non_tensors=request.non_tensor_batch,
                                 meta_info={"eos_token_ids": [77]})

        def compute_log_prob(self, generated):
            return FakeDataProto({"old_log_probs": torch.tensor([[-0.7, -0.8], [-0.9, -1.0]]),
                                  "entropys": torch.ones((2, 2))})

    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [11, 12, 13],
                                eos_token_id=2, pad_token_id=0, decode=lambda ids, **kwargs: "actual-sample")
    config = load_config(ROOT / "config/agentic_rl/verl.yaml")
    config["verl"]["actor_rollout_ref"]["rollout"]["response_length"] = 2
    policy = VerlPolicy(Worker(), tokenizer, config)
    generation = policy.generate([{"role": "user", "content": "actual prompt"}])
    assert generation.prompt_token_ids == [11, 12, 13]
    assert generation.response_token_ids == [50, 77]
    assert generation.old_log_probs == pytest.approx([-0.7, -0.8])
    assert generation.finish_reason == "stop"  # worker's EOS, not tokenizer's.
    assert policy.sample_counter == 1


@pytest.mark.parametrize("reorder", [False, True])
def test_batch_generation_preserves_variable_tokens_seeds_scores_and_dp_padding(fake_verl, monkeypatch, reorder):
    torch = pytest.importorskip("torch")
    import numpy as np
    protocol = ModuleType("verl.protocol")

    def pad(data, divisor):
        assert divisor == 4 and len(data) == 3
        return FakeDataProto({k: torch.cat([v, v[:1]]) for k, v in data.batch.items()},
                             {k: np.concatenate([v, v[:1]]) for k, v in data.non_tensor_batch.items()},
                             data.meta_info), 1

    def unpad(data, count):
        assert count == 1
        return FakeDataProto({k: v[:-1] for k, v in data.batch.items()},
                             {k: v[:-1] for k, v in data.non_tensor_batch.items()}, data.meta_info)

    protocol.pad_dataproto_to_divisor = pad
    protocol.unpad_dataproto = unpad
    monkeypatch.setitem(sys.modules, "verl.protocol", protocol)

    class Worker:
        world_size = 4
        calls = []

        def generate_sequences(self, request):
            self.calls.append("generate")
            assert request.batch["input_ids"].tolist() == [[0, 0, 11, 12], [21, 22, 23, 24], [0, 0, 0, 31], [0, 0, 11, 12]]
            assert request.batch["position_ids"].tolist() == [[0, 0, 0, 1], [0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 1]]
            assert request.non_tensor_batch["sample_seeds"].tolist() == [101, 102, 103, 101]
            assert list(request.non_tensor_batch["raw_prompt_ids"]) == [[11, 12], [21, 22, 23, 24], [31], [11, 12]]
            non_tensors = deepcopy(request.non_tensor_batch)
            if reorder:
                non_tensors["request_ids"] = non_tensors["request_ids"][::-1]
            return FakeDataProto({
                "prompts": request.batch["input_ids"],
                "responses": torch.tensor([[50, 77, 0], [51, 52, 77], [77, 0, 0], [50, 77, 0]]),
                "attention_mask": torch.cat([request.batch["attention_mask"],
                                             torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]])], dim=-1),
            }, non_tensors, {"eos_token_ids": [77]})

        def compute_log_prob(self, generated):
            self.calls.append("score")
            return FakeDataProto({
                "old_log_probs": torch.tensor([[-0.2, -0.3, float("nan")], [-0.4, -0.5, -0.6],
                                                [-0.7, float("nan"), float("nan")], [-0.2, -0.3, float("nan")]]),
                "entropys": torch.ones((4, 3)),
            })

    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: json.loads(messages[0]["content"]),
                                pad_token_id=0, eos_token_id=2, decode=lambda ids, **kwargs: str(ids))
    config = load_config(ROOT / "config/agentic_rl/verl.yaml")
    config["verl"]["actor_rollout_ref"]["rollout"].update(prompt_length=4, response_length=3)
    worker = Worker()
    policy = VerlPolicy(worker, tokenizer, config)
    requests = [GenerationRequest(name, [{"role": "user", "content": json.dumps(ids)}], seed, "update-0")
                for name, ids, seed in [("a", [11, 12], 101), ("too-long", list(range(5)), 999),
                                       ("b", [21, 22, 23, 24], 102), ("c", [31], 103)]]
    if reorder:
        with pytest.raises(InfrastructureError, match="order"):
            policy.generate_batch(requests)
        return
    result = policy.generate_batch(requests)
    assert isinstance(result[1], ContextLimitError)
    good = [result[i] for i in (0, 2, 3)]
    assert [g.prompt_token_ids for g in good] == [[11, 12], [21, 22, 23, 24], [31]]
    assert [g.response_token_ids for g in good] == [[50, 77], [51, 52, 77], [77]]
    assert good[0].old_log_probs == pytest.approx([-0.2, -0.3])
    assert good[1].old_log_probs == pytest.approx([-0.4, -0.5, -0.6])
    assert good[2].old_log_probs == pytest.approx([-0.7])
    assert [g.metadata["request_id"] for g in good] == ["a", "b", "c"]
    assert [g.metadata["sample_seed"] for g in good] == [101, 102, 103]
    assert all(g.finish_reason == "stop" for g in good)
    assert policy.sample_counter == 3 and worker.calls == ["generate", "score"]


@pytest.mark.parametrize("fail", [False, True])
def test_worker_seed_adapter_uses_one_sampling_params_per_prompt_and_restores_engine(fail):
    captured = []

    class Engine:
        def generate(self, **kwargs):
            captured.append(kwargs)
            if fail:
                raise RuntimeError("GPU failure")
            return "generated"

    engine = Engine()
    original = engine.generate
    template = SimpleNamespace(seed=None, temperature=0.7)
    rollout = SimpleNamespace(inference_engine=engine)

    def invoke():
        with _per_request_sampling(rollout, [101, 102, 103]):
            return engine.generate(prompts=[{}, {}, {}], sampling_params=template, lora_request=["adapter"] * 3)

    if fail:
        with pytest.raises(RuntimeError, match="GPU failure"):
            invoke()
    else:
        assert invoke() == "generated"
    assert engine.generate == original
    parameters = captured[0]["sampling_params"]
    assert [parameter.seed for parameter in parameters] == [101, 102, 103]
    assert all(parameter.temperature == 0.7 for parameter in parameters)
    assert len({id(parameter) for parameter in parameters}) == 3
    assert template.seed is None
    assert captured[0]["lora_request"] == ["adapter"] * 3


class FakeBackend:
    def __init__(self):
        self.policy = SimpleNamespace(policy_version="update-0", sample_counter=0)
        self.update_count = 0
        self.updated_versions = []

    def update(self, rows):
        assert {r.policy_version for r in rows} == {self.policy.policy_version}
        self.updated_versions.append(self.policy.policy_version)
        self.update_count += 1
        self.policy.policy_version = f"update-{self.update_count}"
        return {"actor/test_metric": 1.0}

    def state_dict(self):
        return {"update_count": self.update_count, "policy_version": self.policy.policy_version,
                "sample_counter": self.policy.sample_counter}

    def save_checkpoint(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mock-model.json").write_text(json.dumps(self.state_dict()))

    def load_checkpoint(self, directory, state):
        assert json.loads((Path(directory) / "mock-model.json").read_text()) == state
        self.update_count = state["update_count"]
        self.policy.policy_version = state["policy_version"]
        self.policy.sample_counter = state["sample_counter"]


def fake_sampler(task, policy, env_factory, *, group_size, group_id, **kwargs):
    policy.sample_counter += group_size
    return [episode(i, float(i == group_size - 1), version=policy.policy_version, group=group_id)
            for i in range(group_size)]


def loop_config(tmp_path):
    return {"output_dir": str(tmp_path / "run"), "epochs": 3, "max_updates": 2,
            "group_size": 2, "tasks_per_update": 2, "seed": 17,
            "checkpoint_every": 1, "eval_every": 0, "grpo": {"normalize_std": False}}


def test_online_loop_updates_before_next_rollout_and_uses_distinct_cross_epoch_groups(tmp_path):
    config = loop_config(tmp_path)
    backend = FakeBackend()
    result = run_online_loop(config, [TaskSpec("t", "goal", {})], backend, lambda: None,
                             sample_group_fn=fake_sampler)
    assert result["updates"] == 2
    assert backend.updated_versions == ["update-0", "update-1"]
    lines = (Path(config["output_dir"]) / "episodes/batch-000001.jsonl").read_text().splitlines()
    assert len({json.loads(line)["group_id"] for line in lines}) == 2
    assert backend.policy.sample_counter == 6
    with pytest.raises(FileExistsError, match="fresh"):
        run_online_loop(config, [TaskSpec("t", "goal", {})], FakeBackend(), lambda: None,
                        sample_group_fn=fake_sampler)


def test_checkpoint_restores_policy_sampling_counter_and_cursor_before_new_rollout(tmp_path):
    config = loop_config(tmp_path)
    config["tasks_per_update"] = 1
    tasks = [TaskSpec("t", "goal", {})]
    backend = FakeBackend()
    backend.update([SimpleNamespace(policy_version="update-0")])
    backend.policy.sample_counter = 12
    cursor = TaskCursor(1, 3, 17)
    cursor.take(1)
    checkpoint = save_training_checkpoint(backend, config, tasks, cursor, batch_count=1)
    restored = FakeBackend()
    resumed_config = config | {"output_dir": str(tmp_path / "resumed"), "resume_from": str(checkpoint)}
    result = run_online_loop(resumed_config, tasks, restored, lambda: None, sample_group_fn=fake_sampler)
    assert result["updates"] == 2
    assert restored.updated_versions == ["update-1"]
    assert restored.policy.sample_counter == 14
    saved = json.loads((Path(result["checkpoint"]) / "training_state.json").read_text())
    assert saved["cursor"]["epoch"] == 1
    assert saved["batch_count"] == 2


def test_cursor_rng_round_trips_through_json():
    cursor = TaskCursor(5, 4, 77)
    cursor.take(7)
    other = TaskCursor.from_state(json.loads(json.dumps(cursor.state_dict())))
    assert cursor.take(12) == other.take(12)


def test_resume_rejects_changed_harness_source_fingerprint(tmp_path, monkeypatch):
    config = loop_config(tmp_path)
    tasks = [TaskSpec("t", "goal", {})]
    checkpoint = save_training_checkpoint(FakeBackend(), config, tasks, TaskCursor(1, 3, 17), 0)
    monkeypatch.setattr("open_apps.agentic_rl.train._harness_fingerprint", lambda config: "changed-source")
    config = config | {"resume_from": str(checkpoint), "output_dir": str(tmp_path / "resumed")}
    with pytest.raises(ValueError, match="fingerprints"):
        run_online_loop(config, tasks, FakeBackend(), lambda: None, sample_group_fn=fake_sampler)


def test_zero_signal_run_saves_explicit_update_zero_checkpoint(tmp_path):
    config = loop_config(tmp_path) | {"epochs": 1, "tasks_per_update": 1}
    def sampler(*args, **kwargs):
        result = fake_sampler(*args, **kwargs)
        for item in result:
            item.reward = 0
        return result
    result = run_online_loop(config, [TaskSpec("t", "goal", {})], FakeBackend(), lambda: None,
                             sample_group_fn=sampler)
    assert result["updates"] == 0
    assert Path(result["checkpoint"], "training_state.json").exists()


@pytest.mark.parametrize("key,value", [("top_p", 0.9), ("min_p", 0.1), ("min_tokens", 2),
                                       ("presence_penalty", 0.1), ("frequency_penalty", 0.1),
                                       ("repetition_penalty", 1.0)])
def test_sampling_distribution_changes_are_rejected(key, value):
    config = load_config(ROOT / "config/agentic_rl/verl.yaml")
    config["verl"]["actor_rollout_ref"]["rollout"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_config_validation_checks_manifest_semantics(tmp_path):
    from open_apps.agentic_rl.tasks import save_manifest
    from open_apps.agentic_rl.testing import smoke_task
    config = load_config(ROOT / "config/agentic_rl/verl.yaml")
    task = smoke_task()
    save_manifest([task], tmp_path / "train.jsonl")
    task.task_id = "different-name-same-family"
    save_manifest([task], tmp_path / "dev.jsonl")
    config.update(manifest=str(tmp_path / "train.jsonl"), eval_manifest=str(tmp_path / "dev.jsonl"))
    with pytest.raises(ValueError, match="overlap"):
        validate_config(config, check_paths=True)
