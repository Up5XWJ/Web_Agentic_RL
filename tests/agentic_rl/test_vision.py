"""CPU contracts: images reach generation, scoring and actual gradient updates.

These use small deterministic processor/worker doubles, not a GPU performance
claim. An optional real Qwen3-VL processor contract lives at the bottom.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from open_apps.agentic_rl import vision
from open_apps.agentic_rl.train import load_config, validate_config
from open_apps.agentic_rl.training_data import prepare_training_batch, tensor_batch
from open_apps.agentic_rl.types import ContextLimitError, Episode, GenerationRequest, InfrastructureError
from open_apps.agentic_rl.verl_backend import VerlPolicy, VerlBackend, _online_update_policy, _per_request_sampling
from tests.agentic_rl.test_training import FakeDataProto, episode, fake_verl


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [999 if c == "@" else ord(c) + 3 for c in text]

    def decode(self, ids, **kwargs):
        return "generated"


class Qwen3VLProcessor:
    """Tiny deterministic image expansion; distinct pixels produce distinct inputs."""

    def __init__(self):
        self.tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        return "|".join("".join(p["text"] if p["type"] == "text" else "@" for p in m["content"])
                        for m in messages)

    def __call__(self, *, text, images, **kwargs):
        raw = self.tokenizer.encode(text[0])
        ids = [i for value in raw for i in ([999] * 4 if value == 999 else [value])]
        result = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        if images:
            result.update(pixel_values=torch.tensor([np.asarray(i).mean() / 255 for i in images], dtype=torch.float32).view(-1, 1),
                          image_grid_thw=torch.tensor([[1, 4, 4]] * len(images)))
        return result


@pytest.fixture
def processor(monkeypatch):
    # Make the spatial channels visibly different from ordinary text positions.
    monkeypatch.setattr(vision, "_qwen_positions", lambda processor, ids, inputs:
                        torch.stack([torch.arange(len(ids)) // i for i in (1, 2, 3, 4)]))
    return Qwen3VLProcessor()


def messages(value=100):
    return [{"role": "user", "content": [
        {"type": "text", "text": "Inspect"},
        {"type": "image_url", "image_url": {"url": vision.screenshot_data_url(np.full((16, 32, 3), value, np.uint8))}},
    ]}]


def config():
    result = load_config(Path(__file__).resolve().parents[2] / "config/agentic_rl/verl.yaml")
    result["harness"]["use_screenshot"] = True
    return result


def visual_rows(processor):
    episodes = [episode(0, 0), episode(1, 1)]
    for i, ep in enumerate(episodes):
        attempt = ep.attempts[0]
        attempt.messages = messages(30 + 160 * i)
        item = vision.encode_prompt(attempt.messages, processor.tokenizer, processor)
        attempt.generation.prompt_token_ids = item.token_ids
        attempt.generation.metadata["input_encoding"] = item.provenance
    # Prove the saved JSON (not an in-memory PIL/Tensor reference) is sufficient.
    restored = [Episode.from_dict(json.loads(json.dumps(ep.to_dict()))) for ep in episodes]
    return prepare_training_batch(restored, group_size=2, policy_version="update-0").rows


def pack_rows(rows, processor, world=1):
    return tensor_batch(rows, pad_token_id=0, world_size=world, max_prompt_tokens=1000,
                        max_response_tokens=10, temperature=1, tokenizer=processor.tokenizer, processor=processor)


def test_png_is_lossless_and_mutating_observation_does_not_mutate_messages():
    array = np.full((16, 32, 3), 27, np.uint8)
    encoded = vision.screenshot_data_url(array)
    array[:] = 0
    _, images = vision._processor_messages([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": encoded}},
    ]}])
    assert np.asarray(images[0]).min() == 27


@pytest.mark.parametrize("url", ["https://example.com/x.png", "file:///tmp/x.png", "data:image/png;base64,bad"])
def test_remote_paths_and_corrupt_images_fail_closed(url):
    prompt = messages()
    prompt[0]["content"][1]["image_url"]["url"] = url
    with pytest.raises(InfrastructureError):
        vision._processor_messages(prompt)


def test_image_tokens_mrope_and_pixels_survive_json_and_training_padding(fake_verl, processor):
    rows = visual_rows(processor)
    batch = pack_rows(rows, processor, world=4)
    assert batch.batch["position_ids"].shape == (4, 4, len(rows[0].prompt_token_ids) + 2)
    assert batch.batch["response_mask"].tolist() == [[1, 1], [1, 1], [0, 0], [0, 0]]
    assert batch.meta_info["global_response_tokens"] == 4
    inputs = batch.non_tensor_batch["multi_modal_inputs"]
    assert not torch.equal(inputs[0]["pixel_values"], inputs[1]["pixel_values"])
    assert torch.equal(inputs[0]["pixel_values"], inputs[2]["pixel_values"])
    item = vision.encode_prompt(rows[0].messages, processor.tokenizer, processor)
    assert len(item.token_ids) > len(item.raw_token_ids)
    assert torch.equal(batch.batch["position_ids"][0, :, :-2], item.positions)
    assert torch.equal(batch.batch["position_ids"][0, :, -2:], item.positions[:, -1:] + torch.arange(1, 3))


@pytest.mark.parametrize("change", ["image", "tokens", "digest", "messages", "processor"])
def test_changed_or_missing_replay_input_cannot_enter_update(fake_verl, processor, change):
    rows = visual_rows(processor)
    if change == "image":
        rows[0].messages = messages(255)
    elif change == "tokens":
        rows[0].prompt_token_ids[0] += 1
    elif change == "digest":
        rows[0].input_encoding["input_sha256"] = "wrong"
    elif change == "messages":
        rows[0].messages = []
    if change == "processor":
        with pytest.raises(ValueError, match="processor"):
            tensor_batch(rows, pad_token_id=0, world_size=1, max_prompt_tokens=1000,
                         max_response_tokens=10, temperature=1)
    else:
        with pytest.raises(ValueError):
            pack_rows(rows, processor)


def test_switch_and_expanded_image_token_budget(processor):
    cfg = config()
    validate_config(cfg)
    with pytest.raises(ValueError, match="processor"):
        VerlPolicy(None, processor.tokenizer, cfg)
    policy = VerlPolicy(None, processor.tokenizer, cfg, processor=processor)
    request = GenerationRequest("a", messages(), 17, "update-0")
    encoded = policy._encode(request)
    policy.max_prompt_tokens = len(encoded.raw_token_ids)
    with pytest.raises(ContextLimitError):
        policy._encode(request)
    with pytest.raises(InfrastructureError, match="mode change"):
        policy._encode(GenerationRequest("b", [{"role": "user", "content": "no image"}], 18, "update-0"))
    cfg["harness"]["use_screenshot"] = False
    policy = VerlPolicy(None, processor.tokenizer, cfg, processor=processor)
    with pytest.raises(InfrastructureError, match="mode change"):
        policy._encode(request)
    text = policy._encode(GenerationRequest("b", [{"role": "user", "content": "no image"}], 18, "update-0"))
    assert text.positions.shape[0] == 4 and not text.image_inputs and not text.images


def test_images_reach_generation_old_scoring_reference_and_update(fake_verl, monkeypatch, processor):
    protocol = ModuleType("verl.protocol")
    protocol.pad_dataproto_to_divisor = lambda data, divisor: (data, 0)
    protocol.unpad_dataproto = lambda data, count: data
    monkeypatch.setitem(sys.modules, "verl.protocol", protocol)

    class Worker:
        world_size = 1

        def generate_sequences(self, data):
            self.sampled = deepcopy(data)
            assert len(data.non_tensor_batch["multi_modal_data"][0]["image"]) == 1
            assert data.batch["position_ids"].ndim == 3
            return FakeDataProto({
                "responses": torch.tensor([[50, 2]]), "prompts": data.batch["input_ids"],
                "attention_mask": torch.cat((data.batch["attention_mask"], torch.ones(1, 2, dtype=torch.long)), -1),
            }, data.non_tensor_batch, {"eos_token_ids": [2]})

        def compute_log_prob(self, data):
            self.old_inputs = data.non_tensor_batch["multi_modal_inputs"]
            return FakeDataProto({"old_log_probs": torch.tensor([[-0.7, -0.8]]), "entropys": torch.ones(1, 2)})

        def compute_ref_log_prob(self, data):
            self.ref_inputs = deepcopy(data.non_tensor_batch["multi_modal_inputs"])
            return FakeDataProto({"ref_log_prob": torch.zeros_like(data.batch["old_log_probs"])})

        def update_actor(self, data):
            self.updated = data
            assert "ref_log_prob" in data.batch
            return FakeDataProto(meta_info={"metrics": {"test/updated": 1.0}})

    worker = Worker()
    policy = VerlPolicy(worker, processor.tokenizer, config(), processor=processor)
    result = policy.generate(messages())
    assert result.metadata["input_encoding"]["image_count"] == 1
    rows = visual_rows(processor)
    # Use the actual generated call for one training row.
    rows[0].messages = messages()
    rows[0].input_encoding = result.metadata["input_encoding"]
    rows[0].prompt_token_ids = result.prompt_token_ids
    rows[0].response_token_ids = result.response_token_ids
    rows[0].old_log_probs = result.old_log_probs
    backend = VerlBackend.__new__(VerlBackend)
    backend.worker, backend.policy, backend.update_count = worker, policy, 0
    backend.owner = SimpleNamespace(ref_in_actor=True)
    backend.verl_config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(use_kl_loss=True)))
    backend.update(rows)
    for key in ("pixel_values", "image_grid_thw"):
        expected = worker.sampled.non_tensor_batch["multi_modal_inputs"][0][key]
        assert torch.equal(expected, worker.old_inputs[0][key])
        assert torch.equal(expected, worker.ref_inputs[0][key])
        assert torch.equal(expected, worker.updated.non_tensor_batch["multi_modal_inputs"][0][key])


def test_vllm_expansion_mismatch_is_rejected_and_hook_restored():
    engine = SimpleNamespace(generate=lambda **kw: [SimpleNamespace(prompt_token_ids=[1, 9, 9, 2])])
    original = engine.generate
    with _per_request_sampling(SimpleNamespace(inference_engine=engine), [17], [[1, 9, 9, 2]]):
        engine.generate(prompts=[{}], sampling_params=SimpleNamespace(seed=None))
    with pytest.raises(RuntimeError, match="expansion differs"):
        with _per_request_sampling(SimpleNamespace(inference_engine=engine), [17], [[1, 9, 2]]):
            engine.generate(prompts=[{}], sampling_params=SimpleNamespace(seed=None))
    assert engine.generate is original


def test_policy_gradient_depends_on_pixels_and_ignores_dummy_rows(fake_verl, monkeypatch, processor):
    core = ModuleType("verl.trainer.ppo.core_algos")
    def objective(*, old_log_prob, log_prob, advantages, response_mask, **kwargs):
        ratio = (log_prob - old_log_prob).exp()
        loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages)
        return (loss * response_mask).sum() / response_mask.shape[-1], {
            "actor/ppo_kl": 0., "actor/pg_clipfrac": 0.,
        }
    core.compute_policy_loss_vanilla = objective
    core.kl_penalty = lambda *args: None
    device = ModuleType("verl.utils.device")
    device.get_device_id = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", core)
    monkeypatch.setitem(sys.modules, "verl.utils.device", device)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)

    class Actor:
        def __init__(self):
            self.actor_module = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.actor_module.weight)
            self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=.1)
            self.config = SimpleNamespace(use_kl_loss=False, entropy_coeff=0.)

        def _forward_micro_batch(self, inputs, **kwargs):
            pixels = inputs["multi_modal_inputs"][0]["pixel_values"]
            logits = self.actor_module(pixels).reshape(())
            probs = torch.nn.functional.logsigmoid(logits).expand_as(inputs["old_log_probs"])
            return torch.zeros_like(probs), probs

        def _optimizer_step(self):
            norm = self.actor_module.weight.grad.norm().clone()
            self.actor_optimizer.step()
            return norm

    rows = visual_rows(processor)
    for row in rows:
        row.old_log_probs = [-float(np.log(2))] * len(row.response_token_ids)
    parameters = []
    for world in (1, 4):
        data = pack_rows(rows, processor, world)
        actor = Actor()
        _online_update_policy(actor, data)
        parameters.append(actor.actor_module.weight.item())
    assert parameters[0] > 0 and parameters[0] == pytest.approx(parameters[1])
    # Only exchange pixels, preserving token IDs, rewards and old likelihoods.
    data = pack_rows(rows, processor)
    data.non_tensor_batch["multi_modal_inputs"] = data.non_tensor_batch["multi_modal_inputs"][::-1].copy()
    actor = Actor()
    _online_update_policy(actor, data)
    assert actor.actor_module.weight.item() == pytest.approx(-parameters[0])
