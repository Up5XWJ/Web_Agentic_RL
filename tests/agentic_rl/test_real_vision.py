"""Opt-in actual Transformers processor/tiny random VLM check (no downloads).

Requires the RL Transformers version and a checkout of pinned veRL source.
This is deliberately not a vLLM/CUDA/FSDP/LoRA integration claim.
"""

import ast
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENAPPS_REAL_VISION_TEST") != "1" or not os.environ.get("OPENAPPS_VERL_SOURCE"),
    reason="optional real Qwen3-VL CPU processor/model test with pinned upstream source",
)


def test_actual_processor_mrope_and_image_conditioned_backward(monkeypatch):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor,
                              Qwen3VLProcessor, Qwen3VLVideoProcessor,
                              Qwen3VLConfig, Qwen3VLForConditionalGeneration)
    from open_apps.agentic_rl import vision

    vocabulary = ["[PAD]", "[UNK]", "[EOS]", "<|vision_start|>", "<|vision_end|>",
                  "<|image_pad|>", "<|video_pad|>", "user", "assistant", "inspect", "done"]
    raw_tokenizer = Tokenizer(WordLevel({s: i for i, s in enumerate(vocabulary)}, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=raw_tokenizer, pad_token="[PAD]",
        eos_token="[EOS]", unk_token="[UNK]", additional_special_tokens=vocabulary[3:7])
    template = (
        "{% for m in messages %}{{ m.role }} {% for p in m.content %}"
        "{% if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ p.text }}{% endif %} {% endfor %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    processor = Qwen3VLProcessor(tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=16, min_pixels=32*32, max_pixels=64*64),
        video_processor=Qwen3VLVideoProcessor(), chat_template=template)
    # Execute the actual pinned standalone helper without importing GPU workers.
    source = Path(os.environ["OPENAPPS_VERL_SOURCE"]) / "verl/models/transformers/qwen3_vl.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_rope_index")
    namespace = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    def positions(proc, ids, inputs):
        rope = namespace["get_rope_index"](proc, ids, image_grid_thw=inputs.get("image_grid_thw"),
                                           attention_mask=torch.ones_like(ids))
        return torch.cat((torch.arange(len(ids)).unsqueeze(0), rope))
    monkeypatch.setattr(vision, "_qwen_positions", positions)
    def encoded(color):
        return vision.encode_prompt([{"role": "user", "content": [
            {"type": "text", "text": "inspect"},
            {"type": "image_url", "image_url": {"url": vision.screenshot_data_url(np.full((64, 64, 3), color, np.uint8))}},
        ]}], tokenizer, processor)
    black, white = encoded(0), encoded(255)
    assert black.token_ids == white.token_ids
    assert len(black.token_ids) > len(black.raw_token_ids)
    assert black.positions.shape == (4, len(black.token_ids))
    assert black.provenance["input_sha256"] != white.provenance["input_sha256"]
    assert black.provenance == encoded(0).provenance
    assert black.token_ids.count(processor.image_token_id) == 4
    torch.manual_seed(7)
    cfg = Qwen3VLConfig(
        text_config={"vocab_size": len(vocabulary), "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
            "head_dim": 8, "max_position_embeddings": 256,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "patch_size": 16, "spatial_merge_size": 2, "out_hidden_size": 32,
            "deepstack_visual_indexes": [0, 1], "num_position_embeddings": 16},
        image_token_id=5, video_token_id=6, vision_start_token_id=3, vision_end_token_id=4,
    )
    model = Qwen3VLForConditionalGeneration(cfg).float().eval()
    outputs = []
    for item in (black, white):
        ids = torch.tensor([item.token_ids + [10, 2]])
        rope = torch.cat((item.positions, item.positions[:, -1:] + torch.arange(1, 3)), -1)
        # HF consumes the three spatial tracks; veRL strips the extra text track
        # in its patched forward used for packed sequences.
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       position_ids=rope[1:].unsqueeze(1), **item.image_inputs)
        outputs.append(result.logits[:, -3:-1])
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-6)
    loss = torch.nn.functional.cross_entropy(outputs[0].reshape(2, -1), torch.tensor([10, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
