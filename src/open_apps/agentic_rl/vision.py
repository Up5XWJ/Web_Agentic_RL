"""Immutable screenshot inputs shared by rollout and policy optimization.

Artifacts store PNGs in AttemptRecord.messages, not temporary browser paths or
large float tensors. Training reprocesses those same bytes and checks a digest
of token IDs, MRoPE positions and every image tensor against sampling time.
GPU/Transformers/veRL imports stay lazy so the text harness works on a laptop.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
from typing import Any

from .types import InfrastructureError


def screenshot_data_url(screenshot: Any) -> str:
    """Capture a lossless RGB PNG; missing/invalid observations fail closed."""
    import numpy as np
    from PIL import Image

    if isinstance(screenshot, Image.Image):
        image = screenshot.convert("RGB")
    elif (isinstance(screenshot, np.ndarray) and screenshot.dtype == np.uint8
          and screenshot.ndim == 3 and screenshot.shape[2] in (3, 4)
          and min(screenshot.shape[:2]) > 0):
        image = Image.fromarray(screenshot).convert("RGB")
    else:
        raise InfrastructureError("use_screenshot=true requires a valid current RGB screenshot")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def has_images(messages: list[dict[str, Any]]) -> bool:
    return any(isinstance(m.get("content"), list) and any(
        part.get("type") in {"image", "image_url"} for part in m["content"]
    ) for m in messages)


def _processor_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Normalize OpenAI-style blocks without downloading URLs or opening files."""
    from PIL import Image

    normalized, images = [], []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    prefix = "data:image/png;base64,"
                    if not isinstance(url, str) or not url.startswith(prefix):
                        raise InfrastructureError("RL images must be embedded PNGs, not remote URLs or paths")
                    try:
                        data = base64.b64decode(url[len(prefix):], validate=True)
                        with Image.open(BytesIO(data)) as image:
                            if image.format != "PNG":
                                raise ValueError("Not a PNG")
                            images.append(image.convert("RGB"))
                    except Exception as exc:
                        raise InfrastructureError("Invalid recorded screenshot PNG") from exc
                    parts.append({"type": "image"})
                else:
                    raise InfrastructureError("Unsupported RL message block; expected text or image_url")
        else:
            raise InfrastructureError("RL message content must be text or a list of typed blocks")
        normalized.append({"role": message["role"], "content": parts})
    return normalized, images


def _qwen_positions(processor: Any, ids: Any, inputs: dict[str, Any]) -> Any:
    import torch
    from verl.models.transformers.qwen3_vl import get_rope_index

    # Match veRL 0.6.1 RLHFDataset: first channel is the ordinary text position
    # (used for packed sequences), then three temporal/height/width MRoPE tracks.
    vision = get_rope_index(
        processor, input_ids=ids, image_grid_thw=inputs.get("image_grid_thw"),
        attention_mask=torch.ones_like(ids),
    )
    return torch.cat((torch.arange(len(ids)).unsqueeze(0), vision), dim=0)


@dataclass
class EncodedPrompt:
    token_ids: list[int]
    raw_token_ids: list[int]
    positions: Any = None
    images: list[Any] = field(default_factory=list)
    image_inputs: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def _input_hash(ids: Any, positions: Any, image_inputs: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted({"input_ids": ids, "position_ids": positions, **image_inputs}.items()):
        if not isinstance(tensor, torch.Tensor):
            raise InfrastructureError(f"Unsupported non-tensor processor output: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encode_prompt(
    messages: list[dict[str, Any]], tokenizer: Any, processor: Any = None,
    *, chat_template_kwargs: dict[str, Any] | None = None,
) -> EncodedPrompt:
    kwargs = deepcopy(chat_template_kwargs or {})
    if processor is None:
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise InfrastructureError("Screenshot inputs require a Qwen3-VL processor/model")
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
        ids = [int(i) for i in ids]
        return EncodedPrompt(ids, list(ids))
    if type(processor).__name__ != "Qwen3VLProcessor":
        raise InfrastructureError("This RL multimodal adapter currently supports Qwen3VLProcessor only")
    try:
        normalized, images = _processor_messages(messages)
        raw = processor.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True, **kwargs)
        model_inputs = dict(processor(text=[raw], images=images or None, return_tensors="pt"))
        ids = model_inputs.pop("input_ids")[0]
        attention = model_inputs.pop("attention_mask")
        if not bool(attention.all()):
            raise InfrastructureError("Unexpected padding in a single unpadded prompt")
        if images and not {"pixel_values", "image_grid_thw"}.issubset(model_inputs):
            raise InfrastructureError("Processor discarded screenshot pixels/grid")
        if not images and model_inputs:
            raise InfrastructureError("Image tensors appeared in a text-only prompt")
        positions = _qwen_positions(processor, ids, model_inputs)
        return EncodedPrompt(
            ids.tolist(), tokenizer.encode(raw, add_special_tokens=False), positions,
            images, model_inputs,
            {"format": "qwen3-vl-input-v1", "input_sha256": _input_hash(ids, positions, model_inputs),
             "chat_template_kwargs": kwargs, "image_count": len(images)},
        )
    except InfrastructureError:
        raise
    except Exception as exc:
        raise InfrastructureError("Qwen3-VL prompt encoding failed; check processor and GPU dependencies") from exc


def object_array(values: list[Any]) -> Any:
    import numpy as np

    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def pack_prompts(encoded: list[EncodedPrompt], pad_token_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Left-pad actor tokens; vLLM gets unexpanded placeholders plus the PNG pixels."""
    import torch

    width = max(len(e.token_ids) for e in encoded)
    ids = torch.full((len(encoded), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    multimodal = encoded[0].positions is not None
    if any((e.positions is not None) != multimodal for e in encoded):
        raise InfrastructureError("Cannot mix processor and tokenizer-only prompts in one policy batch")
    positions = torch.zeros((len(encoded), 4, width) if multimodal else ids.shape, dtype=torch.long)
    for row, item in enumerate(encoded):
        size = len(item.token_ids)
        ids[row, -size:] = torch.tensor(item.token_ids)
        attention[row, -size:] = 1
        positions[row, ..., -size:] = item.positions if multimodal else torch.arange(size)
    non_tensors = {"raw_prompt_ids": object_array([e.raw_token_ids for e in encoded])}
    if multimodal:
        non_tensors["multi_modal_inputs"] = object_array([e.image_inputs for e in encoded])
        non_tensors["expected_prompt_ids"] = object_array([e.token_ids for e in encoded])
        if any(e.images for e in encoded):
            non_tensors["multi_modal_data"] = object_array([{"image": e.images} if e.images else {} for e in encoded])
    return {"input_ids": ids, "attention_mask": attention, "position_ids": positions}, non_tensors
