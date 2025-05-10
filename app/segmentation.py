import base64
import functools
import io
import logging
import os
import re
from typing import Any, Callable, Dict, List

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.image_utils import load_image

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

VAE_MODEL = "vae-oid.npz"
MODEL_ID = os.getenv("MODEL_ID", "google/paligemma2-3b-mix-448")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")


class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        original_x = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)

        return x + original_x


class Decoder(nn.Module):
    """Upscales quantized vectors to mask."""

    @nn.compact
    def __call__(self, x):
        num_res_blocks = 2
        dim = 128
        num_upsample_layers = 4

        x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
        x = nn.relu(x)

        for _ in range(num_res_blocks):
            x = ResBlock(features=dim)(x)

        for _ in range(num_upsample_layers):
            x = nn.ConvTranspose(
                features=dim,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding=2,
                transpose_kernel=True,
            )(x)
            x = nn.relu(x)
            dim //= 2

        x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)

        return x


def _get_params(checkpoint):
    """Converts PyTorch checkpoint to Flax params."""

    def transp(kernel):
        return np.transpose(kernel, (2, 3, 1, 0))

    def conv(name):
        return {
            "bias": checkpoint[name + ".bias"],
            "kernel": transp(checkpoint[name + ".weight"]),
        }

    def resblock(name):
        return {
            "Conv_0": conv(name + ".0"),
            "Conv_1": conv(name + ".2"),
            "Conv_2": conv(name + ".4"),
        }

    return {
        "_embeddings": checkpoint["_vq_vae._embedding"],
        "Conv_0": conv("decoder.0"),
        "ResBlock_0": resblock("decoder.2.net"),
        "ResBlock_1": resblock("decoder.3.net"),
        "ConvTranspose_0": conv("decoder.4"),
        "ConvTranspose_1": conv("decoder.6"),
        "ConvTranspose_2": conv("decoder.8"),
        "ConvTranspose_3": conv("decoder.10"),
        "Conv_1": conv("decoder.12"),
    }


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    _, embedding_dim = embeddings.shape

    encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)

    return encodings.reshape((batch_size, 4, 4, embedding_dim))


@functools.cache
def get_reconstruct_masks():
    """Reconstructs masks from codebook indices.

    Based on code from https://arxiv.org/abs/2301.02229

    Verified in
    https://colab.research.google.com/drive/1AOr0cokOpM6-N9Z5HmxoeGxGj6jS37Vl

    Args:
        model: Model to use for conversion.

    Returns:
        A function that expects indices shaped `[B, 16]` of dtype int32, each
        ranging from 0 to 127 (inclusive), and that returns a decoded masks sized
        `[B, 64, 64, 1]`, of dtype float32, in range [-1, 1].
    """

    def reconstruct_masks(codebook_indices):
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params["_embeddings"]
        )
        return Decoder().apply({"params": params}, quantized)

    vae_path = os.path.join(MODELS_DIR, VAE_MODEL)
    with open(vae_path, "rb") as f:
        params = _get_params(dict(np.load(f)))

    return jax.jit(reconstruct_masks, backend="cpu")


def extract_and_create_arrays(pattern: str):
    """Extracts segmentation tokens from each object in the pattern and returns a list of MLX arrays."""
    object_strings = [obj.strip() for obj in pattern.split(";") if obj.strip()]

    seg_tokens_arrays = []
    for obj in object_strings:
        seg_tokens = re.findall(r"<seg(\d{3})>", obj)
        if seg_tokens:
            seg_numbers = [int(token) for token in seg_tokens]
            seg_tokens_arrays.append(np.array(seg_numbers))

    return seg_tokens_arrays


def parse_bbox(model_output: str):
    entries = model_output.split(";")

    results = []
    for entry in entries:
        entry = entry.strip()
        numbers = re.findall(r"<loc(\d+)>", entry)
        if len(numbers) == 4:
            bbox = [int(num) for num in numbers]
            results.append(bbox)

    return results


def gather_masks(
    output: str, codes_list: List[List[int]], reconstruct_fn: Callable
) -> List[Dict[str, Any]]:
    masks_list = []

    target_width, target_height = 448, 448
    for i, codes in enumerate(codes_list):
        codes_batch = codes[None, :]
        masks = reconstruct_fn(codes_batch)
        mask_np = np.array(masks[0, :, :, 0], copy=False)
        mask_np = np.clip(mask_np * 0.5 + 0.5, 0, 1)
        mask_np = Image.fromarray((mask_np * 255).astype("uint8"))
        buffer = io.BytesIO()
        mask_np.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        y_min, x_min, y_max, x_max = parse_bbox(output)[i]
        x_min_norm = int(x_min / 1024 * target_width)
        y_min_norm = int(y_min / 1024 * target_height)
        x_max_norm = int(x_max / 1024 * target_width)
        y_max_norm = int(y_max / 1024 * target_height)

        masks_list.append(
            {
                "mask": mask_base64,
                "coordinates": (x_min_norm, y_min_norm, x_max_norm, y_max_norm),
            }
        )

    return masks_list


def segment_image(
    model_id: str,
    prompt: str = None,
    image_url: str = None,
    image_file=None,
) -> list:
    """Returns a list of dicts: {mask: np.ndarray, coordinates: (x0,y0,x1,y1)}"""

    log.info(f"Loading model: {model_id}")
    model_dir = os.path.join(MODELS_DIR, "huggingface")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=model_dir,
        local_files_only=True,
    ).eval()
    processor = PaliGemmaProcessor.from_pretrained(
        MODEL_ID, cache_dir=model_dir, local_files_only=True
    )

    log.info(f"Model loaded successfully. Processing image with prompt: {prompt}")

    if image_file:
        image_content = image_file.file.read()
        image = Image.open(io.BytesIO(image_content))
        image = load_image(image)
    elif image_url:
        image = load_image(image_url)
    else:
        raise ValueError("No image provided")

    model_inputs = (
        processor(text=prompt, images=image, return_tensors="pt")
        .to(torch.bfloat16)
        .to(model.device)
    )
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    codes = extract_and_create_arrays(decoded)
    reconstruct_fn = get_reconstruct_masks()

    return gather_masks(decoded, codes, reconstruct_fn)
