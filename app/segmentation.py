"""Segmentation logic."""

import functools
import logging
import re
from typing import Callable, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import apply_chat_template, generate, load
from mlx_vlm.utils import load_image

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ResBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=1, padding=0
        )

    def __call__(self, x: mx.array) -> mx.array:
        original_x = x
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = self.conv3(x)
        return x + original_x


class Decoder(nn.Module):
    """
    Decoder that upscales quantized vectors to produce a mask.
    The architecture is parameterized to avoid hardcoded layer definitions.
    Takes channels-last input data (B, H, W, C).
    """

    def __init__(
        self,
        in_channels: int = 512,
        res_channels: int = 128,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        upsample_channels: Tuple[int, ...] = (128, 64, 32, 16),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels=in_channels, out_channels=res_channels, kernel_size=1, padding=0
        )
        self.res_blocks = [
            ResBlock(features=res_channels) for _ in range(num_res_blocks)
        ]
        self.upsample_layers = []
        out_up_ch = res_channels
        for ch in upsample_channels:
            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    in_channels=out_up_ch,
                    out_channels=ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            out_up_ch = ch
        self.conv_out = nn.Conv2d(
            in_channels=upsample_channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        for layer in self.upsample_layers:
            x = nn.relu(layer(x))

        return self.conv_out(x)


def _get_params(checkpoint: dict) -> dict:
    """Converts PyTorch checkpoint to MLX params (nested dict).
    Uses transpositions yielding (Out, H, W, In) format weights."""

    def transp(kernel: np.ndarray) -> mx.array:
        return mx.transpose(mx.array(kernel), (0, 2, 3, 1))

    def transp_transpose(kernel: np.ndarray) -> mx.array:
        intermediate = mx.transpose(mx.array(kernel), (1, 0, 2, 3))

        return mx.transpose(intermediate, (0, 2, 3, 1))

    def conv(name: str) -> dict:
        return {
            "bias": mx.array(checkpoint[f"{name}.bias"]),
            "weight": transp(checkpoint[f"{name}.weight"]),
        }

    def conv_transpose(name: str) -> dict:
        return {
            "bias": mx.array(checkpoint[f"{name}.bias"]),
            "weight": transp_transpose(checkpoint[f"{name}.weight"]),
        }

    def resblock(name: str) -> dict:
        return {
            "conv1": conv(f"{name}.0"),
            "conv2": conv(f"{name}.2"),
            "conv3": conv(f"{name}.4"),
        }

    params = {
        "_embeddings": mx.array(checkpoint["_vq_vae._embedding"]),
        "conv_in": conv("decoder.0"),
        "res_blocks": [
            resblock("decoder.2.net"),
            resblock("decoder.3.net"),
        ],
        "upsample_layers": [
            conv_transpose("decoder.4"),
            conv_transpose("decoder.6"),
            conv_transpose("decoder.8"),
            conv_transpose("decoder.10"),
        ],
        "conv_out": conv("decoder.12"),
    }

    return params


def _quantized_values_from_codebook_indices(
    codebook_indices: mx.array, embeddings: mx.array
) -> mx.array:
    batch_size, num_tokens = codebook_indices.shape
    expected_tokens = 16
    if num_tokens != expected_tokens:
        log.error(f"Expected {expected_tokens} tokens, got {codebook_indices.shape}")

    encodings = mx.take(embeddings, codebook_indices.reshape((-1,)), axis=0)

    return encodings.reshape((batch_size, 4, 4, embeddings.shape[1]))


@functools.cache
def get_reconstruct_masks(checkpoint_path: str) -> Callable[[mx.array], mx.array]:
    """Loads the checkpoint and returns a function that reconstructs masks
    from codebook indices using a preloaded MLX decoder.
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = dict(np.load(f))

    params = _get_params(checkpoint_data)
    embeddings = params.pop("_embeddings")
    log.info(f"VAE embedding dimension: {embeddings.shape[1]}")

    decoder = Decoder()
    decoder.update(params)

    def reconstruct_masks(codebook_indices: mx.array) -> mx.array:
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, embeddings
        )
        return decoder(quantized)

    return reconstruct_masks


def extract_and_create_arrays(pattern: str) -> List[mx.array]:
    """Extracts segmentation tokens from each object in the pattern and returns a list of MLX arrays."""
    object_strings = [obj.strip() for obj in pattern.split(";") if obj.strip()]

    seg_tokens_arrays = []
    for obj in object_strings:
        seg_tokens = re.findall(r"<seg(\d{3})>", obj)
        if seg_tokens:
            seg_numbers = [int(token) for token in seg_tokens]
            seg_tokens_arrays.append(mx.array(seg_numbers))

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


def gather_masks(output, codes_list, reconstruct_fn):
    masks_list = []

    target_width, target_height = 448, 448
    for i, codes in enumerate(codes_list):
        codes_batch = codes[None, :]
        masks = reconstruct_fn(codes_batch)
        mask_np = np.array(masks[0, :, :, 0], copy=False)

        y_min, x_min, y_max, x_max = parse_bbox(output)[i]
        x_min_norm = int(x_min / 1024 * target_width)
        y_min_norm = int(y_min / 1024 * target_height)
        x_max_norm = int(x_max / 1024 * target_width)
        y_max_norm = int(y_max / 1024 * target_height)

        masks_list.append(
            {
                "mask": mask_np,
                "coordinates": (x_min_norm, y_min_norm, x_max_norm, y_max_norm),
            }
        )

    return masks_list


def segment_image(
    model_path: str,
    vae_checkpoint: str,
    image_input: str,
    prompt: str,
) -> list:
    """Returns a list of dicts: {mask: np.ndarray, coordinates: (x0,y0,x1,y1)}"""
    model, processor = load(model_path)
    reconstruct_fn = get_reconstruct_masks(vae_checkpoint)

    primed = apply_chat_template(processor, model.config, prompt + "\n", num_images=1)
    image = (
        image_input if isinstance(image_input, mx.array) else load_image(image_input)
    )

    output = generate(model, processor, primed, image, verbose=False)
    codes = extract_and_create_arrays(output)

    return gather_masks(output, codes, reconstruct_fn)
