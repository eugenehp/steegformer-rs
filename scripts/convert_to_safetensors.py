#!/usr/bin/env python3
"""
Convert ST-EEGFormer PyTorch .pth checkpoints to safetensors format.

Splits into:
  - encoder-only weights (for inference)
  - full MAE weights (encoder + decoder, for reconstruction)

Usage:
    python scripts/convert_to_safetensors.py --all
"""

import argparse
import os
import sys

import torch
from safetensors.torch import save_file


CHECKPOINTS = {
    "small": {
        "input": "data/checkpoint-300.pth",
        "encoder": "ST-EEGFormer_small_encoder.safetensors",
        "mae": "ST-EEGFormer_small_mae.safetensors",
        "config": {"patch_size": 16, "embed_dim": 512, "depth": 8, "num_heads": 8, "mlp_ratio": 4.0,
                    "decoder_embed_dim": 384, "decoder_depth": 4, "decoder_num_heads": 16},
    },
    "base": {
        "input": "data/checkpoint-288.pth",
        "encoder": "ST-EEGFormer_base_encoder.safetensors",
        "mae": "ST-EEGFormer_base_mae.safetensors",
        "config": {"patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
                    "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16},
    },
    "large": {
        "input": "data/large_weights_only_196.pth",
        "encoder": "ST-EEGFormer_large_encoder.safetensors",
        "mae": "ST-EEGFormer_large_mae.safetensors",
        "config": {"patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0,
                    "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16},
    },
    "largeV2": {
        "input": "data/large_weights_only_210.pth",
        "encoder": "ST-EEGFormer_largeV2_encoder.safetensors",
        "mae": "ST-EEGFormer_largeV2_mae.safetensors",
        "config": {"patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0,
                    "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16},
    },
}

ENCODER_PREFIXES = [
    "patch_embed.",
    "cls_token",
    "enc_channel_emd.",
    "enc_temporal_emd.",
    "blocks.",
    "norm.",
]

DECODER_PREFIXES = [
    "decoder_",
    "dec_channel_emd.",
    "dec_temporal_emd.",
    "mask_token",
]


def is_encoder_key(key):
    return any(key.startswith(p) for p in ENCODER_PREFIXES)


def convert_checkpoint(variant, info, output_dir):
    input_path = info["input"]
    if not os.path.exists(input_path):
        print(f"  ⚠ Skipping {variant}: {input_path} not found")
        return

    print(f"\n━━━ {variant} ━━━")
    print(f"  Loading {input_path}...")

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Filter to float tensors and make contiguous
    all_tensors = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            all_tensors[k] = v.float().contiguous()

    # Split encoder / full
    encoder_tensors = {k: v for k, v in all_tensors.items() if is_encoder_key(k)}
    mae_tensors = all_tensors  # full model

    # Count params
    enc_params = sum(v.numel() for v in encoder_tensors.values())
    mae_params = sum(v.numel() for v in mae_tensors.values())

    # Save encoder
    enc_path = os.path.join(output_dir, info["encoder"])
    save_file(encoder_tensors, enc_path)
    enc_size = os.path.getsize(enc_path) / 1e6
    print(f"  ✓ Encoder: {len(encoder_tensors)} tensors, {enc_params/1e6:.1f}M params, {enc_size:.0f} MB → {enc_path}")

    # Save full MAE
    mae_path = os.path.join(output_dir, info["mae"])
    save_file(mae_tensors, mae_path)
    mae_size = os.path.getsize(mae_path) / 1e6
    print(f"  ✓ MAE:     {len(mae_tensors)} tensors, {mae_params/1e6:.1f}M params, {mae_size:.0f} MB → {mae_path}")

    # Print key summary
    enc_keys = sorted(encoder_tensors.keys())
    dec_keys = sorted(k for k in mae_tensors if k not in encoder_tensors)
    print(f"  Encoder keys: {len(enc_keys)}")
    print(f"  Decoder keys: {len(dec_keys)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(CHECKPOINTS.keys()), help="Convert a single variant")
    parser.add_argument("--all", action="store_true", help="Convert all variants")
    parser.add_argument("--output", default="data/safetensors", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.all:
        variants = list(CHECKPOINTS.keys())
    elif args.variant:
        variants = [args.variant]
    else:
        parser.print_help()
        return

    for v in variants:
        convert_checkpoint(v, CHECKPOINTS[v], args.output)

    # Write config.yaml
    import json
    config = {"models": {v: CHECKPOINTS[v]["config"] for v in variants}}
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  ✓ Config → {config_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
