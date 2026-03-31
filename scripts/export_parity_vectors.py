#!/usr/bin/env python3
"""
Export intermediate tensors from ST-EEGFormer Python model for Rust parity testing.

Exports:
  1. Model weights as safetensors (encoder-only)
  2. Reference input/output tensors at each stage
  3. A fixed random seed for reproducibility

Usage:
    python scripts/export_parity_vectors.py \
        --checkpoint data/checkpoint-small-300.pth \
        --output tests/vectors/
"""

import argparse
import sys
import os
import numpy as np
import torch
import math

# Add STEEGFormer to path
sys.path.insert(0, "/Users/Shared/STEEGFormer/pretrain")

from models_mae_eeg import MaskedAutoencoderViT, mae_vit_small_patch16
from safetensors.torch import save_file


def build_encoder_model(checkpoint_path):
    """Load the MAE model and extract the encoder."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    # Build MAE small model
    model = mae_vit_small_patch16()
    model.load_state_dict(state_dict)
    model.eval()
    return model, state_dict


def export_encoder_weights(state_dict, output_dir):
    """Export encoder-only weights as safetensors."""
    encoder_keys = {}
    skip_prefixes = ("decoder_", "dec_", "mask_token")

    for k, v in state_dict.items():
        skip = False
        for prefix in skip_prefixes:
            if k.startswith(prefix):
                skip = True
                break
        if not skip:
            encoder_keys[k] = v.contiguous()

    path = os.path.join(output_dir, "encoder_small.safetensors")
    save_file(encoder_keys, path)
    print(f"Saved {len(encoder_keys)} encoder weight tensors → {path}")
    return path


def generate_test_input(n_channels=4, n_samples=768, seed=42):
    """Generate a deterministic test EEG input."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Simulate EEG: sum of sinusoids + noise
    t = torch.linspace(0, 6.0, n_samples)  # 6 seconds at 128 Hz
    signal = torch.zeros(1, n_channels, n_samples)
    for ch in range(n_channels):
        alpha = torch.sin(2 * math.pi * (9.5 + ch * 0.3) * t) * 20e-6
        theta = torch.sin(2 * math.pi * (5.0 + ch * 0.1) * t) * 15e-6
        noise = torch.randn(n_samples) * 2e-6
        signal[0, ch] = alpha + theta + noise

    # Channel indices: use Fz(25), C3(130), C4(2), Pz(1)
    chan_idx = torch.tensor([[25, 130, 2, 1]], dtype=torch.long)

    return signal, chan_idx


def export_intermediates(model, signal, chan_idx, output_dir):
    """Run the encoder forward pass and export intermediate values."""
    tensors = {}

    # Input
    tensors["input_signal"] = signal[0]  # [C, T]
    tensors["input_chan_idx"] = chan_idx[0].float()  # [C]

    with torch.no_grad():
        # 1. Patch embed
        x = model.patch_embed(signal)  # [B, Seq, Ch_all, Dmodel]
        B, Seq, Ch_all, Dmodel = x.shape
        Seq_total = Seq * Ch_all
        tensors["after_patch_embed"] = x[0].reshape(Seq_total, Dmodel)  # [Seq_total, D]

        x_flat = x.view(B, Seq_total, Dmodel)

        # 2a. Channel embeddings
        if chan_idx.dim() == 1:
            chan_idx_exp = chan_idx.unsqueeze(0).expand(B, -1)
        else:
            chan_idx_exp = chan_idx
        ch_emb_small = model.enc_channel_emd(chan_idx_exp)  # [B, Ch_all, D]
        ch_emb = (
            ch_emb_small
              .unsqueeze(1)
              .repeat_interleave(Seq, dim=1)
              .view(B, Seq_total, Dmodel)
        )
        tensors["channel_embedding"] = ch_emb[0]  # [Seq_total, D]

        # 2b. Temporal embeddings
        temp_idx = torch.arange(Seq, device=signal.device, dtype=torch.long).unsqueeze(0)
        temp_emb_small_2d = model.enc_temporal_emd(temp_idx)  # [1, Seq, D]
        temp_emb_small = temp_emb_small_2d.squeeze(0)  # [Seq, D]
        temp_emb_flat = (
            temp_emb_small
              .unsqueeze(1)
              .repeat_interleave(Ch_all, dim=1)
              .view(Seq_total, Dmodel)
        )
        tp_emb = temp_emb_flat.unsqueeze(0).expand(B, -1, -1)
        tensors["temporal_embedding"] = tp_emb[0]  # [Seq_total, D]

        # Wait — Python uses 1-indexed temporal indices!
        # Let me check: in forward_encoder the Python code is:
        #   temp_idx = torch.arange(Seq, device=x.device, dtype=torch.long).unsqueeze(0)
        # But in the ORIGINAL forward_encoder in models_mae_eeg.py, the optimized version uses:
        #   temp_idx = torch.arange(Seq, ...) — 0-indexed
        # While forward_encoder_demo and forward_decoder use 1-indexed:
        #   seq_tensor = torch.arange(1, Seq+1, ...)
        # Let me use the ACTUAL forward_encoder code path

        # Re-do with the actual forward_encoder
        # Actually let me just call forward_encoder directly
        pass

    # Actually, let's just call forward_encoder and capture the full output
    with torch.no_grad():
        # The encoder uses mask_ratio=0.0 to get all tokens
        # But we can't set mask_ratio=0 easily. Let's trace manually.
        # Use the encoder demo with 0 masking
        x = model.patch_embed(signal)
        B, Seq, Ch_all, Dmodel = x.shape
        Seq_total = Seq * Ch_all
        x_flat = x.view(B, Seq_total, Dmodel)

        # Channel embedding (using the optimized forward_encoder path)
        if chan_idx.dim() == 1:
            chan_idx_b = chan_idx.unsqueeze(0).expand(B, -1)
        else:
            chan_idx_b = chan_idx
        ch_emb_small = model.enc_channel_emd(chan_idx_b)
        ch_emb = (
            ch_emb_small
              .unsqueeze(1)
              .repeat_interleave(Seq, dim=1)
              .view(B, Seq_total, Dmodel)
        )

        # Temporal embedding — the optimized forward_encoder uses 0-indexed
        temp_idx = torch.arange(Seq, device=signal.device, dtype=torch.long).unsqueeze(0)
        temp_emb_small_2d = model.enc_temporal_emd(temp_idx)
        temp_emb_small = temp_emb_small_2d.squeeze(0)
        temp_emb_flat = (
            temp_emb_small
              .unsqueeze(1)
              .repeat_interleave(Ch_all, dim=1)
              .view(Seq_total, Dmodel)
        )
        tp_emb = temp_emb_flat.unsqueeze(0).expand(B, -1, -1)

        # Add positional encodings
        x_pos = x_flat + tp_emb + ch_emb
        tensors["after_pos_encoding"] = x_pos[0]  # [Seq_total, D]

        # CLS token
        cls_token = model.cls_token + model.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat((cls_tokens, x_pos), dim=1)
        tensors["after_cls_prepend"] = x_with_cls[0]  # [1+Seq_total, D]

        # After block 0
        x_blk = x_with_cls
        x_blk = model.blocks[0](x_blk)
        tensors["after_block_0"] = x_blk[0]

        # After all blocks
        x_all = x_with_cls
        for blk in model.blocks:
            x_all = blk(x_all)
        tensors["after_all_blocks"] = x_all[0]

        # After norm
        x_normed = model.norm(x_all)
        tensors["after_norm"] = x_normed[0]

        # CLS token output
        cls_output = x_normed[0, 0].clone()  # [D]
        tensors["cls_output"] = cls_output

        # Global pool output (mean of all tokens except CLS)
        global_pool_output = x_normed[0, 1:].mean(dim=0).clone()  # [D]
        tensors["global_pool_output"] = global_pool_output

    # Save
    path = os.path.join(output_dir, "parity.safetensors")
    save_file(tensors, path)
    print(f"Saved {len(tensors)} intermediate tensors → {path}")

    # Print shapes and stats
    for k, v in sorted(tensors.items()):
        print(f"  {k:40s}  {str(list(v.shape)):20s}  mean={v.float().mean():.6f}  std={v.float().std():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/checkpoint-small-300.pth")
    parser.add_argument("--output", default="tests/vectors/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading model...")
    model, state_dict = build_encoder_model(args.checkpoint)

    print("Exporting encoder weights...")
    export_encoder_weights(state_dict, args.output)

    print("Generating test input...")
    signal, chan_idx = generate_test_input()

    print("Running forward pass and exporting intermediates...")
    export_intermediates(model, signal, chan_idx, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
