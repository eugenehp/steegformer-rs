#!/usr/bin/env python3
"""
Benchmark ST-EEGFormer Python inference for comparison with Rust.

Usage:
    python scripts/benchmark_python.py --checkpoint data/checkpoint-small-300.pth
    python scripts/benchmark_python.py --checkpoint data/checkpoint-small-300.pth --json > bench_results/python.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/Users/Shared/STEEGFormer/pretrain")
from models_mae_eeg import mae_vit_small_patch16


def generate_synthetic_eeg(n_channels, n_samples, sfreq=128.0, seed=42):
    """Match the Rust synthetic EEG generator."""
    torch.manual_seed(seed)
    t = torch.linspace(0, n_samples / sfreq, n_samples)
    signal = torch.zeros(1, n_channels, n_samples)
    for ch in range(n_channels):
        alpha = torch.sin(2 * math.pi * (9.0 + ch * 0.3) * t) * 20e-6
        beta = torch.sin(2 * math.pi * (18.0 + ch * 0.5) * t) * 5e-6
        theta = torch.sin(2 * math.pi * (5.0 + ch * 0.2) * t) * 15e-6
        signal[0, ch] = alpha + beta + theta
    return signal


def run_encoder_forward(model, signal, chan_idx):
    """Run the encoder forward pass (no masking)."""
    with torch.no_grad():
        x = model.patch_embed(signal)
        B, Seq, Ch_all, Dmodel = x.shape
        Seq_total = Seq * Ch_all
        x = x.reshape(B, Seq_total, Dmodel)

        if chan_idx.dim() == 1:
            chan_idx_b = chan_idx.unsqueeze(0).expand(B, -1)
        else:
            chan_idx_b = chan_idx

        ch_emb_small = model.enc_channel_emd(chan_idx_b)
        ch_emb = (
            ch_emb_small.unsqueeze(1)
            .repeat_interleave(Seq, dim=1)
            .view(B, Seq_total, Dmodel)
        )

        temp_idx = torch.arange(Seq, device=signal.device, dtype=torch.long).unsqueeze(0)
        temp_emb_small = model.enc_temporal_emd(temp_idx).squeeze(0)
        temp_emb_flat = (
            temp_emb_small.unsqueeze(1)
            .repeat_interleave(Ch_all, dim=1)
            .view(Seq_total, Dmodel)
        )
        tp_emb = temp_emb_flat.unsqueeze(0).expand(B, -1, -1)

        x = x + tp_emb + ch_emb

        cls_token = model.cls_token + model.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

        return x[:, 0]  # CLS token


def benchmark_config(model, n_channels, n_samples, chan_idx, warmup, runs, device):
    signal = generate_synthetic_eeg(n_channels, n_samples).to(device)
    if chan_idx.shape[1] != n_channels:
        chan_idx_use = chan_idx[:, :n_channels] if n_channels <= chan_idx.shape[1] else torch.arange(n_channels, device=device).unsqueeze(0)
    else:
        chan_idx_use = chan_idx

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = run_encoder_forward(model, signal, chan_idx_use)
    sync()

    # Timed
    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        _ = run_encoder_forward(model, signal, chan_idx_use)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/checkpoint-small-300.pth")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    t0 = time.perf_counter()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = mae_vit_small_patch16()
    model.load_state_dict(ckpt["model"])
    model.eval()
    model = model.to(device)
    ms_load = (time.perf_counter() - t0) * 1000.0

    if not args.json:
        print(f"Model loaded in {ms_load:.0f} ms (device={device})", file=sys.stderr)

    # Standard channels (BCI Competition IV-2a equivalent: 22 channels)
    bci_channels = [25, 70, 18, 14, 34, 75, 20, 130, 0, 110, 10, 2, 127, 84, 85, 119, 46, 62, 99, 1, 117, 63]
    n_channels = len(bci_channels)
    n_samples = 768
    chan_idx = torch.tensor([bci_channels], dtype=torch.long, device=device)

    # Standard benchmark
    if not args.json:
        print(f"\n▸ Standard: {n_channels}ch × {n_samples} samples", file=sys.stderr)

    times = benchmark_config(model, n_channels, n_samples, chan_idx, args.warmup, args.runs, device)
    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5

    if not args.json:
        print(f"  mean={mean_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms  std={std_ms:.1f}ms", file=sys.stderr)

    # Channel scaling
    channel_counts = [4, 8, 16, 22, 32, 64]
    channel_scaling = []
    if not args.json:
        print(f"\n▸ Channel scaling (T={n_samples}):", file=sys.stderr)

    for nc in channel_counts:
        nc_idx = torch.arange(nc, device=device).unsqueeze(0)
        t_vec = benchmark_config(model, nc, n_samples, nc_idx, 1, max(3, args.runs // 2), device)
        avg = sum(t_vec) / len(t_vec)
        if not args.json:
            print(f"  {nc:>4}ch: {avg:.1f} ms", file=sys.stderr)
        channel_scaling.append({
            "channels": nc,
            "mean_ms": round(avg, 2),
            "min_ms": round(min(t_vec), 2),
            "max_ms": round(max(t_vec), 2),
            "runs": [round(t, 2) for t in t_vec],
        })

    # Sequence scaling
    sample_counts = [128, 256, 512, 768, 1024]
    seq_scaling = []
    if not args.json:
        print(f"\n▸ Sequence scaling (C={n_channels}):", file=sys.stderr)

    for ns in sample_counts:
        t_vec = benchmark_config(model, n_channels, ns, chan_idx, 1, max(3, args.runs // 2), device)
        avg = sum(t_vec) / len(t_vec)
        if not args.json:
            print(f"  {ns:>5} samples ({ns/128:.1f}s): {avg:.1f} ms", file=sys.stderr)
        seq_scaling.append({
            "samples": ns,
            "duration_s": round(ns / 128.0, 2),
            "mean_ms": round(avg, 2),
            "runs": [round(t, 2) for t in t_vec],
        })

    result = {
        "variant": "small",
        "backend": f"PyTorch {torch.__version__} ({args.device})",
        "config": {
            "embed_dim": 512,
            "depth": 8,
            "num_heads": 8,
            "patch_size": 16,
            "mlp_ratio": 4.0,
        },
        "load_ms": round(ms_load, 2),
        "inference": {
            "channels": n_channels,
            "samples": n_samples,
            "warmup": args.warmup,
            "runs": args.runs,
            "mean_ms": round(mean_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "std_ms": round(std_ms, 2),
            "all_ms": [round(t, 2) for t in times],
        },
        "channel_scaling": channel_scaling,
        "sequence_scaling": seq_scaling,
    }

    if args.json:
        print(json.dumps([result], indent=2))
    else:
        print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
