#!/usr/bin/env python3
"""
Generate comparison charts: Rust CPU vs Rust GPU vs Python CPU vs Python MPS.

Usage:
    python scripts/generate_charts.py bench_results/20260330_233322_apple_m4_pro_virtual/
"""

import json
import sys
from pathlib import Path

import numpy as np

# Try matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — will print text summary only", file=sys.stderr)

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench_results/latest")

# ── Load data ────────────────────────────────────────────────────────────────

def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

rust_cpu = load_json(run_dir / "cpu.json")
rust_gpu = load_json(run_dir / "gpu.json")
py_cpu   = load_json(run_dir / "python_cpu.json")
py_mps   = load_json(run_dir / "python_mps.json")

meta = load_json(run_dir / "meta.json") or {}
cpu_name = meta.get("cpu", "Unknown CPU")
date_str = meta.get("timestamp", "")

# Flatten: each JSON is a list with one entry (for the 'small' variant)
def first(d):
    return d[0] if d and isinstance(d, list) else d

rc = first(rust_cpu)
rg = first(rust_gpu)
pc = first(py_cpu)
pm = first(py_mps)

# ── Collect backends ─────────────────────────────────────────────────────────

backends = []
if rc: backends.append(("Rust CPU\n(Accelerate)", rc, "#4ecdc4"))
if pc: backends.append(("Python CPU\n(PyTorch)", pc, "#45b7d1"))
if rg: backends.append(("Rust GPU\n(Metal/MSL)", rg, "#ff6b6b"))
if pm: backends.append(("Python MPS\n(PyTorch)", pm, "#ffa07a"))

if not backends:
    print("No benchmark data found!")
    sys.exit(1)

# ── Text summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  ST-EEGFormer-Small Benchmark — {cpu_name}")
print(f"{'='*72}")
print(f"\n  Standard inference: 22ch × 768 samples (6s @ 128Hz)")
print(f"  {'Backend':<35} {'Mean':>8} {'Min':>8} {'Max':>8}")
print(f"  {'-'*63}")
for label, d, _ in backends:
    inf = d["inference"]
    label_1line = label.replace('\n', ' ')
    print(f"  {label_1line:<35} {inf['mean_ms']:>7.1f}  {inf['min_ms']:>7.1f}  {inf['max_ms']:>7.1f}")

# Speedup table
if rc:
    base = rc["inference"]["mean_ms"]
    print(f"\n  Speedup vs Rust CPU ({base:.1f} ms):")
    for label, d, _ in backends:
        ms = d["inference"]["mean_ms"]
        ratio = base / ms if ms > 0 else 0
        label_1line = label.replace('\n', ' ')
        print(f"    {label_1line:<35} {ratio:>5.1f}×")

if not HAS_MPL:
    sys.exit(0)

# ── Matplotlib style ─────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.6,
    "font.size":        11,
    "font.family":      "sans-serif",
})

# ── Chart 1: Inference Latency Bar Chart ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(backends))
width = 0.6

means = [d["inference"]["mean_ms"] for _, d, _ in backends]
stds  = [d["inference"]["std_ms"] for _, d, _ in backends]
colors = [c for _, _, c in backends]
labels = [l for l, _, _ in backends]

bars = ax.bar(x, means, width, yerr=stds, color=colors, alpha=0.85,
              edgecolor="white", linewidth=0.5, capsize=5)

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
            f"{m:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Inference Latency (ms)")
ax.set_title(f"ST-EEGFormer-Small — Inference Latency\n22ch × 768 samples (6s @ 128Hz) · {cpu_name}",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "inference_latency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  ✓ {run_dir}/inference_latency.png")

# ── Chart 2: Speedup vs Rust CPU ────────────────────────────────────────────

if rc:
    fig, ax = plt.subplots(figsize=(10, 6))
    base_ms = rc["inference"]["mean_ms"]
    speedups = [base_ms / d["inference"]["mean_ms"] for _, d, _ in backends]

    bars = ax.bar(x, speedups, width, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{s:.1f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(y=1.0, color="#e0e0e0", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup (× vs Rust CPU)")
    ax.set_title(f"ST-EEGFormer-Small — Speedup vs Rust CPU\n{cpu_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(run_dir / "speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {run_dir}/speedup.png")

# ── Chart 3: Channel Scaling ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
markers = ["o", "D", "s", "^"]
for i, (label, d, color) in enumerate(backends):
    cs = d.get("channel_scaling", [])
    if not cs:
        continue
    chans = [c["channels"] for c in cs]
    ms = [c["mean_ms"] for c in cs]
    label_1line = label.replace('\n', ' ')
    ax.plot(chans, ms, f"{markers[i % len(markers)]}-", color=color,
            linewidth=2, markersize=7, label=label_1line, zorder=3)

ax.set_xlabel("Number of EEG Channels")
ax.set_ylabel("Inference Latency (ms)")
ax.set_title(f"ST-EEGFormer-Small — Channel Scaling (T=768)\n{cpu_name}",
             fontsize=13, fontweight="bold")
ax.legend(framealpha=0.8, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)
fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "channel_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {run_dir}/channel_scaling.png")

# ── Chart 4: Sequence Scaling ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
for i, (label, d, color) in enumerate(backends):
    ss = d.get("sequence_scaling", [])
    if not ss:
        continue
    samples = [s["samples"] for s in ss]
    ms = [s["mean_ms"] for s in ss]
    label_1line = label.replace('\n', ' ')
    ax.plot(samples, ms, f"{markers[i % len(markers)]}-", color=color,
            linewidth=2, markersize=7, label=label_1line, zorder=3)

ax.set_xlabel("Segment Length (samples @ 128 Hz)")
ax.set_ylabel("Inference Latency (ms)")
ax.set_title(f"ST-EEGFormer-Small — Sequence Scaling (C=22)\n{cpu_name}",
             fontsize=13, fontweight="bold")
sec_ticks = [128, 256, 512, 768, 1024]
ax.set_xticks(sec_ticks)
ax.set_xticklabels([f"{s}\n({s/128:.0f}s)" for s in sec_ticks])
ax.legend(framealpha=0.8, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)
fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "sequence_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {run_dir}/sequence_scaling.png")

# ── Chart 5: Latency Distribution Box Plot ──────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
all_data = []
positions = []
box_colors = []
box_labels = []
pos = 1
for label, d, color in backends:
    all_ms = d["inference"].get("all_ms", [d["inference"]["mean_ms"]])
    all_data.append(all_ms)
    positions.append(pos)
    box_colors.append(color)
    box_labels.append(label)
    pos += 1

bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                showmeans=True, meanline=True,
                meanprops=dict(color="white", linewidth=1.5),
                medianprops=dict(color="yellow", linewidth=1.5),
                whiskerprops=dict(color="#e0e0e0"),
                capprops=dict(color="#e0e0e0"),
                flierprops=dict(markerfacecolor="#e0e0e0", markersize=4))
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels(box_labels, fontsize=10)
ax.set_ylabel("Latency (ms)")
ax.set_title(f"ST-EEGFormer-Small — Latency Distribution\n22ch × 768 (6s @ 128Hz) · {cpu_name}",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "latency_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {run_dir}/latency_distribution.png")

print(f"\n  All charts saved to {run_dir}/")
