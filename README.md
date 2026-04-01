# steegformer-rs

**ST-EEGFormer** (Spatio-Temporal EEG Transformer) Foundation Model — inference in Rust with [Burn ML](https://burn.dev).

A pure-Rust implementation of the [ST-EEGFormer](https://github.com/LiuyinYang1101/STEEGFormer) model (KU Leuven), a ViT-based EEG foundation model pre-trained via Masked Autoencoder (MAE) reconstruction on raw EEG signals. Winner of the **NeurIPS 2025 EEG Foundation Challenge** (1st Place), accepted at **ICLR 2026**.

Numerical parity with the Python implementation is verified to **RMSE 0.000001** (Pearson r = 1.000000).

## Architecture

ST-EEGFormer uses a minimal, transparent ViT architecture with spatio-temporal patch tokenization:

```
EEG signal (B, C, T)   — C channels, T samples @ 128 Hz
    │
    ├─→ PatchEmbedEEG
    │     Unfold each channel into patches of 16 samples
    │     Linear(16, embed_dim)
    │     → (B, num_patches, C, embed_dim)
    │     → flatten → (B, num_patches × C, embed_dim)
    │
    ├─→ + Sinusoidal Temporal Positional Encoding (fixed, per time-patch)
    ├─→ + Learned Channel Embedding (nn.Embedding(145, D), per electrode)
    │
    ├─→ Prepend [CLS] token (+ temporal PE for position 0)
    │
    ├─→ N × Transformer Encoder Block
    │     Pre-norm: LayerNorm → Multi-Head Self-Attention → residual
    │               LayerNorm → FFN (Linear→GELU→Linear) → residual
    │
    └─→ LayerNorm → CLS token output (or global average pool)
              │
              └─→ (B, embed_dim)  — latent representation
```

### Key Design Choices

| Feature | Detail |
|---|---|
| **Input** | 128 Hz, ≤ 6s segments (768 samples), up to 142 channels |
| **Patch size** | 16 samples (0.125s temporal resolution) |
| **Temporal PE** | Fixed sinusoidal (generalizes to any segment length) |
| **Channel PE** | Learned embeddings (handles variable electrode montages) |
| **Pre-training** | MAE with 75% masking ratio |
| **Architecture** | Standard ViT (timm `Block` with qkv_bias) |

### Model Variants

| Variant | Params | Layers | Heads | embed_dim |
|---------|--------|--------|-------|-----------|
| ST-EEGFormer-Small | ~30M | 8 | 8 | 512 |
| ST-EEGFormer-Base  | ~86M | 12 | 12 | 768 |
| ST-EEGFormer-Large | ~300M | 24 | 16 | 1024 |

Pre-trained weights on [HuggingFace](https://huggingface.co/eugenehp/ST-EEGFormer) (safetensors) and [GitHub Releases](https://github.com/LiuyinYang1101/STEEGFormer/releases) (PyTorch .pth).

---

## Benchmarks

Inference benchmarks on **Apple M4 Pro** (10C/10T, 64GB RAM). All runs use ST-EEGFormer-Small (30M params), 22 EEG channels × 768 samples (6s @ 128Hz). PyTorch 2.8, Burn 0.20.

### Inference Latency

| Backend | Mean | Min | Std | vs Python CPU |
|---|---|---|---|---|
| Rust CPU (NdArray + Accelerate) | 507 ms | 503 ms | 2.6 ms | 6.3× slower |
| **Rust CPU Fused (NEON + BLAS + rayon)** | **89 ms** | **88 ms** | **1.1 ms** | **1.1× (near parity)** |
| Python CPU (PyTorch 2.8) | 80 ms | 78 ms | 1.2 ms | 1.0× |
| **Rust GPU (Burn wgpu + CubeCL → MSL)** | **30 ms** | **30 ms** | **0.2 ms** | **2.6× faster** |
| Python MPS (PyTorch 2.8 + Metal) | 20 ms | 19 ms | 0.3 ms | 4.1× faster |

### CPU: Rust Fused vs Python (near parity)

The fused CPU path bypasses Burn's tensor abstraction, operating directly on `&[f32]` slices with NEON SIMD, rayon parallelism, and direct `cblas_sgemm` calls. This eliminates ~1300 heap allocations per forward pass.

| Channels | Python CPU | Rust CPU Fused | Ratio |
|---|---|---|---|
| 4 | 21.9 ms | **17.4 ms** | **0.8× (Rust wins)** |
| 8 | 33.8 ms | 37.0 ms | 1.1× |
| 16 | 59.2 ms | 68.1 ms | 1.2× |
| 22 | 80.4 ms | 99.6 ms | 1.2× |
| 32 | 125.8 ms | 158.6 ms | 1.3× |
| 64 | 303.2 ms | 420.1 ms | 1.4× |

Rust wins at small channel counts due to lower dispatch overhead. At larger sizes, PyTorch's vDSP/AMX integration gives it the edge.

### GPU: Rust wgpu vs Python MPS

| Channels | Python MPS | Rust GPU | Ratio |
|---|---|---|---|
| 4 | 4.0 ms | 8.1 ms | 2.0× |
| 8 | 6.8 ms | 10.6 ms | 1.6× |
| 16 | 13.5 ms | 19.6 ms | 1.5× |
| 22 | 19.7 ms | 30.3 ms | 1.5× |
| 32 | 31.9 ms | 51.7 ms | 1.6× |
| 64 | 90.7 ms | 134.8 ms | 1.5× |

Python MPS is ~1.5× faster because it uses Apple's proprietary Metal Performance Shaders (GPU microcode), while Burn generates MSL shaders via CubeCL. The Rust GPU path is **158× more consistent** (std=0.2ms vs 31.5ms before optimization).

### Sequence Length Scaling

| Samples | Duration | Python CPU | Rust Fused | Python MPS | Rust GPU |
|---|---|---|---|---|---|
| 128 | 1.0s | 20.4 ms | **14.5 ms** | 3.6 ms | 7.7 ms |
| 256 | 2.0s | 30.9 ms | 32.6 ms | 6.0 ms | 10.4 ms |
| 512 | 4.0s | 54.8 ms | 62.4 ms | 11.9 ms | 17.4 ms |
| 768 | 6.0s | 81.2 ms | 100.8 ms | 19.5 ms | 31.3 ms |
| 1024 | 8.0s | 111.3 ms | 142.3 ms | 28.5 ms | 45.1 ms |

### Optimization Summary

| Optimization | Impact |
|---|---|
| **CPU: Fused kernels** (`cpu_fused.rs`) | 507ms → 89ms **(5.7× faster)** |
| Fused LayerNorm (2-pass NEON SIMD) | Replaces 5 tensor ops → 1 loop |
| Fused Softmax (3-pass) | Replaces 5 tensor ops → 1 loop |
| Fused GELU (libm::erff) | Replaces 3 tensor ops → 1 loop |
| Fused QKV split+scale | Replaces 10 tensor ops → 1 pass |
| Per-head attention (rayon parallel) | 8 heads processed in parallel |
| Direct cblas_sgemm | Bypasses Burn tensor wrapper |
| Pre-allocated ScratchBuffers | Eliminates ~1300 allocs/forward |
| **GPU: K-transpose in split_qkv** | Eliminates 8 copy dispatches |
| **GPU: Weight cache** | Eliminates ~80 Arc clones/forward |
| **GPU: Fast GELU** x·σ(1.702x) | 2.3× faster GELU kernel |
| **GPU: `#[comptime]` constant folding** | Modulo → bitwise AND |
| **GPU: 4-wide kernel unrolling** | 1.4× faster element-wise ops |
| **GPU: Cross-block norm fusion** | Saves 7 dispatches |
| **Total GPU** | 38ms → 30ms **(21% faster, 158× lower variance)** |

---

## Numerical Parity

Verified against the Python implementation at every stage:

| Stage | RMSE | Pearson r |
|---|---|---|
| Patch embedding | 0.000000 | 1.000000 |
| Channel embedding | 0.000000 | 1.000000 |
| Temporal encoding | 0.000000 | 1.000000 |
| After positional encoding | 0.000000 | 1.000000 |
| After CLS prepend | 0.000000 | 1.000000 |
| After transformer block 0 | 0.000004 | 1.000000 |
| **Full encoder (8 blocks)** | **0.000001** | **1.000000** |

---

## Quick Start

### Download weights

```bash
# From HuggingFace (recommended)
huggingface-cli download eugenehp/ST-EEGFormer \
    ST-EEGFormer_small_encoder.safetensors \
    config.json \
    --local-dir weights/
```

### Build

```bash
# CPU (default — NdArray + Rayon)
cargo build --release

# CPU with Apple Accelerate (macOS) — recommended for CPU inference
cargo build --release --features blas-accelerate

# GPU — Metal on macOS (with fused CubeCL kernels)
cargo build --release --no-default-features --features wgpu-kernels

# GPU — Metal on macOS (without custom kernels)
cargo build --release --no-default-features --features metal

# GPU — Vulkan on Linux
cargo build --release --no-default-features --features vulkan
```

### Generate test data

```bash
cargo run --bin gen_sample_eeg --release
```

### Run inference

```bash
cargo run --bin infer --release -- \
  --weights model.safetensors \
  --config config.json
```

### Extract embeddings

```bash
cargo run --example embed --release -- \
  --weights model.safetensors \
  --variant small
```

### Run benchmarks

```bash
# CPU benchmark (standard + fused)
cargo run --example benchmark_fused --release --features "ndarray,blas-accelerate"

# GPU benchmark
cargo run --example benchmark --release --no-default-features --features wgpu-kernels

# GPU kernel profiling
cargo run --example gpu_profile --release --no-default-features --features wgpu-kernels

# Full CPU vs GPU benchmark suite
./bench.sh small 3 10 model.safetensors

# Python comparison
python scripts/benchmark_python.py --checkpoint checkpoint.pth --device cpu
python scripts/benchmark_python.py --checkpoint checkpoint.pth --device mps
```

---

## API Usage

```rust
use steegformer::{STEEGFormerEncoder, ModelConfig, data};
use burn::backend::NdArray as B;
use std::path::Path;

let device = burn::backend::ndarray::NdArrayDevice::Cpu;

// Load model
let cfg = ModelConfig::small();
let (encoder, ms) = STEEGFormerEncoder::<B>::load_from_config(
    cfg,
    Path::new("model.safetensors"),
    device.clone(),
)?;

// Build input from channel names
let channels = &["Fz", "C3", "C4", "Pz"];
let signal = vec![0.0f32; channels.len() * 768];
let batch = data::build_batch_named::<B>(signal, channels, 768, &device);

// Get embeddings
let result = encoder.run_batch(&batch)?;
println!("Embedding: {:?}", result.shape);  // [512] for small
```

### Fused CPU Inference (fastest CPU path)

```rust
use steegformer::model::cpu_fused::{
    extract_model_weights, forward_fused, ScratchBuffers, channel_normalize,
};

// Extract weights once (after loading model)
let raw_weights = extract_model_weights(&steeg);

// Pre-allocate scratch buffers (reuse across calls)
let mut scratch = ScratchBuffers::new(seq_with_cls, dim, n_heads, ff_dim);

// Normalize signal
let mut signal = signal_data.clone();
channel_normalize(&mut signal, n_channels, n_samples);

// Run inference — no heap allocations, direct BLAS calls
let embedding = forward_fused(&signal, &channel_indices, &raw_weights, &mut scratch);
```

---

## Channel Vocabulary

ST-EEGFormer uses a 142-channel vocabulary covering the extended 10-20 system and BCI-specific electrodes. Common subsets are provided:

- `STANDARD_10_20` — 19 standard channels
- `BCI_COMP_IV_2A` — 22 motor imagery channels

See `src/channel_vocab.rs` for the full mapping.

---

## Project Structure

```
src/
├── lib.rs              — Public API and re-exports
├── config.rs           — ModelConfig (small/base/large), DataConfig
├── channel_vocab.rs    — 142-channel vocabulary mapping
├── data.rs             — InputBatch construction, z-score normalization
├── encoder.rs          — High-level STEEGFormerEncoder
├── weights.rs          — Safetensors weight loading
├── model/
│   ├── steegformer.rs  — Full encoder model + WeightCache
│   ├── patch_embed.rs  — EEG patch embedding (unfold + linear)
│   ├── positional.rs   — Sinusoidal temporal + learned channel PE
│   ├── attention.rs    — Multi-head self-attention (qkv_bias)
│   ├── encoder_block.rs — Pre-norm transformer block
│   ├── feedforward.rs  — FFN (Linear→GELU→Linear)
│   ├── norm.rs         — LayerNorm wrapper
│   ├── fused.rs        — FusedOps trait + backend dispatch
│   ├── cpu_fused.rs    — Fused CPU kernels (NEON SIMD + BLAS)
│   └── kernels.rs      — CubeCL GPU kernels (Metal MSL)
├── bin/
│   ├── infer.rs        — CLI inference
│   ├── gen_sample_eeg.rs — Generate synthetic EEG CSV
│   └── safetensors_info.rs — Inspect weight files
scripts/
├── export_parity_vectors.py — Export Python reference tensors
├── benchmark_python.py      — Python inference benchmark
└── generate_charts.py       — Generate comparison charts
examples/
├── embed.rs            — Embedding extraction
├── reconstruct.rs      — Forward pass test
├── benchmark.rs        — GPU/CPU inference latency benchmark
├── benchmark_fused.rs  — Fused CPU benchmark (vs standard path)
├── gpu_profile.rs      — Per-kernel GPU profiling
├── matmul_test.rs      — GPU matmul strategy comparison
└── dump_msl.rs         — MSL shader source inspection
tests/
├── forward_pass.rs     — Smoke tests (5 tests)
└── python_parity.rs    — Numerical parity tests (5 tests)
```

---

## References

- **Paper**: [Are EEG Foundation Models Worth It?](https://openreview.net/forum?id=5Xwm8e6vbh) (ICLR 2026)
- **Original code**: [LiuyinYang1101/STEEGFormer](https://github.com/LiuyinYang1101/STEEGFormer)
- **MAE**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- **Burn ML**: [burn.dev](https://burn.dev)

## License

Apache-2.0
