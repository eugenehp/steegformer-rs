---
license: mit
tags:
  - eeg
  - bci
  - brain-computer-interface
  - foundation-model
  - vit
  - masked-autoencoder
  - mae
  - neuroscience
  - safetensors
  - burn
  - rust
language:
  - en
library_name: steegformer
pipeline_tag: feature-extraction
---

# ST-EEGFormer — Safetensors Weights

Pre-converted [safetensors](https://github.com/huggingface/safetensors) weights for the [ST-EEGFormer](https://github.com/LiuyinYang1101/STEEGFormer) EEG foundation model, ready for use with **[steegformer](https://github.com/eugenehp/steegformer-rs)** (pure-Rust inference on [Burn 0.20](https://burn.dev)) or any framework that supports safetensors.

Weights are converted from the official PyTorch `.pth` checkpoints published at [LiuyinYang1101/STEEGFormer](https://github.com/LiuyinYang1101/STEEGFormer/releases).

ST-EEGFormer won **1st Place** in the NeurIPS 2025 EEG Foundation Challenge and was accepted at **ICLR 2026**.

## Model Files

### Encoder Only (for inference / embedding extraction)

| File | Variant | Params | Size | Layers | Heads | embed_dim |
|------|---------|--------|------|--------|-------|-----------|
| [`ST-EEGFormer_small_encoder.safetensors`](ST-EEGFormer_small_encoder.safetensors) | **Small** | 25.6 M | 102 MB | 8 | 8 | 512 |
| [`ST-EEGFormer_base_encoder.safetensors`](ST-EEGFormer_base_encoder.safetensors) | **Base** | 85.6 M | 342 MB | 12 | 12 | 768 |
| [`ST-EEGFormer_large_encoder.safetensors`](ST-EEGFormer_large_encoder.safetensors) | **Large** | 303.0 M | 1,212 MB | 24 | 16 | 1024 |
| [`ST-EEGFormer_largeV2_encoder.safetensors`](ST-EEGFormer_largeV2_encoder.safetensors) | **Large V2** | 303.1 M | 1,212 MB | 24 | 16 | 1024 |

### Full MAE (encoder + decoder, for reconstruction / fine-tuning)

| File | Variant | Params | Size | Decoder dim | Decoder depth |
|------|---------|--------|------|-------------|---------------|
| [`ST-EEGFormer_small_mae.safetensors`](ST-EEGFormer_small_mae.safetensors) | **Small** | 33.1 M | 132 MB | 384 | 4 |
| [`ST-EEGFormer_base_mae.safetensors`](ST-EEGFormer_base_mae.safetensors) | **Base** | 111.5 M | 446 MB | 512 | 8 |
| [`ST-EEGFormer_large_mae.safetensors`](ST-EEGFormer_large_mae.safetensors) | **Large** | 329.1 M | 1,316 MB | 512 | 8 |
| [`ST-EEGFormer_largeV2_mae.safetensors`](ST-EEGFormer_largeV2_mae.safetensors) | **Large V2** | 329.3 M | 1,317 MB | 512 | 8 |

### Config

| File | Description |
|------|-------------|
| [`config.json`](config.json) | Model hyperparameters for all variants |

> **Large V2** has undergone further pre-training on the HBN dataset for the NeurIPS 2025 EEG Foundation Challenge.

## Quick Start — Rust

```bash
# Install
cargo add steegformer

# Download weights
huggingface-cli download eugenehp/ST-EEGFormer \
    ST-EEGFormer_small_encoder.safetensors \
    config.json \
    --local-dir weights/

# Run inference
cargo run --release --bin infer -- \
    --config weights/config.json \
    --weights weights/ST-EEGFormer_small_encoder.safetensors
```

### Library API

```rust
use steegformer::{STEEGFormerEncoder, ModelConfig, data};
use std::path::Path;

// Load model
let cfg = ModelConfig::small();
let (encoder, _ms) = STEEGFormerEncoder::<B>::load_from_config(
    cfg,
    Path::new("ST-EEGFormer_small_encoder.safetensors"),
    device,
)?;

// Build input: 4 channels × 6 seconds @ 128 Hz
let channels = &["Fz", "C3", "C4", "Pz"];
let signal = vec![0.0f32; channels.len() * 768];
let batch = data::build_batch_named::<B>(signal, channels, 768, &device);

// Extract embeddings
let result = encoder.run_batch(&batch)?;
println!("Embedding shape: {:?}", result.shape);  // [512]
```

## Quick Start — Python

```python
from safetensors.torch import load_file

# Load encoder weights
state_dict = load_file("ST-EEGFormer_small_encoder.safetensors")

# Build model and load
from models_mae_eeg import mae_vit_small_patch16
model = mae_vit_small_patch16()
model.load_state_dict(state_dict, strict=False)
model.eval()
```

## Architecture

```
EEG signal (B, C, T)  — up to 142 channels, 128 Hz, ≤ 6s
    │
    ▼
┌──────────────────────────────────────┐
│  PatchEmbedEEG                       │
│  Unfold → 16-sample patches          │
│  Linear(16, embed_dim)               │
│  → (B, num_patches × C, D)           │
└──────────────────────────────────────┘
    │
    + Sinusoidal Temporal PE (fixed)
    + Learned Channel Embedding (nn.Embedding(145, D))
    │
    ▼
┌──────────────────────────────────────┐
│  [CLS] token prepend                 │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  N × Transformer Encoder Block       │
│  Pre-norm: LN → MHSA → residual     │
│            LN → FFN  → residual     │
│  (qkv_bias=True, GELU activation)   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  LayerNorm → CLS token              │
│  → (B, embed_dim) embedding         │
└──────────────────────────────────────┘
```

### MAE Pre-training (decoder, included in `*_mae.safetensors`)

```
Encoder output (25% of tokens)
    │
    ▼
  Linear(embed_dim → decoder_dim)
  + Insert mask tokens at masked positions
  + Decoder temporal/channel PE
    │
    ▼
  M × Decoder Transformer Blocks
    │
    ▼
  Linear(decoder_dim → patch_size)
  → Reconstructed EEG patches
```

## Numerical Parity (Rust vs Python)

Verified at every stage against the official PyTorch implementation:

| Stage | RMSE | Pearson r |
|---|---|---|
| Patch embedding | 0.000000 | 1.000000 |
| Channel embedding | 0.000000 | 1.000000 |
| Temporal encoding | 0.000000 | 1.000000 |
| After positional encoding | 0.000000 | 1.000000 |
| After transformer block 0 | 0.000004 | 1.000000 |
| **Full encoder (8 blocks)** | **0.000001** | **1.000000** |

## Benchmarks

**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)

### Inference Latency — ST-EEGFormer-Small (22ch × 768 samples)

| Backend | Mean | Min |
|---|---|---|
| Rust CPU (NdArray + Accelerate) | 608.4 ms | 601.4 ms |
| Python CPU (PyTorch 2.6) | 78.1 ms | 77.2 ms |
| **Rust GPU (Burn wgpu + Metal)** | **38.1 ms** | **7.9 ms** |
| Python MPS (PyTorch + Metal) | 19.2 ms | 19.0 ms |

### Channel Scaling (T=768)

| Channels | Rust CPU | Python CPU | Rust GPU | Python MPS |
|---|---|---|---|---|
| 4 | 75.5 ms | 21.8 ms | 11.5 ms | 4.0 ms |
| 22 | 596.0 ms | 77.9 ms | 32.7 ms | 19.3 ms |
| 64 | 3853.2 ms | 301.9 ms | 119.4 ms | 90.1 ms |

## Weight Key Format

### Encoder keys

```
patch_embed.proj.weight             [embed_dim, 16]
patch_embed.proj.bias               [embed_dim]
cls_token                           [1, 1, embed_dim]
enc_channel_emd.channel_transformation.weight  [145, embed_dim]
enc_temporal_emd.pe                 [1, 512, embed_dim]
blocks.{i}.norm1.weight             [embed_dim]
blocks.{i}.norm1.bias               [embed_dim]
blocks.{i}.attn.qkv.weight         [3*embed_dim, embed_dim]
blocks.{i}.attn.qkv.bias           [3*embed_dim]
blocks.{i}.attn.proj.weight        [embed_dim, embed_dim]
blocks.{i}.attn.proj.bias          [embed_dim]
blocks.{i}.norm2.weight             [embed_dim]
blocks.{i}.norm2.bias               [embed_dim]
blocks.{i}.mlp.fc1.weight          [4*embed_dim, embed_dim]
blocks.{i}.mlp.fc1.bias            [4*embed_dim]
blocks.{i}.mlp.fc2.weight          [embed_dim, 4*embed_dim]
blocks.{i}.mlp.fc2.bias            [embed_dim]
norm.weight                         [embed_dim]
norm.bias                           [embed_dim]
```

### Decoder keys (MAE only)

```
decoder_embed.weight                [dec_dim, embed_dim]
decoder_embed.bias                  [dec_dim]
mask_token                          [1, 1, dec_dim]
dec_channel_emd.channel_transformation.weight  [145, dec_dim]
dec_temporal_emd.pe                 [1, 512, dec_dim]
decoder_blocks.{i}.*               (same structure as encoder)
decoder_norm.weight                 [dec_dim]
decoder_norm.bias                   [dec_dim]
decoder_pred.weight                 [16, dec_dim]
decoder_pred.bias                   [16]
```

## Conversion

These weights were converted from the official `.pth` files:

```python
import torch
from safetensors.torch import save_file

ckpt = torch.load("checkpoint.pth", map_location="cpu", weights_only=False)
state_dict = ckpt["model"]

# Encoder only
encoder = {k: v.float().contiguous() for k, v in state_dict.items()
           if any(k.startswith(p) for p in
                  ["patch_embed.", "cls_token", "enc_", "blocks.", "norm."])}
save_file(encoder, "encoder.safetensors")
```

Or use the included conversion script:

```bash
python scripts/convert_to_safetensors.py --all
```

## Citation

```bibtex
@inproceedings{yang2026_steegformer,
  title={Are {EEG} Foundation Models Worth It? Comparative Evaluation
         with Traditional Decoders in Diverse {BCI} Tasks},
  author={Liuyin Yang and Qiang Sun and Ang Li and Marc M. Van Hulle},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=5Xwm8e6vbh}
}
```

## License

MIT — same as the original ST-EEGFormer release.

## Links

| | |
|---|---|
| **Rust crate** | [github.com/eugenehp/steegformer](https://github.com/eugenehp/steegformer-rs) |
| **Original code** | [github.com/LiuyinYang1101/STEEGFormer](https://github.com/LiuyinYang1101/STEEGFormer) |
| **Original weights** | [GitHub Releases](https://github.com/LiuyinYang1101/STEEGFormer/releases) |
| **Paper** | [OpenReview (ICLR 2026)](https://openreview.net/forum?id=5Xwm8e6vbh) |
| **Burn framework** | [burn.dev](https://burn.dev) |
