# ST-EEGFormer-RS — GPU Performance TODO

## Current Status (2026-03-31)

| Backend | Mean | Std | Min |
|---|---|---|---|
| Python MPS (PyTorch 2.8, SDPA) | 19.17ms | 0.13ms | 19.03ms |
| Rust wgpu — before optimization | 38.06ms | 31.53ms | 7.90ms |
| Rust wgpu — after (stable) | 38.07ms | 0.29ms | 37.50ms |
| Rust CubeCL fused kernels | 28.35ms | 0.27ms | 28.05ms |

**Config:** ST-EEGFormer-small (D=512, depth=8, heads=8, patch=16), 22ch × 768 samples, Apple M4 Pro.

---

## Completed Optimizations

### 1. Variance elimination (31.53ms → 0.27ms std)
- Per-run GPU sync during warmup (10 runs) so autotune gets accurate timing data
- Full `into_data().to_vec::<f32>()` sync in timed runs (not `narrow` partial read)
- Default warmup increased from 3 → 10

### 2. 4D broadcast positional encoding addition
- **Before:** expand PE to `[B, seq_total, D]` via `expand()` + `reshape()` — each materializes a copy
- **After:** keep patch embeddings as `[B, Seq, Ch, D]`, add PE using 4D broadcasting:
  - `ch_emb: [B, 1, Ch, D]` broadcasts over Seq
  - `tp_emb: [1, Seq, 1, D]` broadcasts over B and Ch
  - Burn's fusion engine combines into a single kernel dispatch
- Eliminates 2 expand+reshape copy dispatches per forward pass

### 3. Fused CubeCL kernels (`--features wgpu-kernels`)
- **Softmax:** fixed 32× redundant global memory reads in max pass by using `plane_max` instead of per-lane full-row scan. For S=1057, 8 heads: saves ~275M reads (~1.1GB bandwidth).
- **Split QKV + bias + scale:** fused QKV bias addition into the split kernel. Saves 1 dispatch per block (8 total). Single kernel reads `[B, S, 3D]` + `[3D]` bias and writes Q, K, V as contiguous `[B, H, S, dh]` with Q pre-scaled.
- **Bias + GELU:** FC1 matmul done without bias, then `fused_bias_gelu` kernel adds bias and computes GELU in one pass. Saves 1 dispatch per block.
- **Bias + residual add:** attention projection matmul done without bias, then `fused_bias_residual_add` kernel computes `residual + matmul_out + proj_bias` in one pass. Saves 1 dispatch per block.
- **Add + LayerNorm:** fused residual addition with norm2 in encoder block. Computes `(a+b, layernorm(a+b))` in a single kernel with two outputs.
- **Manual matmul:** QKV, projection, FC1 bypass `burn::nn::Linear::forward()` (which uses `weight.unsqueeze()` + matmul + separate bias add) and instead do `reshape + matmul(weight)` directly. Avoids unnecessary unsqueeze dispatch on CubeBackend.

### 4. Optimized `channel_wise_normalize`
- Reduced intermediate tensor count from 7 to 4
- Reuse centered tensor, use `powf_scalar` + `powf_scalar(-0.5)` instead of separate `pow`, `sqrt`, `/`

---

## Remaining Gap: 28ms → 19ms (Python MPS)

### Root cause analysis via MPS profiling

Python benchmark uses `timm.models.vision_transformer.Block` which calls:
- `F.scaled_dot_product_attention()` — **fused flash attention**
- `F.layer_norm()` — single optimized Metal kernel
- `F.linear()` — Metal Performance Shaders GEMM
- `F.gelu()` — single Metal kernel

#### Per-block timing breakdown (MPS Python vs Rust CubeCL)

| Operation | Python MPS | Rust CubeCL | Gap |
|---|---|---|---|
| LayerNorm | 0.28ms | ~0.3ms | ~0ms |
| QKV matmul | 0.62ms | ~0.7ms | ~0.1ms |
| **Attention core** | **1.26ms (SDPA)** | **~1.5ms (3-op)** | **~0.3ms** |
| Merge heads | 0.28ms | ~0.2ms | ~0ms |
| Proj matmul | 0.37ms | ~0.3ms | ~0ms |
| LayerNorm | 0.29ms | ~0.3ms | ~0ms |
| FC1 matmul | 0.67ms | ~0.7ms | ~0.1ms |
| GELU | 0.34ms | ~0.2ms | ~0ms |
| FC2 matmul | 0.68ms | ~0.7ms | ~0.1ms |
| **Per-block total** | **~2.35ms** | **~3.5ms** | **~1.2ms** |
| **× 8 blocks** | **~18.8ms** | **~28ms** | **~9ms** |

### TODO #1: Flash Attention (SDPA) — ~5.6ms savings potential

**Priority: HIGH** · **Difficulty: HARD**

PyTorch's `F.scaled_dot_product_attention` on MPS fuses Q×K^T → softmax → attn×V into a **single Metal kernel** that never materializes the S×S attention matrix (34MB for S=1057, H=8).

**What we tried:**

1. **Naive per-row kernel (v1):** Each workgroup handled one (batch, head, query_row) with 32 SIMD lanes. **Result: 3× slower** (85ms) — each of 8,456 workgroups independently loaded all K/V (4.4GB total reads vs 68MB for decomposed GEMM).

2. **Tiled SharedMemory kernel (v2):** CubeDim=(32, 32, 1)=1024 threads. TILE_Q=32 query rows per workgroup. K/V tiles (32×dh) loaded cooperatively into `SharedMemory` and reused by all 32 query rows. `sync_cube()` barriers between tiles. **Result: still 3× slower** (83ms) because:
   - Online softmax requires 2× `exp()` per KV row per query row = 17.9M exp() calls total
   - Metal `exp()` ≈ 8 cycles → 17.9M × 5.3ns ≈ 95ms of ALU alone
   - Decomposed GEMM kernels use register tiling and run all S×S elements in parallel
   - For S=1057, the 34MB attention matrix fits in M4 Pro's 32MB L2 cache — no bandwidth bottleneck

**Key insight:** Flash attention trades memory bandwidth for compute. It's beneficial when S>4096 (S×S matrix overflows cache). For our S=1057, decomposed GEMM+softmax is faster because:
- Optimized GEMM has much better ALU utilization (register tiling, SIMD matrix multiply)
- Softmax processes rows fully in parallel (8456 workgroups × 33 elements/lane)
- No sequential online softmax overhead

**To beat Python SDPA for S=1057**, would need either:
- A flash attention kernel using Metal SIMD group matrix multiply intrinsics (`simdgroup_matrix`) for the Q×K^T tile matmul inside the kernel
- Or a graph-compilation approach (like MPSGraph) that fuses the three operations at the command-buffer level while keeping optimized GEMM sub-kernels

**Estimated savings if achieved:** 0.1ms/block × 8 = 0.8ms (measured SDPA vs decomposed on MPS Python: only 0.09ms/call savings).

### TODO #2: GEMM kernel tuning — ~2ms savings potential

**Priority: MEDIUM** · **Difficulty: MEDIUM**

MPS uses Apple's Metal Performance Shaders (MPS) GEMM kernels which are hand-tuned for Apple Silicon:
- Tiling strategies matched to M4 Pro's L1 (128KB per core) / L2 (shared) cache hierarchy
- Use of `simdgroup_matrix` intrinsics (hardware matrix multiply units)
- Optimal register pressure for the specific GPU architecture

Measured raw matmul timings (Python MPS vs what burn/cubecl likely achieves):

| Shape | MPS | Estimated burn |
|---|---|---|
| [1057,512] × [512,1536] | 0.66ms | ~0.8ms |
| [1057,512] × [512,512] | 0.39ms | ~0.5ms |
| [1057,512] × [512,2048] | 0.63ms | ~0.8ms |
| [1057,2048] × [2048,512] | 0.68ms | ~0.8ms |

**Options:**
- Verify burn's autotune is selecting the best GEMM kernel for these shapes
- Investigate if cubecl's GEMM supports `simdgroup_matrix` on Metal
- Consider using wgpu's native matmul dispatch (which may use MPS internally) instead of cubecl's GEMM when available
- Profile which autotune kernel variant is being selected for each shape

### TODO #3: Per-dispatch overhead reduction — ~1.5ms savings potential

**Priority: LOW** · **Difficulty: FRAMEWORK-LEVEL**

MPS command encoding overhead: **~3µs per op** (measured empirically: 1000 elementwise adds take 3.45ms).
wgpu/cubecl estimated overhead: **~50-90µs per dispatch** (derived from small-input scaling: 128 samples takes 8.9ms for ~100 dispatches).

For ~100 dispatches per forward pass:
- MPS overhead: ~0.3ms
- wgpu/cubecl overhead: ~5-9ms

This is fundamentally a burn/wgpu framework issue. Possible approaches:
- Investigate if burn wgpu can batch multiple kernels into fewer command buffer submissions
- Check if the `into_contiguous()` calls inside CubeCL kernel launchers are triggering unnecessary copy dispatches (inputs from previous kernels should already be contiguous)
- Profile actual Metal GPU timeline with Instruments to measure per-kernel launch latency

### TODO #4: Fuse remaining elementwise dispatches

**Priority: MEDIUM** · **Difficulty: EASY**

On CubeBackend (no JIT fusion), these operations are separate dispatches:
- FC2 bias add (from `burn::nn::Linear::forward`) — could bypass Linear and use `fused_bias_residual_add` for FC2 + final residual
- PE broadcasting adds in `forward_features_impl` — 2-3 dispatches for elementwise adds
- CLS token expand + cat — could pre-allocate output buffer

Estimated savings: 3-4 dispatches/block × 8 = 24-32 dispatches, ~2-3ms.

### TODO #5: f16 computation

**Priority: MEDIUM** · **Difficulty: EASY**

Apple Silicon GPUs have 2× throughput for f16 vs f32. The Python benchmark uses f32, so switching to f16 would be a net advantage. Burn wgpu supports f16 via the `Wgpu<half::f16, i32>` backend.

**Caveats:**
- Need to verify numerical accuracy with f16 (LayerNorm eps may need adjustment)
- All CubeCL kernels already have f16 codepaths
- Weight loading already converts from bf16/f16 safetensors

---

## Architecture Notes

### PyTorch MPS execution model

PyTorch MPS uses Apple's `MPSGraph` framework which:
1. Builds a computation graph of all operations
2. Compiles the graph into optimized Metal compute commands
3. Executes the entire graph in a single `MTLCommandBuffer` submission
4. Individual ops are encoded as `MTLComputeCommandEncoder` dispatches within the buffer
5. MPS automatically fuses compatible operations (elementwise chains, LayerNorm, etc.)
6. SDPA is a single `MPSGraph` operation that compiles to a tiled Metal kernel

### Burn wgpu execution model

Burn's `Wgpu` backend (standard):
1. Uses JIT fusion engine that automatically fuses elementwise operation chains
2. Each non-fusable operation (matmul, reduction) is a separate wgpu compute pass
3. Operations are batched into a command buffer and submitted on sync (`into_data()`)
4. Autotune selects the best kernel variant for each matmul shape

Burn's `CubeBackend` (used by `wgpu-kernels`):
1. **No JIT fusion** — every operation is a separate dispatch
2. Custom CubeCL kernels replace burn's decomposed ops (LayerNorm, softmax, GELU, etc.)
3. Fewer total dispatches due to fused kernels, but no elementwise fusion
4. Each `launch_*` function calls `into_contiguous()` which is a no-op if input is already contiguous

### Key shapes in the forward pass

```
B=1, C=22, T=768
num_patches = T/16 = 48
seq_total = 48 × 22 = 1056
S = 1057 (with CLS token)
D = 512, H = 8, dh = 64
MLP hidden = 2048

Matmul shapes per block (6 matmuls):
  QKV:    [1057, 512] × [512, 1536]   829M FLOPs
  Q×K^T:  [8, 1057, 64] × [8, 64, 1057]   1147M FLOPs
  attn×V: [8, 1057, 1057] × [8, 1057, 64]  1147M FLOPs
  proj:   [1057, 512] × [512, 512]    277M FLOPs
  FC1:    [1057, 512] × [512, 2048]   1108M FLOPs
  FC2:    [1057, 2048] × [2048, 512]  1108M FLOPs

Total: 5616M FLOPs/block × 8 blocks = 44.9 GFLOPs
M4 Pro GPU ~4.5 TFLOPS → theoretical minimum: 10ms
Python achieves 19ms (1.9× theoretical), Rust 28ms (2.8× theoretical)
```
