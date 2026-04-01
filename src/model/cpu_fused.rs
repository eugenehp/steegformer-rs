/// Fused CPU kernels for ST-EEGFormer inference.
///
/// These bypass Burn's tensor abstraction for element-wise operations,
/// operating directly on contiguous `&[f32]` / `&mut [f32]` slices.
///
/// Key optimizations vs Burn's standard NdArray ops:
/// - **LayerNorm**: 2-pass (mean+var → normalize) vs 5+ separate tensor ops
/// - **Softmax**: 3-pass (max → exp+sum → normalize) vs 5+ tensor ops
/// - **GELU**: 1-pass fused vs 3+ tensor ops
/// - **Bias+Residual+LayerNorm**: fused into 2 passes
///
/// Each eliminated tensor op saves: 1 heap alloc + 1 full data pass + metadata copy.
///
/// On Apple M4 Pro, this eliminates ~300+ tensor allocations per forward pass.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use rayon::prelude::*;

// ─── LayerNorm ────────────────────────────────────────────────────────────────

/// Fused LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
///
/// 2-pass algorithm:
///   Pass 1: compute mean and variance (Welford's online algorithm)
///   Pass 2: normalize, scale, shift
///
/// Input layout: [rows, cols] row-major, weight/bias: [cols]
/// Output written to `out` (may alias `x`).
#[inline]
pub fn layernorm(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    debug_assert_eq!(weight.len(), cols);
    debug_assert_eq!(bias.len(), cols);

    let inv_n = 1.0 / cols as f32;

    // Parallel if large enough (S=1057, D=512 → 541K elements)
    if rows * cols > 100_000 {
        out.par_chunks_mut(cols)
            .zip(x.par_chunks(cols))
            .for_each(|(o_row, x_row)| {
                let (mean, var) = mean_var_f32(x_row, inv_n);
                let inv_std = 1.0 / (var + eps).sqrt();
                #[cfg(target_arch = "aarch64")]
                {
                    layernorm_row_neon(o_row, x_row, weight, bias, mean, inv_std, cols);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    for j in 0..cols {
                        o_row[j] = (x_row[j] - mean) * inv_std * weight[j] + bias[j];
                    }
                }
            });
    } else {
        for row in 0..rows {
            let off = row * cols;
            let x_row = &x[off..off + cols];
            let o_row = &mut out[off..off + cols];
            let (mean, var) = mean_var_f32(x_row, inv_n);
            let inv_std = 1.0 / (var + eps).sqrt();
            #[cfg(target_arch = "aarch64")]
            {
                layernorm_row_neon(o_row, x_row, weight, bias, mean, inv_std, cols);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for j in 0..cols {
                    o_row[j] = (x_row[j] - mean) * inv_std * weight[j] + bias[j];
                }
            }
        }
    }
}

/// Compute mean and variance in a single pass.
#[inline]
fn mean_var_f32(x: &[f32], inv_n: f32) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        mean_var_neon(x, inv_n)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for &v in x {
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum * inv_n;
        let var = sum_sq * inv_n - mean * mean;
        (mean, var)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn mean_var_neon(x: &[f32], inv_n: f32) -> (f32, f32) {
    unsafe {
        let mut sum_v = vdupq_n_f32(0.0);
        let mut sq_v = vdupq_n_f32(0.0);
        let chunks = x.len() / 4;
        let remainder = x.len() % 4;

        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            sum_v = vaddq_f32(sum_v, v);
            sq_v = vfmaq_f32(sq_v, v, v);
        }

        let mut sum = vaddvq_f32(sum_v);
        let mut sum_sq = vaddvq_f32(sq_v);

        for i in 0..remainder {
            let v = x[chunks * 4 + i];
            sum += v;
            sum_sq += v * v;
        }

        let mean = sum * inv_n;
        let var = sum_sq * inv_n - mean * mean;
        (mean, var)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn layernorm_row_neon(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    mean: f32,
    inv_std: f32,
    cols: usize,
) {
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let inv_std_v = vdupq_n_f32(inv_std);
        let chunks = cols / 4;
        let remainder = cols % 4;

        for i in 0..chunks {
            let off = i * 4;
            let xv = vld1q_f32(x.as_ptr().add(off));
            let wv = vld1q_f32(weight.as_ptr().add(off));
            let bv = vld1q_f32(bias.as_ptr().add(off));
            let centered = vsubq_f32(xv, mean_v);
            let normed = vmulq_f32(centered, inv_std_v);
            let result = vfmaq_f32(bv, normed, wv);
            vst1q_f32(out.as_mut_ptr().add(off), result);
        }

        for i in 0..remainder {
            let j = chunks * 4 + i;
            out[j] = (x[j] - mean) * inv_std * weight[j] + bias[j];
        }
    }
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// In-place softmax along the last dimension.
///
/// data layout: [outer, inner] row-major
/// 3-pass: max → exp(x-max) + sum → normalize
#[inline]
pub fn softmax_inplace(data: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(data.len(), rows * cols);

    for row in 0..rows {
        let off = row * cols;
        let slice = &mut data[off..off + cols];

        // Pass 1: find max
        let max_val = slice_max(slice);

        // Pass 2: exp(x - max) and sum
        let sum;
        #[cfg(target_arch = "aarch64")]
        {
            sum = softmax_exp_sum_neon(slice, max_val);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut s = 0.0f32;
            for v in slice.iter_mut() {
                *v = (*v - max_val).exp();
                s += *v;
            }
            sum = s;
        }

        // Pass 3: normalize
        let inv_sum = 1.0 / sum;
        scale_inplace(slice, inv_sum);
    }
}

#[inline]
fn slice_max(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let chunks = x.len() / 4;
            let remainder = x.len() % 4;
            let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = vld1q_f32(x.as_ptr().add(i * 4));
                max_v = vmaxq_f32(max_v, v);
            }
            let mut m = vmaxvq_f32(max_v);
            for i in 0..remainder {
                m = m.max(x[chunks * 4 + i]);
            }
            m
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn softmax_exp_sum_neon(data: &mut [f32], max_val: f32) -> f32 {
    // Fast exp: For softmax, we can use a fast approximation since
    // the relative error is what matters (normalized away by division).
    // But for numerical parity with Python, use standard exp.
    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    sum
}

#[inline]
fn scale_inplace(data: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let scale_v = vdupq_n_f32(scale);
            let chunks = data.len() / 4;
            let remainder = data.len() % 4;
            for i in 0..chunks {
                let off = i * 4;
                let v = vld1q_f32(data.as_ptr().add(off));
                vst1q_f32(data.as_mut_ptr().add(off), vmulq_f32(v, scale_v));
            }
            for i in 0..remainder {
                data[chunks * 4 + i] *= scale;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in data.iter_mut() {
            *v *= scale;
        }
    }
}

// ─── GELU ─────────────────────────────────────────────────────────────────────

/// In-place GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
///
/// Uses the exact GELU formula to maintain numerical parity with Python.
#[inline]
pub fn gelu_inplace(data: &mut [f32]) {
    let inv_sqrt2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in data.iter_mut() {
        let x = *v;
        *v = x * 0.5 * (1.0 + libm::erff(x * inv_sqrt2));
    }
}

/// Fused bias + GELU: out[i] = GELU(x[i] + bias[i % cols])
///
/// Parallelized across rows using rayon for large tensors.
#[inline]
pub fn bias_gelu_inplace(data: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(data.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);
    let inv_sqrt2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    // Parallel if large enough (ff_dim=2048, S=1057 → 2.16M elements)
    if rows * cols > 100_000 {
        data.par_chunks_mut(cols).for_each(|row| {
            for j in 0..cols {
                let x = row[j] + bias[j];
                row[j] = x * 0.5 * (1.0 + libm::erff(x * inv_sqrt2));
            }
        });
    } else {
        for row in 0..rows {
            let off = row * cols;
            for j in 0..cols {
                let x = data[off + j] + bias[j];
                data[off + j] = x * 0.5 * (1.0 + libm::erff(x * inv_sqrt2));
            }
        }
    }
}

// ─── Fused Bias + Residual Add ────────────────────────────────────────────────

/// out = residual + matmul_out + bias (broadcast bias over rows)
#[inline]
pub fn bias_residual_add(
    out: &mut [f32],
    residual: &[f32],
    matmul_out: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(out.len(), rows * cols);
    debug_assert_eq!(residual.len(), rows * cols);
    debug_assert_eq!(matmul_out.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);

    // Use rayon for large tensors
    if rows * cols > 100_000 {
        out.par_chunks_mut(cols)
            .zip(residual.par_chunks(cols))
            .zip(matmul_out.par_chunks(cols))
            .for_each(|((o_row, r_row), m_row)| {
                bias_residual_add_row(o_row, r_row, m_row, bias, cols);
            });
    } else {
        for row in 0..rows {
            let off = row * cols;
            bias_residual_add_row(
                &mut out[off..off + cols],
                &residual[off..off + cols],
                &matmul_out[off..off + cols],
                bias, cols,
            );
        }
    }
}

#[inline]
fn bias_residual_add_row(
    out: &mut [f32],
    residual: &[f32],
    matmul_out: &[f32],
    bias: &[f32],
    cols: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let chunks = cols / 4;
            let remainder = cols % 4;
            for i in 0..chunks {
                let j = i * 4;
                let r = vld1q_f32(residual.as_ptr().add(j));
                let m = vld1q_f32(matmul_out.as_ptr().add(j));
                let b = vld1q_f32(bias.as_ptr().add(j));
                let sum = vaddq_f32(vaddq_f32(r, m), b);
                vst1q_f32(out.as_mut_ptr().add(j), sum);
            }
            for i in 0..remainder {
                let j = chunks * 4 + i;
                out[j] = residual[j] + matmul_out[j] + bias[j];
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..cols {
            out[j] = residual[j] + matmul_out[j] + bias[j];
        }
    }
}

/// Fused: sum = residual + matmul_out + bias; norm = layernorm(sum)
/// Writes sum to `sum_out` and norm to `norm_out`.
#[inline]
pub fn bias_residual_add_layernorm(
    sum_out: &mut [f32],
    norm_out: &mut [f32],
    residual: &[f32],
    matmul_out: &[f32],
    bias: &[f32],
    ln_weight: &[f32],
    ln_bias: &[f32],
    eps: f32,
    rows: usize,
    cols: usize,
) {
    // Step 1: compute sum
    bias_residual_add(sum_out, residual, matmul_out, bias, rows, cols);
    // Step 2: layernorm the sum
    layernorm(norm_out, sum_out, ln_weight, ln_bias, eps, rows, cols);
}

// ─── QKV Split + Scale ───────────────────────────────────────────────────────

/// Split QKV tensor [S, 3*D] into Q, K, V each [H, S, dh] and scale Q.
///
/// Input: qkv_biased [S, 3*D] — already has bias added
/// Output: q [H, S, dh], k [H, S, dh], v [H, S, dh] — contiguous per-head layout
///
/// This replaces: 3×narrow + 3×reshape + 3×swap_dims + 1×mul_scalar = 10 tensor ops
/// with a single pass that writes directly into the transposed layout.
#[inline]
pub fn split_qkv_scaled(
    q_out: &mut [f32],   // [H, S, dh]
    k_out: &mut [f32],   // [H, S, dh]
    v_out: &mut [f32],   // [H, S, dh]
    qkv: &[f32],         // [S, 3*D]
    bias: &[f32],        // [3*D]
    n_heads: usize,
    head_dim: usize,
    scale: f32,
    seq_len: usize,
) {
    let dim = n_heads * head_dim;
    debug_assert_eq!(qkv.len(), seq_len * 3 * dim);
    debug_assert_eq!(bias.len(), 3 * dim);

    for s in 0..seq_len {
        let qkv_row = s * 3 * dim;
        for h in 0..n_heads {
            let head_off = h * head_dim;
            let out_off = h * seq_len * head_dim + s * head_dim;

            for d in 0..head_dim {
                // Q: bias + scale
                q_out[out_off + d] = (qkv[qkv_row + head_off + d] + bias[head_off + d]) * scale;
                // K: bias only
                k_out[out_off + d] = qkv[qkv_row + dim + head_off + d] + bias[dim + head_off + d];
                // V: bias only
                v_out[out_off + d] = qkv[qkv_row + 2 * dim + head_off + d] + bias[2 * dim + head_off + d];
            }
        }
    }
}

/// Merge heads: [H, S, dh] → [S, D] (transpose and interleave)
#[inline]
pub fn merge_heads(
    out: &mut [f32],    // [S, D]
    heads: &[f32],      // [H, S, dh]
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
) {
    let dim = n_heads * head_dim;
    debug_assert_eq!(out.len(), seq_len * dim);
    debug_assert_eq!(heads.len(), n_heads * seq_len * head_dim);

    for s in 0..seq_len {
        for h in 0..n_heads {
            let src_off = h * seq_len * head_dim + s * head_dim;
            let dst_off = s * dim + h * head_dim;
            out[dst_off..dst_off + head_dim]
                .copy_from_slice(&heads[src_off..src_off + head_dim]);
        }
    }
}

// ─── Scratch Buffer ──────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for transformer forward pass.
/// Reused across blocks to avoid per-op heap allocation.
pub struct ScratchBuffers {
    // Transformer block buffers
    pub norm_out: Vec<f32>,     // [S, D] — LayerNorm output
    pub qkv: Vec<f32>,         // [S, 3*D] — QKV projection output
    pub q: Vec<f32>,            // [H, S, dh] — split Q
    pub k: Vec<f32>,            // [H, S, dh] — split K
    pub v: Vec<f32>,            // [H, S, dh] — split V
    pub attn: Vec<f32>,         // [H, S, S] — attention scores
    pub attn_out: Vec<f32>,     // [H, S, dh] — attention output
    pub merged: Vec<f32>,       // [S, D] — merged heads
    pub proj_out: Vec<f32>,     // [S, D] — projection output
    pub residual: Vec<f32>,     // [S, D] — residual sum
    pub ffn_hidden: Vec<f32>,   // [S, ff_dim] — FFN hidden
    pub ffn_out: Vec<f32>,      // [S, D] — FFN output
}

impl ScratchBuffers {
    pub fn new(max_seq: usize, dim: usize, n_heads: usize, ff_dim: usize) -> Self {
        let head_dim = dim / n_heads;
        Self {
            norm_out: vec![0.0; max_seq * dim],
            qkv: vec![0.0; max_seq * 3 * dim],
            q: vec![0.0; n_heads * max_seq * head_dim],
            k: vec![0.0; n_heads * max_seq * head_dim],
            v: vec![0.0; n_heads * max_seq * head_dim],
            attn: vec![0.0; n_heads * max_seq * max_seq],
            attn_out: vec![0.0; n_heads * max_seq * head_dim],
            merged: vec![0.0; max_seq * dim],
            proj_out: vec![0.0; max_seq * dim],
            residual: vec![0.0; max_seq * dim],
            ffn_hidden: vec![0.0; max_seq * ff_dim],
            ffn_out: vec![0.0; max_seq * dim],
        }
    }

    /// Resize buffers for actual sequence length (no realloc if smaller).
    pub fn resize(&mut self, seq: usize, dim: usize, n_heads: usize, ff_dim: usize) {
        let head_dim = dim / n_heads;
        resize_vec(&mut self.norm_out, seq * dim);
        resize_vec(&mut self.qkv, seq * 3 * dim);
        resize_vec(&mut self.q, n_heads * seq * head_dim);
        resize_vec(&mut self.k, n_heads * seq * head_dim);
        resize_vec(&mut self.v, n_heads * seq * head_dim);
        resize_vec(&mut self.attn, n_heads * seq * seq);
        resize_vec(&mut self.attn_out, n_heads * seq * head_dim);
        resize_vec(&mut self.merged, seq * dim);
        resize_vec(&mut self.proj_out, seq * dim);
        resize_vec(&mut self.residual, seq * dim);
        resize_vec(&mut self.ffn_hidden, seq * ff_dim);
        resize_vec(&mut self.ffn_out, seq * dim);
    }
}

#[inline]
fn resize_vec(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

// ─── Batched MatMul using Accelerate BLAS ────────────────────────────────────

#[cfg(feature = "blas-accelerate")]
extern "C" {
    fn cblas_sgemm(
        order: i32,    // CblasRowMajor = 101
        transa: i32,   // CblasNoTrans = 111, CblasTrans = 112
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// C = alpha * A × B + beta * C
/// A: [M, K], B: [K, N], C: [M, N]   (row-major)
#[cfg(feature = "blas-accelerate")]
#[inline]
pub fn sgemm(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    unsafe {
        cblas_sgemm(
            101,  // CblasRowMajor
            111,  // CblasNoTrans
            111,  // CblasNoTrans
            m as i32, n as i32, k as i32,
            alpha,
            a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            beta,
            c.as_mut_ptr(), n as i32,
        );
    }
}

/// C = alpha * A × B^T + beta * C
/// A: [M, K], B: [N, K] (B stored row-major, transposed for multiply), C: [M, N]
#[cfg(feature = "blas-accelerate")]
#[inline]
pub fn sgemm_at(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    unsafe {
        cblas_sgemm(
            101,  // CblasRowMajor
            111,  // CblasNoTrans
            112,  // CblasTrans
            m as i32, n as i32, k as i32,
            alpha,
            a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,  // B is [N, K], ldb = K
            beta,
            c.as_mut_ptr(), n as i32,
        );
    }
}

// ─── Full Transformer Block (fused CPU) ──────────────────────────────────────

/// Execute one transformer encoder block entirely on raw f32 slices.
///
/// x: [S, D] (in/out) — updated in-place
/// scratch: pre-allocated buffers
///
/// Replaces ~56 Burn tensor ops with ~6 BLAS calls + fused element-wise ops.
#[cfg(feature = "blas-accelerate")]
pub fn transformer_block_fused(
    x: &mut [f32],          // [S, D] in/out
    scratch: &mut ScratchBuffers,
    // Block weights (all pre-extracted as contiguous f32 slices)
    norm1_weight: &[f32],   // [D]
    norm1_bias: &[f32],     // [D]
    norm1_eps: f32,
    qkv_weight: &[f32],    // [D, 3*D] (already transposed for row-major matmul)
    qkv_bias: &[f32],      // [3*D]
    proj_weight: &[f32],   // [D, D]
    proj_bias: &[f32],     // [D]
    norm2_weight: &[f32],   // [D]
    norm2_bias: &[f32],     // [D]
    norm2_eps: f32,
    fc1_weight: &[f32],    // [D, ff_dim]
    fc1_bias: &[f32],      // [ff_dim]
    fc2_weight: &[f32],    // [ff_dim, D]
    fc2_bias: &[f32],      // [D]
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
    dim: usize,
    ff_dim: usize,
) {
    // ── Attention sub-layer ──────────────────────────────────────────────

    // 1. LayerNorm(x) → norm_out
    let norm_out = &mut scratch.norm_out[..seq_len * dim];
    layernorm(norm_out, &x[..seq_len * dim], norm1_weight, norm1_bias, norm1_eps, seq_len, dim);

    // 2. QKV = norm_out × qkv_weight   [S, D] × [D, 3D] → [S, 3D]
    //    This is the largest matmul per block. Let BLAS handle threading.
    let qkv = &mut scratch.qkv[..seq_len * 3 * dim];
    sgemm(qkv, norm_out, qkv_weight, seq_len, 3 * dim, dim, 1.0, 0.0);

    // 3. Split QKV → Q[H,S,dh], K[H,S,dh], V[H,S,dh] with bias + Q scaling
    let q = &mut scratch.q[..n_heads * seq_len * head_dim];
    let k = &mut scratch.k[..n_heads * seq_len * head_dim];
    let v = &mut scratch.v[..n_heads * seq_len * head_dim];
    let scale = (head_dim as f32).powf(-0.5);
    split_qkv_scaled(q, k, v, qkv, qkv_bias, n_heads, head_dim, scale, seq_len);

    // 4-6. Per-head attention: Q×K^T → softmax → attn×V  (parallel across heads)
    let attn = &mut scratch.attn[..n_heads * seq_len * seq_len];
    let attn_out = &mut scratch.attn_out[..n_heads * seq_len * head_dim];
    let ss = seq_len * seq_len;
    let sdh = seq_len * head_dim;

    // Parallel per-head: each head does QK^T + softmax + AV independently
    attn.par_chunks_mut(ss)
        .zip(attn_out.par_chunks_mut(sdh))
        .enumerate()
        .for_each(|(h, (a_h, o_h))| {
            let q_h = &q[h * sdh..(h + 1) * sdh];
            let k_h = &k[h * sdh..(h + 1) * sdh];
            let v_h = &v[h * sdh..(h + 1) * sdh];

            // Q[S,dh] × K[S,dh]^T = A[S,S]
            sgemm_at(a_h, q_h, k_h, seq_len, seq_len, head_dim, 1.0, 0.0);
            // Softmax per row
            softmax_inplace(a_h, seq_len, seq_len);
            // A[S,S] × V[S,dh] = O[S,dh]
            sgemm(o_h, a_h, v_h, seq_len, head_dim, seq_len, 1.0, 0.0);
        });

    // 7. Merge heads: [H, S, dh] → [S, D]
    let merged = &mut scratch.merged[..seq_len * dim];
    merge_heads(merged, attn_out, n_heads, head_dim, seq_len);

    // 8. Projection: proj_out = merged × proj_weight   [S, D] × [D, D] → [S, D]
    let proj_out = &mut scratch.proj_out[..seq_len * dim];
    sgemm(proj_out, merged, proj_weight, seq_len, dim, dim, 1.0, 0.0);

    // 9. Residual + bias + LayerNorm: residual = x + proj_out + proj_bias; norm2 = LN(residual)
    let residual = &mut scratch.residual[..seq_len * dim];
    let norm2_out = &mut scratch.norm_out[..seq_len * dim]; // reuse norm_out buffer
    bias_residual_add_layernorm(
        residual, norm2_out,
        &x[..seq_len * dim], proj_out, proj_bias,
        norm2_weight, norm2_bias, norm2_eps,
        seq_len, dim,
    );

    // ── FFN sub-layer ───────────────────────────────────────────────────

    // 10. FC1: hidden = norm2_out × fc1_weight   [S, D] × [D, ff] → [S, ff]
    let ffn_hidden = &mut scratch.ffn_hidden[..seq_len * ff_dim];
    sgemm(ffn_hidden, norm2_out, fc1_weight, seq_len, ff_dim, dim, 1.0, 0.0);

    // 11. Fused bias + GELU
    bias_gelu_inplace(ffn_hidden, fc1_bias, seq_len, ff_dim);

    // 12. FC2: ffn_out = hidden × fc2_weight   [S, ff] × [ff, D] → [S, D]
    let ffn_out = &mut scratch.ffn_out[..seq_len * dim];
    sgemm(ffn_out, ffn_hidden, fc2_weight, seq_len, dim, ff_dim, 1.0, 0.0);

    // 13. Residual: x = residual + ffn_out + fc2_bias
    bias_residual_add(
        &mut x[..seq_len * dim],
        residual, ffn_out, fc2_bias,
        seq_len, dim,
    );
}

// ─── Weight extraction helpers ───────────────────────────────────────────────

/// Pre-extracted weights for a single transformer block.
/// Stored as contiguous f32 slices for direct BLAS/fused-op use.
pub struct BlockWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub norm1_eps: f32,
    pub qkv_weight: Vec<f32>,   // [D, 3*D] transposed for row-major
    pub qkv_bias: Vec<f32>,
    pub proj_weight: Vec<f32>,  // [D, D] transposed for row-major
    pub proj_bias: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub norm2_eps: f32,
    pub fc1_weight: Vec<f32>,   // [D, ff_dim] transposed for row-major
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,   // [ff_dim, D] transposed for row-major
    pub fc2_bias: Vec<f32>,
}

/// Pre-extracted weights for the full model.
pub struct ModelWeightsRaw {
    pub blocks: Vec<BlockWeights>,
    pub final_norm_weight: Option<Vec<f32>>,
    pub final_norm_bias: Option<Vec<f32>>,
    pub final_norm_eps: f32,
    pub patch_weight: Vec<f32>,   // [patch_size, embed_dim] transposed
    pub patch_bias: Vec<f32>,
    pub cls_with_pe: Vec<f32>,    // [1, D]
    pub channel_embed_weight: Vec<f32>,  // [max_channels, D]
    pub temporal_pe: Vec<f32>,    // [max_len, D]
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ff_dim: usize,
    pub patch_size: usize,
    pub global_pool: bool,
}

/// Extract all model weights into contiguous f32 buffers.
///
/// Burn's Linear stores weights as [in, out] (transposed from PyTorch's [out, in]).
/// For row-major sgemm(A, W) where A=[S, in], W=[in, out], result=[S, out],
/// we need W in [in, out] layout — which is exactly how Burn stores it.
pub fn extract_model_weights<B: burn::prelude::Backend>(
    model: &crate::model::steegformer::STEEGFormerWithPE<B>,
) -> ModelWeightsRaw {
    let m = &model.model;
    let dim = m.embed_dim;
    let n_heads = m.blocks[0].attn.n_heads;
    let head_dim = m.blocks[0].attn.head_dim;

    // Determine ff_dim from fc1 weight shape
    let fc1_shape = m.blocks[0].mlp.fc1.weight.val().dims();
    let ff_dim = fc1_shape[1]; // Burn stores as [in, out], so out = ff_dim

    let blocks: Vec<BlockWeights> = m.blocks.iter().map(|blk| {
        BlockWeights {
            norm1_weight: tensor_to_vec(&blk.norm1.inner.gamma.val()),
            norm1_bias: tensor_to_vec(&blk.norm1.inner.beta.as_ref().unwrap().val()),
            norm1_eps: blk.norm1.eps as f32,
            qkv_weight: tensor_to_vec(&blk.attn.qkv.weight.val()),   // [in=D, out=3D]
            qkv_bias: tensor_to_vec(&blk.attn.qkv.bias.as_ref().unwrap().val()),
            proj_weight: tensor_to_vec(&blk.attn.proj.weight.val()),  // [in=D, out=D]
            proj_bias: tensor_to_vec(&blk.attn.proj.bias.as_ref().unwrap().val()),
            norm2_weight: tensor_to_vec(&blk.norm2.inner.gamma.val()),
            norm2_bias: tensor_to_vec(&blk.norm2.inner.beta.as_ref().unwrap().val()),
            norm2_eps: blk.norm2.eps as f32,
            fc1_weight: tensor_to_vec(&blk.mlp.fc1.weight.val()),     // [in=D, out=ff]
            fc1_bias: tensor_to_vec(&blk.mlp.fc1.bias.as_ref().unwrap().val()),
            fc2_weight: tensor_to_vec(&blk.mlp.fc2.weight.val()),     // [in=ff, out=D]
            fc2_bias: tensor_to_vec(&blk.mlp.fc2.bias.as_ref().unwrap().val()),
        }
    }).collect();

    let (final_norm_weight, final_norm_bias, final_norm_eps) = if let Some(ref norm) = m.norm {
        (
            Some(tensor_to_vec(&norm.inner.gamma.val())),
            Some(tensor_to_vec(&norm.inner.beta.as_ref().unwrap().val())),
            norm.eps as f32,
        )
    } else if let Some(ref norm) = m.fc_norm {
        (
            Some(tensor_to_vec(&norm.inner.gamma.val())),
            Some(tensor_to_vec(&norm.inner.beta.as_ref().unwrap().val())),
            norm.eps as f32,
        )
    } else {
        (None, None, 1e-6)
    };

    // Patch embed: Burn Linear weight is [in=patch_size, out=embed_dim]
    let patch_weight = tensor_to_vec(&m.patch_embed.proj.weight.val());
    let patch_bias = tensor_to_vec(&m.patch_embed.proj.bias.as_ref().unwrap().val());

    // CLS + PE
    let cls_with_pe = tensor_to_vec(&model.cls_with_pe.clone().reshape([dim]));

    // Channel embeddings
    let channel_embed_weight = tensor_to_vec(&m.channel_embed.embedding.weight.val());

    // Temporal PE
    let temporal_pe = tensor_to_vec(&model.temporal_pe.pe.clone());

    ModelWeightsRaw {
        blocks,
        final_norm_weight,
        final_norm_bias,
        final_norm_eps,
        patch_weight,
        patch_bias,
        cls_with_pe,
        channel_embed_weight,
        temporal_pe,
        dim,
        n_heads,
        head_dim,
        ff_dim,
        patch_size: m.patch_size,
        global_pool: m.global_pool,
    }
}

fn tensor_to_vec<B: burn::prelude::Backend, const D: usize>(
    t: &burn::prelude::Tensor<B, D>,
) -> Vec<f32> {
    t.to_data().to_vec::<f32>().expect("tensor to vec")
}

// ─── Full Forward Pass (fused CPU) ──────────────────────────────────────────

/// Run the complete ST-EEGFormer forward pass on raw f32 data.
///
/// signal: [C, T] row-major f32 (already normalized)
/// channel_indices: [C] i64 channel embedding indices
///
/// Returns: [D] embedding vector
#[cfg(feature = "blas-accelerate")]
pub fn forward_fused(
    signal: &[f32],           // [C, T]
    channel_indices: &[i64],  // [C]
    weights: &ModelWeightsRaw,
    scratch: &mut ScratchBuffers,
) -> Vec<f32> {
    let c = channel_indices.len();
    let t = signal.len() / c;
    let d = weights.dim;
    let ps = weights.patch_size;
    let num_patches = t / ps;
    let seq_total = num_patches * c;
    let seq_with_cls = seq_total + 1;

    // Resize scratch for this sequence length
    scratch.resize(seq_with_cls, d, weights.n_heads, weights.ff_dim);

    // ── 1. Patch embedding ──────────────────────────────────────────────
    // signal [C, T] → patches [num_patches, C, ps] → embedded [num_patches, C, D]
    // Then add temporal PE + channel PE, flatten to [seq_total, D]

    // Allocate token buffer: [seq_with_cls, D]
    let mut tokens = vec![0.0f32; seq_with_cls * d];

    // First token = CLS (pre-computed with PE)
    tokens[..d].copy_from_slice(&weights.cls_with_pe);

    // Patch embed + positional encoding for remaining tokens
    // Token order: for each time patch t, for each channel c: token[t*C + c]
    for tp in 0..num_patches {
        for ch in 0..c {
            let token_idx = 1 + tp * c + ch; // +1 for CLS
            let token_out = &mut tokens[token_idx * d..(token_idx + 1) * d];

            // Extract patch: signal[ch, tp*ps .. (tp+1)*ps]
            let patch_start = ch * t + tp * ps;
            let patch = &signal[patch_start..patch_start + ps];

            // Linear projection: patch[ps] × weight[ps, D] + bias[D]
            // Using direct computation (patch_size=16 is tiny, BLAS overhead not worth it)
            for j in 0..d {
                let mut sum = weights.patch_bias[j];
                for i in 0..ps {
                    sum += patch[i] * weights.patch_weight[i * d + j];
                }
                token_out[j] = sum;
            }

            // Add temporal PE (index = tp, offset by 0 since CLS PE is pre-added)
            let pe_off = tp * d;
            for j in 0..d {
                token_out[j] += weights.temporal_pe[pe_off + j];
            }

            // Add channel embedding
            let ch_idx = channel_indices[ch] as usize;
            let ch_off = ch_idx * d;
            for j in 0..d {
                token_out[j] += weights.channel_embed_weight[ch_off + j];
            }
        }
    }

    // ── 2. Transformer blocks ───────────────────────────────────────────
    for blk in &weights.blocks {
        transformer_block_fused(
            &mut tokens,
            scratch,
            &blk.norm1_weight, &blk.norm1_bias, blk.norm1_eps,
            &blk.qkv_weight, &blk.qkv_bias,
            &blk.proj_weight, &blk.proj_bias,
            &blk.norm2_weight, &blk.norm2_bias, blk.norm2_eps,
            &blk.fc1_weight, &blk.fc1_bias,
            &blk.fc2_weight, &blk.fc2_bias,
            weights.n_heads, weights.head_dim,
            seq_with_cls, d, weights.ff_dim,
        );
    }

    // ── 3. Output ───────────────────────────────────────────────────────
    let mut output = vec![0.0f32; d];

    if weights.global_pool {
        // Global average pool (skip CLS token)
        let inv_n = 1.0 / seq_total as f32;
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 1..seq_with_cls {
                sum += tokens[i * d + j];
            }
            output[j] = sum * inv_n;
        }
        // Apply fc_norm if present
        if let (Some(ref w), Some(ref b)) = (&weights.final_norm_weight, &weights.final_norm_bias) {
            let mut normed = vec![0.0f32; d];
            layernorm(&mut normed, &output, w, b, weights.final_norm_eps, 1, d);
            output = normed;
        }
    } else {
        // CLS token output
        let cls = &tokens[..d];
        // Apply final norm
        if let (Some(ref w), Some(ref b)) = (&weights.final_norm_weight, &weights.final_norm_bias) {
            // Need to norm the full sequence first, then extract CLS
            let mut normed_all = vec![0.0f32; seq_with_cls * d];
            layernorm(&mut normed_all, &tokens, w, b, weights.final_norm_eps, seq_with_cls, d);
            output.copy_from_slice(&normed_all[..d]);
        } else {
            output.copy_from_slice(cls);
        }
    }

    output
}

// ─── Channel-wise z-score normalization ──────────────────────────────────────

/// Normalize signal [C, T] channel-wise: (x - mean) / std
pub fn channel_normalize(signal: &mut [f32], n_channels: usize, n_samples: usize) {
    let inv_n = 1.0 / n_samples as f32;
    for ch in 0..n_channels {
        let off = ch * n_samples;
        let row = &mut signal[off..off + n_samples];

        let (mean, var) = mean_var_f32(row, inv_n);
        let inv_std = 1.0 / (var + 1e-8).sqrt();

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                let mean_v = vdupq_n_f32(mean);
                let inv_std_v = vdupq_n_f32(inv_std);
                let chunks = n_samples / 4;
                let remainder = n_samples % 4;
                for i in 0..chunks {
                    let j = i * 4;
                    let v = vld1q_f32(row.as_ptr().add(j));
                    let centered = vsubq_f32(v, mean_v);
                    let normed = vmulq_f32(centered, inv_std_v);
                    vst1q_f32(row.as_mut_ptr().add(j), normed);
                }
                for i in 0..remainder {
                    let j = chunks * 4 + i;
                    row[j] = (row[j] - mean) * inv_std;
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for v in row.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }
    }
}
