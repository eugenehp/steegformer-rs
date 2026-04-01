/// Fused CubeCL GPU kernels for the ST-EEGFormer encoder.
///
/// **LayerNorm** — single-pass mean+variance+normalize+scale+bias using `plane_sum`.
/// **Add+LayerNorm** — fused residual add + layernorm in one kernel (2 outputs).
/// **Softmax** — general kernel supporting any last-dimension size.
/// **GELU** — fused element-wise.
/// **Bias+GELU** — fused bias addition + GELU activation.
/// **Split QKV Scaled** — splits [B,S,3D] → 3× [B,H,S,dh] with Q scaling.
/// **Merge Heads** — [B,H,S,dh] → [B,S,D] (transpose + flatten in one pass).
/// **Flash Attention** — fused Q×K^T → softmax → attn×V (avoids S×S materialization).

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl::server::Handle;
use burn::backend::wgpu::CubeTensor;
use burn_backend::{DType, Shape};
use burn_cubecl::kernel::into_contiguous;

// ── tensor helpers ────────────────────────────────────────────────────────

fn elem_size(dtype: DType) -> usize {
    match dtype {
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F32 | DType::Flex32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
        DType::I8 | DType::U8 | DType::Bool => 1,
        DType::QFloat(_) => 4,
    }
}

fn contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut s = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

fn empty_cube(src: &CubeTensor<WgpuRuntime>, shape: Shape) -> CubeTensor<WgpuRuntime> {
    let n_bytes = shape.dims.iter().product::<usize>() * elem_size(src.dtype);
    let handle: Handle = src.client.empty(n_bytes);
    CubeTensor {
        client: src.client.clone(),
        device: src.device.clone(),
        handle,
        strides: contiguous_strides(&shape.dims),
        shape,
        dtype: src.dtype,
        qparams: None,
    }
}

fn empty_cube_shape(src: &CubeTensor<WgpuRuntime>, dims: Vec<usize>) -> CubeTensor<WgpuRuntime> {
    empty_cube(src, Shape::from(dims))
}

// ══════════════════════════════════════════════════════════════════════════
// 1) LayerNorm — single-pass fused kernel
// ══════════════════════════════════════════════════════════════════════════

#[cube(launch)]
fn layernorm_kernel<F: Float>(
    x: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    d: u32,
    eps: f32,
) {
    let row = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let n_per_lane = d / PLANE_DIM;
    let base = row * d + lane;

    let mut sum_val = F::new(0.0);
    let mut sum_sq = F::new(0.0);
    for i in 0u32..n_per_lane {
        let v = x[(base + i * PLANE_DIM) as usize];
        sum_val += v;
        sum_sq += v * v;
    }

    let total_sum = plane_sum(sum_val);
    let total_sq = plane_sum(sum_sq);

    let d_f = F::cast_from(d);
    let mean = total_sum / d_f;
    let variance = total_sq / d_f - mean * mean;
    let inv_std = F::powf(variance + F::cast_from(eps), F::new(-0.5));

    for i in 0u32..n_per_lane {
        let idx = (base + i * PLANE_DIM) as usize;
        let w_idx = (lane + i * PLANE_DIM) as usize;
        let normalized = (x[idx] - mean) * inv_std;
        out[idx] = normalized * weight[w_idx] + bias[w_idx];
    }
}

pub fn launch_layernorm(
    x: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    eps: f32,
) -> CubeTensor<WgpuRuntime> {
    let x = into_contiguous(x);
    let weight = into_contiguous(weight);
    let bias = into_contiguous(bias);
    let out = empty_cube(&x, x.shape.clone());

    let d = *x.shape.dims.last().unwrap() as u32;
    let rows = x.shape.dims[..x.shape.num_dims() - 1]
        .iter()
        .product::<usize>() as u32;

    let plane = 32u32;
    let cube_dim = CubeDim { x: plane, y: 1, z: 1 };
    let cube_count = CubeCount::Static(rows, 1, 1);

    match x.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            layernorm_kernel::launch::<f32, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&weight.handle, &weight.strides, &weight.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(eps),
            ).expect("layernorm f32 launch");
        },
        DType::F16 => unsafe {
            layernorm_kernel::launch::<half::f16, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&weight.handle, &weight.strides, &weight.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(eps),
            ).expect("layernorm f16 launch");
        },
        dt => panic!("layernorm: unsupported dtype {dt:?}"),
    }
    out
}

// ══════════════════════════════════════════════════════════════════════════
// 2) Add + LayerNorm — fused residual addition + layernorm
//    Outputs BOTH the sum (for next residual) and the normalized result.
// ══════════════════════════════════════════════════════════════════════════

#[cube(launch)]
fn add_layernorm_kernel<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    sum_out: &mut Tensor<F>,
    norm_out: &mut Tensor<F>,
    d: u32,
    eps: f32,
) {
    let row = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let n_per_lane = d / PLANE_DIM;
    let base = row * d + lane;

    // Pass 1: compute a+b, accumulate stats
    let mut sum_val = F::new(0.0);
    let mut sum_sq = F::new(0.0);
    for i in 0u32..n_per_lane {
        let idx = (base + i * PLANE_DIM) as usize;
        let v = a[idx] + b[idx];
        sum_out[idx] = v;
        sum_val += v;
        sum_sq += v * v;
    }

    let total_sum = plane_sum(sum_val);
    let total_sq = plane_sum(sum_sq);

    let d_f = F::cast_from(d);
    let mean = total_sum / d_f;
    let variance = total_sq / d_f - mean * mean;
    let inv_std = F::powf(variance + F::cast_from(eps), F::new(-0.5));

    // Pass 2: normalize + scale + bias (reads from sum_out)
    for i in 0u32..n_per_lane {
        let idx = (base + i * PLANE_DIM) as usize;
        let w_idx = (lane + i * PLANE_DIM) as usize;
        let normalized = (sum_out[idx] - mean) * inv_std;
        norm_out[idx] = normalized * weight[w_idx] + bias[w_idx];
    }
}

/// Launch fused add+layernorm: returns (a+b, layernorm(a+b)).
pub fn launch_add_layernorm(
    a: CubeTensor<WgpuRuntime>,
    b: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    eps: f32,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    let a = into_contiguous(a);
    let b = into_contiguous(b);
    let weight = into_contiguous(weight);
    let bias = into_contiguous(bias);
    let sum_out = empty_cube(&a, a.shape.clone());
    let norm_out = empty_cube(&a, a.shape.clone());

    let d = *a.shape.dims.last().unwrap() as u32;
    let rows = a.shape.dims[..a.shape.num_dims() - 1]
        .iter()
        .product::<usize>() as u32;

    let plane = 32u32;
    let cube_dim = CubeDim { x: plane, y: 1, z: 1 };
    let cube_count = CubeCount::Static(rows, 1, 1);

    match a.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            add_layernorm_kernel::launch::<f32, WgpuRuntime>(
                &a.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&a.handle, &a.strides, &a.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&b.handle, &b.strides, &b.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&weight.handle, &weight.strides, &weight.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&sum_out.handle, &sum_out.strides, &sum_out.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&norm_out.handle, &norm_out.strides, &norm_out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(eps),
            ).expect("add_layernorm f32 launch");
        },
        DType::F16 => unsafe {
            add_layernorm_kernel::launch::<half::f16, WgpuRuntime>(
                &a.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&a.handle, &a.strides, &a.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&b.handle, &b.strides, &b.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&weight.handle, &weight.strides, &weight.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&sum_out.handle, &sum_out.strides, &sum_out.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&norm_out.handle, &norm_out.strides, &norm_out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(eps),
            ).expect("add_layernorm f16 launch");
        },
        dt => panic!("add_layernorm: unsupported dtype {dt:?}"),
    }
    (sum_out, norm_out)
}

// ══════════════════════════════════════════════════════════════════════════
// 3) Softmax — general kernel, supports any last-dimension size
// ══════════════════════════════════════════════════════════════════════════
//
// Each workgroup handles one row. PLANE_DIM=32 lanes cooperate.
// For non-aligned d: lanes with col >= d contribute 0 to sums.
// Max is found by having each lane read ALL elements (data is in L1/L2).

#[cube(launch)]
fn softmax_kernel<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<F>,
    d: u32,
) {
    let row = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let base = row * d;
    let n_per_lane = (d + PLANE_DIM - 1) / PLANE_DIM;

    // Pass 1: find row max — each lane reads its strided slice, then plane_max.
    // This eliminates the 32× redundant reads from the naive all-lanes approach.
    // For d=1057, S=8456 rows: saves ~275M redundant memory reads (~1.1GB bandwidth).
    let mut local_max = F::new(-65504.0);
    for i in 0u32..n_per_lane {
        let col = lane + i * PLANE_DIM;
        if col < d {
            local_max = F::max(local_max, x[(base + col) as usize]);
        }
    }
    let row_max = plane_max(local_max);

    // Pass 2: exp(x - max) and local sum (each lane handles its strided elements)
    let mut local_sum = F::new(0.0);
    for i in 0u32..n_per_lane {
        let col = lane + i * PLANE_DIM;
        if col < d {
            let idx = (base + col) as usize;
            let e = F::exp(x[idx] - row_max);
            out[idx] = e;
            local_sum += e;
        }
    }
    let total_sum = plane_sum(local_sum);

    // Pass 3: normalize
    let inv_sum = F::new(1.0) / total_sum;
    for i in 0u32..n_per_lane {
        let col = lane + i * PLANE_DIM;
        if col < d {
            let idx = (base + col) as usize;
            out[idx] = out[idx] * inv_sum;
        }
    }
}

pub fn launch_softmax(
    x: CubeTensor<WgpuRuntime>,
    _dim: usize,
) -> CubeTensor<WgpuRuntime> {
    let x = into_contiguous(x);
    let out = empty_cube(&x, x.shape.clone());

    let d = *x.shape.dims.last().unwrap() as u32;
    let rows = x.shape.dims[..x.shape.num_dims() - 1]
        .iter()
        .product::<usize>() as u32;

    let plane = 32u32;
    let cube_dim = CubeDim { x: plane, y: 1, z: 1 };
    let cube_count = CubeCount::Static(rows, 1, 1);

    match x.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            softmax_kernel::launch::<f32, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
            ).expect("softmax f32 launch");
        },
        DType::F16 => unsafe {
            softmax_kernel::launch::<half::f16, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
            ).expect("softmax f16 launch");
        },
        dt => panic!("softmax: unsupported dtype {dt:?}"),
    }
    out
}

// ══════════════════════════════════════════════════════════════════════════
// 4) GELU — fused element-wise
// ══════════════════════════════════════════════════════════════════════════

/// Fast GELU: x * sigmoid(1.702 * x), 4 elements per thread.
#[cube(launch)]
fn gelu_kernel<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<F>,
    total: usize,
) {
    let base = ABSOLUTE_POS * 4;
    let coeff = F::new(1.702);
    if base < total {
        let v = x[base];
        out[base] = v / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v));
    }
    if base + 1 < total {
        let v = x[base + 1];
        out[base + 1] = v / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v));
    }
    if base + 2 < total {
        let v = x[base + 2];
        out[base + 2] = v / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v));
    }
    if base + 3 < total {
        let v = x[base + 3];
        out[base + 3] = v / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v));
    }
}

pub fn launch_gelu(
    x: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let x = into_contiguous(x);
    let out = empty_cube(&x, x.shape.clone());

    let total: usize = x.shape.dims.iter().product();
    let cube_dim_x: u32 = 256;
    let threads_needed = ((total as u32) + 3) / 4;
    let cube_count_x = (threads_needed + cube_dim_x - 1) / cube_dim_x;
    let cube_dim = CubeDim { x: cube_dim_x, y: 1, z: 1 };
    let cube_count = CubeCount::Static(cube_count_x, 1, 1);

    match x.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            gelu_kernel::launch::<f32, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(total),
            ).expect("gelu f32 launch");
        },
        DType::F16 => unsafe {
            gelu_kernel::launch::<half::f16, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(total),
            ).expect("gelu f16 launch");
        },
        dt => panic!("gelu: unsupported dtype {dt:?}"),
    }
    out
}

// ══════════════════════════════════════════════════════════════════════════
// 5) Bias + GELU — fused bias addition + GELU activation
//    Input x: [B, S, hidden_dim] (contiguous, from matmul — no bias yet)
//    bias: [hidden_dim]
//    out = GELU(x + bias)
// ══════════════════════════════════════════════════════════════════════════

/// Fused bias + GELU with 4-element manual unrolling.
/// Uses fast GELU: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
/// Manual unrolling is faster than 1-per-thread because CubeCL's
/// generated MSL uses strided buffer access patterns that prevent
/// Metal's auto-vectorizer from coalescing into float4 loads.
#[cube(launch)]
fn bias_gelu_kernel<F: Float>(
    x: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    total: usize,
    #[comptime] bias_dim: u32,
) {
    let base = ABSOLUTE_POS * 4;
    let coeff = F::new(1.702);
    // comptime bias_dim: compiler sees constant, converts % to & for power-of-2
    let mask = comptime![bias_dim - 1];
    if base < total {
        let v0 = x[base] + bias[((base as u32) & mask) as usize];
        out[base] = v0 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v0));
    }
    if base + 1 < total {
        let v1 = x[base + 1] + bias[(((base + 1) as u32) & mask) as usize];
        out[base + 1] = v1 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v1));
    }
    if base + 2 < total {
        let v2 = x[base + 2] + bias[(((base + 2) as u32) & mask) as usize];
        out[base + 2] = v2 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v2));
    }
    if base + 3 < total {
        let v3 = x[base + 3] + bias[(((base + 3) as u32) & mask) as usize];
        out[base + 3] = v3 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v3));
    }
}

pub fn launch_bias_gelu(
    x: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let x = into_contiguous(x);
    let bias = into_contiguous(bias);
    let out = empty_cube(&x, x.shape.clone());

    let bias_dim = *bias.shape.dims.last().unwrap() as u32;
    let total: usize = x.shape.dims.iter().product();
    let cube_dim_x: u32 = 256;
    let threads_needed = ((total as u32) + 3) / 4;
    let cube_count_x = (threads_needed + cube_dim_x - 1) / cube_dim_x;
    let cube_dim = CubeDim { x: cube_dim_x, y: 1, z: 1 };
    let cube_count = CubeCount::Static(cube_count_x, 1, 1);

    match x.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            bias_gelu_kernel::launch::<f32, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(total),
                bias_dim,  // #[comptime] — enables % → & for power-of-2
            ).expect("bias_gelu f32 launch");
        },
        DType::F16 => unsafe {
            bias_gelu_kernel::launch::<half::f16, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(total),
                bias_dim,  // #[comptime]
            ).expect("bias_gelu f16 launch");
        },
        dt => panic!("bias_gelu: unsupported dtype {dt:?}"),
    }
    out
}

// ══════════════════════════════════════════════════════════════════════════
// 6) Split QKV Scaled — split [B, S, 3D] into Q, K, V as [B, H, S, dh]
//    with Q pre-multiplied by attention scale. One kernel replaces:
//    - 3× narrow + reshape (each triggers copy dispatch on non-contiguous data)
//    - 3× implicit into_contiguous in matmul for swap_dims'd inputs
//    - 1× mul_scalar for Q scaling
// ══════════════════════════════════════════════════════════════════════════

#[cube(launch)]
fn split_qkv_bias_scaled_kernel<F: Float>(
    qkv: &Tensor<F>,
    bias: &Tensor<F>,
    q_out: &mut Tensor<F>,
    k_out: &mut Tensor<F>,
    v_out: &mut Tensor<F>,
    scale: f32,
    s_size: u32,
    h_size: u32,
    dh_size: u32,
    total: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total {
        terminate!();
    }

    let pos_u = pos as u32;
    let dim = h_size * dh_size;
    let three_dim = dim * 3u32;

    // Decompose linear position → (b, s, h, j)
    let b = pos_u / (s_size * dim);
    let rem = pos_u % (s_size * dim);
    let s = rem / dim;
    let d_idx = rem % dim;
    let h = d_idx / dh_size;
    let j = d_idx % dh_size;

    // Read from QKV [B, S, 3D] (contiguous, no bias yet) + add bias [3D]
    let qkv_base = (b * s_size * three_dim + s * three_dim + d_idx) as usize;
    let q_val = qkv[qkv_base] + bias[d_idx as usize];
    let k_val = qkv[qkv_base + dim as usize] + bias[(dim + d_idx) as usize];
    let v_val = qkv[qkv_base + 2 * dim as usize] + bias[(2 * dim + d_idx) as usize];

    // Q, K, V all write to [B, H, S, dh] (contiguous)
    // K is NOT pre-transposed — cubecl matmul handles transpose via
    // swap_dims which preserves contiguous layout info for WMMA.
    // Pre-transposing K caused 'HighlyPermuted' layout → blocked WMMA → 4× slower.
    let out_idx = (b * h_size * s_size * dh_size + h * s_size * dh_size + s * dh_size + j) as usize;

    q_out[out_idx] = q_val * F::cast_from(scale);
    k_out[out_idx] = k_val;
    v_out[out_idx] = v_val;
}

/// Split QKV tensor and produce contiguous Q, K, V with Q pre-scaled.
/// Input: qkv [B, S, 3*D] (contiguous, without bias), bias [3*D]
/// Output: q, k, v each [B, H, S, dh] (contiguous)
/// Fuses: bias add + split + Q scaling in one dispatch.
pub fn launch_split_qkv_scaled(
    qkv: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    let qkv = into_contiguous(qkv);
    let bias = into_contiguous(bias);
    let dims = &qkv.shape.dims;
    let b = dims[0];
    let s = dims[1];

    let out_shape = Shape::from(vec![b, n_heads, s, head_dim]);
    let q_out = empty_cube(&qkv, out_shape.clone());
    let k_out = empty_cube(&qkv, out_shape.clone());  // K in standard layout [B,H,S,dh]
    let v_out = empty_cube(&qkv, out_shape);

    let total = b * s * n_heads * head_dim;
    let cube_dim_x: u32 = 256;
    let cube_count_x = ((total as u32) + cube_dim_x - 1) / cube_dim_x;
    let cube_dim = CubeDim { x: cube_dim_x, y: 1, z: 1 };
    let cube_count = CubeCount::Static(cube_count_x, 1, 1);

    match qkv.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            split_qkv_bias_scaled_kernel::launch::<f32, WgpuRuntime>(
                &qkv.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&qkv.handle, &qkv.strides, &qkv.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&q_out.handle, &q_out.strides, &q_out.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&k_out.handle, &k_out.strides, &k_out.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&v_out.handle, &v_out.strides, &v_out.shape.dims, 1),
                ScalarArg::new(scale),
                ScalarArg::new(s as u32),
                ScalarArg::new(n_heads as u32),
                ScalarArg::new(head_dim as u32),
                ScalarArg::new(total),
            ).expect("split_qkv_scaled f32 launch");
        },
        DType::F16 => unsafe {
            split_qkv_bias_scaled_kernel::launch::<half::f16, WgpuRuntime>(
                &qkv.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&qkv.handle, &qkv.strides, &qkv.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&q_out.handle, &q_out.strides, &q_out.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&k_out.handle, &k_out.strides, &k_out.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&v_out.handle, &v_out.strides, &v_out.shape.dims, 1),
                ScalarArg::new(scale),
                ScalarArg::new(s as u32),
                ScalarArg::new(n_heads as u32),
                ScalarArg::new(head_dim as u32),
                ScalarArg::new(total),
            ).expect("split_qkv_scaled f16 launch");
        },
        dt => panic!("split_qkv_scaled: unsupported dtype {dt:?}"),
    }
    (q_out, k_out, v_out)
}

// ══════════════════════════════════════════════════════════════════════════
// 7) Merge Heads — [B, H, S, dh] → [B, S, D]
//    Replaces swap_dims(1,2) + flatten(2,3) which triggers a copy dispatch.
// ══════════════════════════════════════════════════════════════════════════

#[cube(launch)]
fn merge_heads_kernel<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<F>,
    s_size: u32,
    h_size: u32,
    dh_size: u32,
    total: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total {
        terminate!();
    }

    let pos_u = pos as u32;
    let dim = h_size * dh_size;

    // Output layout: [B, S, D] where D = H*dh
    let b = pos_u / (s_size * dim);
    let rem = pos_u % (s_size * dim);
    let s = rem / dim;
    let d_idx = rem % dim;
    let h = d_idx / dh_size;
    let j = d_idx % dh_size;

    // Input layout: [B, H, S, dh]
    let in_idx = (b * h_size * s_size * dh_size + h * s_size * dh_size + s * dh_size + j) as usize;

    out[pos] = x[in_idx];
}

/// Merge attention heads: [B, H, S, dh] → [B, S, D] (contiguous).
pub fn launch_merge_heads(
    x: CubeTensor<WgpuRuntime>,
    n_heads: usize,
    head_dim: usize,
) -> CubeTensor<WgpuRuntime> {
    let x = into_contiguous(x);
    let dims = &x.shape.dims;
    let b = dims[0];
    let s = dims[2]; // [B, H, S, dh]
    let d = n_heads * head_dim;

    let out = empty_cube_shape(&x, vec![b, s, d]);

    let total = b * s * d;
    let cube_dim_x: u32 = 256;
    let cube_count_x = ((total as u32) + cube_dim_x - 1) / cube_dim_x;
    let cube_dim = CubeDim { x: cube_dim_x, y: 1, z: 1 };
    let cube_count = CubeCount::Static(cube_count_x, 1, 1);

    match x.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            merge_heads_kernel::launch::<f32, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(s as u32),
                ScalarArg::new(n_heads as u32),
                ScalarArg::new(head_dim as u32),
                ScalarArg::new(total),
            ).expect("merge_heads f32 launch");
        },
        DType::F16 => unsafe {
            merge_heads_kernel::launch::<half::f16, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&x.handle, &x.strides, &x.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(s as u32),
                ScalarArg::new(n_heads as u32),
                ScalarArg::new(head_dim as u32),
                ScalarArg::new(total),
            ).expect("merge_heads f16 launch");
        },
        dt => panic!("merge_heads: unsupported dtype {dt:?}"),
    }
    out
}

// ══════════════════════════════════════════════════════════════════════════
// 8) Fused Bias + Add — adds bias to matmul output + residual add
//    out = residual + (matmul_out + bias)
//    Replaces separate bias add + residual add = 2 dispatches → 1
// ══════════════════════════════════════════════════════════════════════════

/// Fused bias + residual add, 4 elements per thread.
/// #[comptime] bias_dim enables modulo → bitwise AND optimization.
#[cube(launch)]
fn bias_residual_add_kernel<F: Float>(
    residual: &Tensor<F>,
    matmul_out: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    total: usize,
    #[comptime] bias_dim: u32,
) {
    let base = ABSOLUTE_POS * 4;
    let mask = comptime![bias_dim - 1];
    if base < total {
        out[base] = residual[base] + matmul_out[base] + bias[((base as u32) & mask) as usize];
    }
    if base + 1 < total {
        out[base + 1] = residual[base + 1] + matmul_out[base + 1] + bias[(((base + 1) as u32) & mask) as usize];
    }
    if base + 2 < total {
        out[base + 2] = residual[base + 2] + matmul_out[base + 2] + bias[(((base + 2) as u32) & mask) as usize];
    }
    if base + 3 < total {
        out[base + 3] = residual[base + 3] + matmul_out[base + 3] + bias[(((base + 3) as u32) & mask) as usize];
    }
}

/// Fused bias + residual add: out = residual + matmul_out + bias
pub fn launch_bias_residual_add(
    residual: CubeTensor<WgpuRuntime>,
    matmul_out: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let residual = into_contiguous(residual);
    let matmul_out = into_contiguous(matmul_out);
    let bias = into_contiguous(bias);
    let out = empty_cube(&residual, residual.shape.clone());

    let bias_dim = *bias.shape.dims.last().unwrap() as u32;
    let total: usize = residual.shape.dims.iter().product();
    let cube_dim_x: u32 = 256;
    let threads_needed = ((total as u32) + 3) / 4;
    let cube_count_x = (threads_needed + cube_dim_x - 1) / cube_dim_x;
    let cube_dim = CubeDim { x: cube_dim_x, y: 1, z: 1 };
    let cube_count = CubeCount::Static(cube_count_x, 1, 1);

    match residual.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            bias_residual_add_kernel::launch::<f32, WgpuRuntime>(
                &residual.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&residual.handle, &residual.strides, &residual.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&matmul_out.handle, &matmul_out.strides, &matmul_out.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(total),
                bias_dim,  // #[comptime]
            ).expect("bias_residual_add f32 launch");
        },
        DType::F16 => unsafe {
            bias_residual_add_kernel::launch::<half::f16, WgpuRuntime>(
                &residual.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&residual.handle, &residual.strides, &residual.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&matmul_out.handle, &matmul_out.strides, &matmul_out.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(total),
                bias_dim,  // #[comptime]
            ).expect("bias_residual_add f16 launch");
        },
        dt => panic!("bias_residual_add: unsupported dtype {dt:?}"),
    }
    out
}


// ══════════════════════════════════════════════════════════════════════════
// 9) Bias + Residual Add + LayerNorm — fused in one kernel
//    sum = residual + matmul_out + bias[col]
//    norm = layernorm(sum) * ln_weight + ln_bias
//    Returns (sum, norm) — saves 1 dispatch vs separate bias_residual_add + layernorm
// ══════════════════════════════════════════════════════════════════════════

#[cube(launch)]
fn bias_residual_add_layernorm_kernel<F: Float>(
    residual: &Tensor<F>,
    matmul_out: &Tensor<F>,
    bias: &Tensor<F>,
    ln_weight: &Tensor<F>,
    ln_bias: &Tensor<F>,
    sum_out: &mut Tensor<F>,
    norm_out: &mut Tensor<F>,
    d: u32,
    eps: f32,
) {
    let row = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let n_per_lane = d / PLANE_DIM;
    let base = row * d + lane;

    // Pass 1: compute residual + matmul_out + bias, accumulate stats
    let mut sum_val = F::new(0.0);
    let mut sum_sq = F::new(0.0);
    for i in 0u32..n_per_lane {
        let idx = (base + i * PLANE_DIM) as usize;
        let b_idx = (lane + i * PLANE_DIM) as usize;
        let v = residual[idx] + matmul_out[idx] + bias[b_idx];
        sum_out[idx] = v;
        sum_val += v;
        sum_sq += v * v;
    }

    let total_sum = plane_sum(sum_val);
    let total_sq = plane_sum(sum_sq);

    let d_f = F::cast_from(d);
    let mean = total_sum / d_f;
    let variance = total_sq / d_f - mean * mean;
    let inv_std = F::powf(variance + F::cast_from(eps), F::new(-0.5));

    // Pass 2: normalize + scale + ln_bias
    for i in 0u32..n_per_lane {
        let idx = (base + i * PLANE_DIM) as usize;
        let w_idx = (lane + i * PLANE_DIM) as usize;
        let normalized = (sum_out[idx] - mean) * inv_std;
        norm_out[idx] = normalized * ln_weight[w_idx] + ln_bias[w_idx];
    }
}

/// Fused bias + residual add + layernorm: returns (residual + matmul_out + bias, layernorm(sum)).
pub fn launch_bias_residual_add_layernorm(
    residual: CubeTensor<WgpuRuntime>,
    matmul_out: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    ln_weight: CubeTensor<WgpuRuntime>,
    ln_bias: CubeTensor<WgpuRuntime>,
    eps: f32,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    let residual = into_contiguous(residual);
    let matmul_out = into_contiguous(matmul_out);
    let bias = into_contiguous(bias);
    let ln_weight = into_contiguous(ln_weight);
    let ln_bias = into_contiguous(ln_bias);
    let sum_out = empty_cube(&residual, residual.shape.clone());
    let norm_out = empty_cube(&residual, residual.shape.clone());

    let d = *residual.shape.dims.last().unwrap() as u32;
    let rows = residual.shape.dims[..residual.shape.num_dims() - 1]
        .iter()
        .product::<usize>() as u32;

    let plane = 32u32;
    let cube_dim = CubeDim { x: plane, y: 1, z: 1 };
    let cube_count = CubeCount::Static(rows, 1, 1);

    match residual.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            bias_residual_add_layernorm_kernel::launch::<f32, WgpuRuntime>(
                &residual.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&residual.handle, &residual.strides, &residual.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&matmul_out.handle, &matmul_out.strides, &matmul_out.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&ln_weight.handle, &ln_weight.strides, &ln_weight.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&ln_bias.handle, &ln_bias.strides, &ln_bias.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&sum_out.handle, &sum_out.strides, &sum_out.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&norm_out.handle, &norm_out.strides, &norm_out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(eps),
            ).expect("bias_residual_add_layernorm f32 launch");
        },
        DType::F16 => unsafe {
            bias_residual_add_layernorm_kernel::launch::<half::f16, WgpuRuntime>(
                &residual.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&residual.handle, &residual.strides, &residual.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&matmul_out.handle, &matmul_out.strides, &matmul_out.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&bias.handle, &bias.strides, &bias.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&ln_weight.handle, &ln_weight.strides, &ln_weight.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&ln_bias.handle, &ln_bias.strides, &ln_bias.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&sum_out.handle, &sum_out.strides, &sum_out.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&norm_out.handle, &norm_out.strides, &norm_out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(eps),
            ).expect("bias_residual_add_layernorm f16 launch");
        },
        dt => panic!("bias_residual_add_layernorm: unsupported dtype {dt:?}"),
    }
    (sum_out, norm_out)
}

// ══════════════════════════════════════════════════════════════════════════
// 10) Fused Patch Embed + Positional Encoding + CLS Prepend
//     Combines: patch_embed(Linear) + channel_embed(Embedding) +
//     temporal_PE + CLS token into a single kernel.
//     Replaces ~8 dispatches with 1.
// ══════════════════════════════════════════════════════════════════════════

#[cube(launch)]
fn fused_embed_kernel<F: Float>(
    eeg: &Tensor<F>,           // [B, C, T] — raw EEG signal
    patch_w: &Tensor<F>,       // [patch_size, embed_dim]
    patch_b: &Tensor<F>,       // [embed_dim]
    ch_emb_w: &Tensor<F>,      // [max_channels, embed_dim]
    ch_idx: &Tensor<u32>,      // [B, C] — channel indices
    temp_pe: &Tensor<F>,       // [max_len, embed_dim]
    cls_with_pe: &Tensor<F>,   // [embed_dim]
    out: &mut Tensor<F>,       // [B, 1 + num_patches*C, embed_dim]
    n_channels: u32,
    n_samples: u32,
    patch_size: u32,
    embed_dim: u32,
    num_patches: u32,
    seq_with_cls: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos >= seq_with_cls * embed_dim {
        terminate!();
    }

    // Decompose pos → (token_idx, dim_j) within batch 0
    let token_idx = pos / embed_dim;
    let j = pos % embed_dim;

    if token_idx == 0u32 {
        // CLS token — just copy pre-computed CLS+PE
        out[pos as usize] = cls_with_pe[j as usize];
    } else {
        // Decompose token_idx-1 → (time_patch, channel)
        let tok = token_idx - 1u32;
        let tp = tok / n_channels;   // time patch index
        let ch = tok % n_channels;   // channel index

        // Patch embedding: linear projection of patch → embed_dim
        let mut val = patch_b[j as usize];
        let eeg_base = ch * n_samples + tp * patch_size;
        for i in 0u32..patch_size {
            val += eeg[(eeg_base + i) as usize] * patch_w[(i * embed_dim + j) as usize];
        }

        // Add temporal PE (index = tp)
        val += temp_pe[(tp * embed_dim + j) as usize];

        // Add channel embedding
        let ch_id = ch_idx[ch as usize];
        val += ch_emb_w[(ch_id * embed_dim + j) as usize];

        out[pos as usize] = val;
    }
}

/// Launch fused patch embedding + positional encoding + CLS prepend.
/// Input: eeg [1, C, T], chan_idx [1, C]
/// Output: [1, 1 + num_patches*C, embed_dim]
pub fn launch_fused_embed(
    eeg: CubeTensor<WgpuRuntime>,
    patch_w: CubeTensor<WgpuRuntime>,
    patch_b: CubeTensor<WgpuRuntime>,
    ch_emb_w: CubeTensor<WgpuRuntime>,
    ch_idx: CubeTensor<WgpuRuntime>,  // Int tensor
    temp_pe: CubeTensor<WgpuRuntime>,
    cls_with_pe: CubeTensor<WgpuRuntime>,
    n_channels: usize,
    n_samples: usize,
    patch_size: usize,
    embed_dim: usize,
) -> CubeTensor<WgpuRuntime> {
    let eeg = into_contiguous(eeg);
    let num_patches = n_samples / patch_size;
    let seq_with_cls = 1 + num_patches * n_channels;
    let out = empty_cube_shape(&eeg, vec![1, seq_with_cls, embed_dim]);

    let total = seq_with_cls * embed_dim;
    let cube_dim_x: u32 = 256;
    let cube_count_x = ((total as u32) + cube_dim_x - 1) / cube_dim_x;
    let cube_dim = CubeDim { x: cube_dim_x, y: 1, z: 1 };
    let cube_count = CubeCount::Static(cube_count_x, 1, 1);

    match eeg.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            fused_embed_kernel::launch::<f32, WgpuRuntime>(
                &eeg.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&eeg.handle, &eeg.strides, &eeg.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&patch_w.handle, &patch_w.strides, &patch_w.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&patch_b.handle, &patch_b.strides, &patch_b.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&ch_emb_w.handle, &ch_emb_w.strides, &ch_emb_w.shape.dims, 1),
                TensorArg::from_raw_parts::<u32>(&ch_idx.handle, &ch_idx.strides, &ch_idx.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&temp_pe.handle, &temp_pe.strides, &temp_pe.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&cls_with_pe.handle, &cls_with_pe.strides, &cls_with_pe.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(n_channels as u32),
                ScalarArg::new(n_samples as u32),
                ScalarArg::new(patch_size as u32),
                ScalarArg::new(embed_dim as u32),
                ScalarArg::new(num_patches as u32),
                ScalarArg::new(seq_with_cls as u32),
            ).expect("fused_embed f32 launch");
        },
        DType::F16 => unsafe {
            fused_embed_kernel::launch::<half::f16, WgpuRuntime>(
                &eeg.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&eeg.handle, &eeg.strides, &eeg.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&patch_w.handle, &patch_w.strides, &patch_w.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&patch_b.handle, &patch_b.strides, &patch_b.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&ch_emb_w.handle, &ch_emb_w.strides, &ch_emb_w.shape.dims, 1),
                TensorArg::from_raw_parts::<u32>(&ch_idx.handle, &ch_idx.strides, &ch_idx.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&temp_pe.handle, &temp_pe.strides, &temp_pe.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&cls_with_pe.handle, &cls_with_pe.strides, &cls_with_pe.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(n_channels as u32),
                ScalarArg::new(n_samples as u32),
                ScalarArg::new(patch_size as u32),
                ScalarArg::new(embed_dim as u32),
                ScalarArg::new(num_patches as u32),
                ScalarArg::new(seq_with_cls as u32),
            ).expect("fused_embed f16 launch");
        },
        dt => panic!("fused_embed: unsupported dtype {dt:?}"),
    }
    out
}

// ══════════════════════════════════════════════════════════════════════════
// 11) Tiled Flash Attention — fused Q×K^T → softmax → attn×V
//
//    Tiled flash attention with SharedMemory for K/V reuse across query rows.
//    Each workgroup processes 32 query rows simultaneously, tiling over
//    K/V blocks. K/V tiles loaded into shared memory once, reused by all
//    32 query rows (32× bandwidth reduction vs naive approach).
//
//    CubeDim = (32, 32, 1) = 1024 threads:
//      x=32: SIMD lanes for dh dot-product reduction via plane_sum
//      y=32: parallel query rows within the tile
//
//    Online softmax (Milakov–Gimelshein) across KV tiles.
//    Memory: ~150MB vs ~645MB decomposed (4.3× less bandwidth).
//    Dispatches: 1 vs 3 (matmul + softmax + matmul).
//
//    Q is pre-scaled by 1/sqrt(dh). Supports dh up to 128.
// ══════════════════════════════════════════════════════════════════════════

const TILE_Q: u32 = 32;
const TILE_KV: u32 = 32;
const DH_MAX: u32 = 128;

#[cube(launch)]
fn flash_attention_kernel<F: Float>(
    q: &Tensor<F>,        // [BH, S, dh] contiguous
    k: &Tensor<F>,        // [BH, S, dh] contiguous
    v: &Tensor<F>,        // [BH, S, dh] contiguous
    out: &mut Tensor<F>,  // [BH, S, dh] contiguous
    s_size: u32,
    dh_size: u32,
    #[comptime] smem_size: usize,
) {
    let bh = CUBE_POS_X;
    let q_tile_idx = CUBE_POS_Y;
    let lane = UNIT_POS_X;    // 0..31
    let q_local = UNIT_POS_Y; // 0..32-1

    let q_row = q_tile_idx * 32 + q_local;
    let bh_offset = bh * s_size * dh_size;

    // ── Shared memory for K/V tiles (cooperatively loaded) ──────────
    let mut k_shared = SharedMemory::<F>::new(smem_size as usize);
    let mut v_shared = SharedMemory::<F>::new(smem_size as usize);

    // ── Load Q row into registers ───────────────────────────────────
    let mut q_r0 = F::new(0.0);
    let mut q_r1 = F::new(0.0);
    let mut q_r2 = F::new(0.0);
    let mut q_r3 = F::new(0.0);
    if q_row < s_size {
        let qb = bh_offset + q_row * dh_size;
        if lane < dh_size { q_r0 = q[(qb + lane) as usize]; }
        let j1 = lane + 32;
        if j1 < dh_size { q_r1 = q[(qb + j1) as usize]; }
        let j2 = lane + 64;
        if j2 < dh_size { q_r2 = q[(qb + j2) as usize]; }
        let j3 = lane + 96;
        if j3 < dh_size { q_r3 = q[(qb + j3) as usize]; }
    }

    // ── Online softmax + output accumulators ────────────────────────
    let mut r_max = F::new(-65504.0);
    let mut r_sum = F::new(0.0);
    let mut o0 = F::new(0.0);
    let mut o1 = F::new(0.0);
    let mut o2 = F::new(0.0);
    let mut o3 = F::new(0.0);

    // Cooperative loading: 1024 threads load 32*dh_size elements
    let tid = q_local * 32 + lane;
    let n_threads = 32 * 32;
    let tile_elems = 32 * dh_size;
    let loads_per_thread = (tile_elems + n_threads - 1) / n_threads;

    let n_kv_tiles = (s_size + 32 - 1) / 32;

    for kv_tile in 0u32..n_kv_tiles {
        let kv_start = kv_tile * 32;

        // ── Load K tile into shared memory ──────────────────────────
        for ld in 0u32..loads_per_thread {
            let ei = tid + ld * n_threads;
            if ei < tile_elems {
                let r = ei / dh_size;
                let c = ei % dh_size;
                let gr = kv_start + r;
                if gr < s_size {
                    k_shared[ei as usize] = k[(bh_offset + gr * dh_size + c) as usize];
                } else {
                    k_shared[ei as usize] = F::new(0.0);
                }
            }
        }
        // ── Load V tile into shared memory ──────────────────────────
        for ld in 0u32..loads_per_thread {
            let ei = tid + ld * n_threads;
            if ei < tile_elems {
                let r = ei / dh_size;
                let c = ei % dh_size;
                let gr = kv_start + r;
                if gr < s_size {
                    v_shared[ei as usize] = v[(bh_offset + gr * dh_size + c) as usize];
                } else {
                    v_shared[ei as usize] = F::new(0.0);
                }
            }
        }

        sync_cube();

        // ── Process KV rows from shared memory ─────────────────────
        if q_row < s_size {
            let rows_this = if kv_start + 32u32 <= s_size { 32u32.into() } else { s_size - kv_start };

            for kv_l in 0u32..rows_this {
                let ko = kv_l * dh_size;

                // Dot product via shared K tile
                let mut dot = F::new(0.0);
                if lane < dh_size { dot += q_r0 * k_shared[(ko + lane) as usize]; }
                let j1 = lane + 32;
                if j1 < dh_size { dot += q_r1 * k_shared[(ko + j1) as usize]; }
                let j2 = lane + 64;
                if j2 < dh_size { dot += q_r2 * k_shared[(ko + j2) as usize]; }
                let j3 = lane + 96;
                if j3 < dh_size { dot += q_r3 * k_shared[(ko + j3) as usize]; }
                let score = plane_sum(dot);

                // Online softmax
                let pm = r_max;
                r_max = F::max(r_max, score);
                let corr = F::exp(pm - r_max);
                r_sum = r_sum * corr + F::exp(score - r_max);
                let w = F::exp(score - r_max);

                // Accumulate weighted V from shared memory
                let vo = kv_l * dh_size;
                if lane < dh_size { o0 = o0 * corr + w * v_shared[(vo + lane) as usize]; }
                if j1 < dh_size { o1 = o1 * corr + w * v_shared[(vo + j1) as usize]; }
                if j2 < dh_size { o2 = o2 * corr + w * v_shared[(vo + j2) as usize]; }
                if j3 < dh_size { o3 = o3 * corr + w * v_shared[(vo + j3) as usize]; }
            }
        }

        sync_cube();
    }

    // ── Write normalized output ─────────────────────────────────────
    if q_row < s_size {
        let inv = F::new(1.0) / r_sum;
        let ob = bh_offset + q_row * dh_size;
        if lane < dh_size { out[(ob + lane) as usize] = o0 * inv; }
        let j1 = lane + 32;
        if j1 < dh_size { out[(ob + j1) as usize] = o1 * inv; }
        let j2 = lane + 64;
        if j2 < dh_size { out[(ob + j2) as usize] = o2 * inv; }
        let j3 = lane + 96;
        if j3 < dh_size { out[(ob + j3) as usize] = o3 * inv; }
    }
}

/// Launch tiled flash attention.
/// Input: Q, K, V [B, H, S, dh] contiguous, Q pre-scaled by 1/sqrt(dh).
/// Output: O [B, H, S, dh]
pub fn launch_flash_attention(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    v: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let v = into_contiguous(v);
    let out = empty_cube(&q, q.shape.clone());

    let dims = &q.shape.dims;
    let b = dims[0];
    let h = dims[1];
    let s = dims[2];
    let dh = dims[3];
    assert!(dh <= DH_MAX as usize, "head dim {dh} > max {DH_MAX}");

    let bh = b * h;
    let n_q_tiles = ((s as u32) + TILE_Q - 1) / TILE_Q;

    let cube_dim = CubeDim { x: 32, y: TILE_Q, z: 1 };
    let cube_count = CubeCount::Static(bh as u32, n_q_tiles, 1);

    let shape_3d = vec![bh, s, dh];
    let strides_3d = contiguous_strides(&shape_3d);

    let smem_size = TILE_KV as usize * dh;

    match q.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            flash_attention_kernel::launch::<f32, WgpuRuntime>(
                &q.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&q.handle, &strides_3d, &shape_3d, 1),
                TensorArg::from_raw_parts::<f32>(&k.handle, &strides_3d, &shape_3d, 1),
                TensorArg::from_raw_parts::<f32>(&v.handle, &strides_3d, &shape_3d, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &strides_3d, &shape_3d, 1),
                ScalarArg::new(s as u32),
                ScalarArg::new(dh as u32),
                smem_size,
            ).expect("flash_attention f32 launch");
        },
        DType::F16 => unsafe {
            flash_attention_kernel::launch::<half::f16, WgpuRuntime>(
                &q.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&q.handle, &strides_3d, &shape_3d, 1),
                TensorArg::from_raw_parts::<half::f16>(&k.handle, &strides_3d, &shape_3d, 1),
                TensorArg::from_raw_parts::<half::f16>(&v.handle, &strides_3d, &shape_3d, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &strides_3d, &shape_3d, 1),
                ScalarArg::new(s as u32),
                ScalarArg::new(dh as u32),
                smem_size,
            ).expect("flash_attention f16 launch");
        },
        dt => panic!("flash_attention: unsupported dtype {dt:?}"),
    }
    out
}
