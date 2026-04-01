/// `FusedOps` — backend-dispatch trait for custom CubeCL kernels.
///
/// Model layers call `B::fused_*()` methods.
/// - `CubeBackend<WgpuRuntime, f32/f16>` → custom CubeCL kernels.
/// - All other backends → standard burn tensor ops.

use burn::prelude::*;
#[cfg(feature = "wgpu-kernels")]
use burn::tensor::TensorPrimitive;

// ── Trait ─────────────────────────────────────────────────────────────────

pub trait FusedOps: Backend {
    /// Fused LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    fn fused_layernorm(
        x: Tensor<Self, 3>,
        weight: Tensor<Self, 1>,
        bias: Tensor<Self, 1>,
        eps: f32,
    ) -> Tensor<Self, 3>;

    /// Fused add + layernorm: returns (a+b, layernorm(a+b)).
    /// Saves one dispatch vs separate add + layernorm.
    fn fused_add_layernorm(
        a: Tensor<Self, 3>,
        b: Tensor<Self, 3>,
        weight: Tensor<Self, 1>,
        bias: Tensor<Self, 1>,
        eps: f32,
    ) -> (Tensor<Self, 3>, Tensor<Self, 3>);

    /// Fused softmax along last dimension.
    /// Supports any last-dim size (not just multiples of 32).
    fn fused_softmax(x: Tensor<Self, 4>, dim: usize) -> Tensor<Self, 4>;

    /// Fused GELU activation.
    fn fused_gelu(x: Tensor<Self, 3>) -> Tensor<Self, 3>;

    /// Fused bias + GELU: out = GELU(x + bias).
    /// x: [B, S, hidden_dim] (matmul output, no bias), bias: [hidden_dim].
    fn fused_bias_gelu(x: Tensor<Self, 3>, bias: Tensor<Self, 1>) -> Tensor<Self, 3>;

    /// Split QKV tensor [B, S, 3D] into Q, K, V each [B, H, S, dh] (contiguous).
    /// Fuses: bias add + split + Q scaling in one dispatch.
    /// qkv: [B, S, 3D] (matmul output, no bias), bias: [3D]
    fn fused_split_qkv_scaled(
        qkv: Tensor<Self, 3>,
        bias: Tensor<Self, 1>,
        n_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> (Tensor<Self, 4>, Tensor<Self, 4>, Tensor<Self, 4>);

    /// Merge attention heads: [B, H, S, dh] → [B, S, D].
    fn fused_merge_heads(
        x: Tensor<Self, 4>,
        n_heads: usize,
        head_dim: usize,
    ) -> Tensor<Self, 3>;

    /// Fused bias + residual add: out = residual + matmul_out + bias.
    fn fused_bias_residual_add(
        residual: Tensor<Self, 3>,
        matmul_out: Tensor<Self, 3>,
        bias: Tensor<Self, 1>,
    ) -> Tensor<Self, 3>;

    /// Fused bias + residual add + layernorm:
    /// sum = residual + matmul_out + bias
    /// norm = layernorm(sum) * ln_weight + ln_bias
    /// Returns (sum, norm). Saves 1 dispatch vs separate bias_residual_add + layernorm.
    fn fused_bias_residual_add_layernorm(
        residual: Tensor<Self, 3>,
        matmul_out: Tensor<Self, 3>,
        bias: Tensor<Self, 1>,
        ln_weight: Tensor<Self, 1>,
        ln_bias: Tensor<Self, 1>,
        eps: f32,
    ) -> (Tensor<Self, 3>, Tensor<Self, 3>);

    /// Flash attention: fused Q×K^T → softmax → attn×V in one kernel.
    /// Q must be pre-scaled. K_T is pre-transposed: [B, H, dh, S].
    /// Q: [B, H, S, dh], K_T: [B, H, dh, S], V: [B, H, S, dh] → O: [B, H, S, dh]
    fn fused_flash_attention(
        q: Tensor<Self, 4>,
        k_t: Tensor<Self, 4>,
        v: Tensor<Self, 4>,
    ) -> Tensor<Self, 4>;
}

// ── Standard burn-ops fallback ────────────────────────────────────────────

fn layernorm_standard<B: Backend>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);
    let diff = x.clone() - mean;
    let var = diff.clone().powf_scalar(2.0).mean_dim(2);
    let inv_std = (var + eps).powf_scalar(-0.5);
    let d = weight.dims()[0];
    diff * inv_std * weight.reshape([1, 1, d]) + bias.reshape([1, 1, d])
}

fn add_layernorm_standard<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f32,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let sum = a + b;
    let norm = layernorm_standard(sum.clone(), weight, bias, eps);
    (sum, norm)
}

fn softmax_standard<B: Backend>(x: Tensor<B, 4>, dim: usize) -> Tensor<B, 4> {
    burn::tensor::activation::softmax(x, dim)
}

fn gelu_standard<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    burn::tensor::activation::gelu(x)
}

fn bias_gelu_standard<B: Backend>(x: Tensor<B, 3>, bias: Tensor<B, 1>) -> Tensor<B, 3> {
    let d = bias.dims()[0];
    let x = x + bias.reshape([1, 1, d]);
    burn::tensor::activation::gelu(x)
}

fn split_qkv_scaled_standard<B: Backend>(
    qkv: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let [b, s, three_d] = qkv.dims();
    let dim = n_heads * head_dim;
    let qkv = qkv + bias.reshape([1, 1, three_d]);
    let q = qkv.clone().narrow(2, 0, dim)
        .reshape([b, s, n_heads, head_dim]).swap_dims(1, 2)
        .mul_scalar(scale);
    // K in standard layout [B, H, S, dh] — swap_dims at matmul time preserves contiguity
    let k = qkv.clone().narrow(2, dim, dim)
        .reshape([b, s, n_heads, head_dim]).swap_dims(1, 2);
    let v = qkv.narrow(2, dim * 2, dim)
        .reshape([b, s, n_heads, head_dim]).swap_dims(1, 2);
    (q, k, v)
}

fn merge_heads_standard<B: Backend>(
    x: Tensor<B, 4>,
    _n_heads: usize,
    _head_dim: usize,
) -> Tensor<B, 3> {
    let [b, _h, s, _dh] = x.dims();
    let d = _n_heads * _head_dim;
    x.swap_dims(1, 2).flatten::<3>(2, 3).reshape([b, s, d])
}

fn bias_residual_add_standard<B: Backend>(
    residual: Tensor<B, 3>,
    matmul_out: Tensor<B, 3>,
    bias: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let d = bias.dims()[0];
    residual + matmul_out + bias.reshape([1, 1, d])
}

fn bias_residual_add_layernorm_standard<B: Backend>(
    residual: Tensor<B, 3>,
    matmul_out: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    ln_weight: Tensor<B, 1>,
    ln_bias: Tensor<B, 1>,
    eps: f32,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let sum = bias_residual_add_standard(residual, matmul_out, bias);
    let norm = layernorm_standard(sum.clone(), ln_weight, ln_bias, eps);
    (sum, norm)
}

fn flash_attention_standard<B: Backend>(
    q: Tensor<B, 4>,
    k_t: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    // Fallback: decomposed Q×K^T → softmax → attn×V
    // k_t may be [B,H,dh,S] (swap_dims'd) or already transposed
    let scores = q.matmul(k_t);
    let attn = burn::tensor::activation::softmax(scores, 3);
    attn.matmul(v)
}

// ── CubeBackend (no-fusion) → custom CubeCL kernels ──────────────────────

#[cfg(feature = "wgpu-kernels")]
mod cube_impls {
    use super::*;
    use burn::backend::wgpu::{CubeBackend, CubeTensor};
    use burn_cubecl::{FloatElement, IntElement, BoolElement};
    use cubecl::wgpu::WgpuRuntime;
    use crate::model::kernels;

    fn cube<B, const D: usize>(t: Tensor<B, D>) -> CubeTensor<WgpuRuntime>
    where
        B: Backend,
        B::FloatTensorPrimitive: Into<CubeTensor<WgpuRuntime>>,
    {
        match t.into_primitive() {
            TensorPrimitive::Float(p) => p.into(),
            _ => panic!("expected float tensor"),
        }
    }

    fn wrap<B, const D: usize>(c: CubeTensor<WgpuRuntime>) -> Tensor<B, D>
    where
        B: Backend,
        CubeTensor<WgpuRuntime>: Into<B::FloatTensorPrimitive>,
    {
        Tensor::from_primitive(TensorPrimitive::Float(c.into()))
    }

    impl<F, I, BT> FusedOps for CubeBackend<WgpuRuntime, F, I, BT>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
        <Self as Backend>::FloatTensorPrimitive: Into<CubeTensor<WgpuRuntime>>,
        CubeTensor<WgpuRuntime>: Into<<Self as Backend>::FloatTensorPrimitive>,
    {
        fn fused_layernorm(
            x: Tensor<Self, 3>,
            weight: Tensor<Self, 1>,
            bias: Tensor<Self, 1>,
            eps: f32,
        ) -> Tensor<Self, 3> {
            let last_dim = x.dims()[2];
            if last_dim % 32 == 0 {
                wrap(kernels::launch_layernorm(cube(x), cube(weight), cube(bias), eps))
            } else {
                layernorm_standard(x, weight, bias, eps)
            }
        }

        fn fused_add_layernorm(
            a: Tensor<Self, 3>,
            b: Tensor<Self, 3>,
            weight: Tensor<Self, 1>,
            bias: Tensor<Self, 1>,
            eps: f32,
        ) -> (Tensor<Self, 3>, Tensor<Self, 3>) {
            let last_dim = a.dims()[2];
            if last_dim % 32 == 0 {
                let (s, n) = kernels::launch_add_layernorm(
                    cube(a), cube(b), cube(weight), cube(bias), eps,
                );
                (wrap(s), wrap(n))
            } else {
                add_layernorm_standard(a, b, weight, bias, eps)
            }
        }

        fn fused_softmax(x: Tensor<Self, 4>, dim: usize) -> Tensor<Self, 4> {
            // Now supports any last-dim size (not just multiples of 32)
            if dim == 3 {
                wrap(kernels::launch_softmax(cube(x), dim))
            } else {
                burn::tensor::activation::softmax(x, dim)
            }
        }

        fn fused_gelu(x: Tensor<Self, 3>) -> Tensor<Self, 3> {
            wrap(kernels::launch_gelu(cube(x)))
        }

        fn fused_bias_gelu(x: Tensor<Self, 3>, bias: Tensor<Self, 1>) -> Tensor<Self, 3> {
            wrap(kernels::launch_bias_gelu(cube(x), cube(bias)))
        }

        fn fused_split_qkv_scaled(
            qkv: Tensor<Self, 3>,
            bias: Tensor<Self, 1>,
            n_heads: usize,
            head_dim: usize,
            scale: f32,
        ) -> (Tensor<Self, 4>, Tensor<Self, 4>, Tensor<Self, 4>) {
            let (q, k, v) = kernels::launch_split_qkv_scaled(
                cube(qkv), cube(bias), n_heads, head_dim, scale,
            );
            (wrap(q), wrap(k), wrap(v))
        }

        fn fused_merge_heads(
            x: Tensor<Self, 4>,
            n_heads: usize,
            head_dim: usize,
        ) -> Tensor<Self, 3> {
            wrap(kernels::launch_merge_heads(cube(x), n_heads, head_dim))
        }

        fn fused_bias_residual_add(
            residual: Tensor<Self, 3>,
            matmul_out: Tensor<Self, 3>,
            bias: Tensor<Self, 1>,
        ) -> Tensor<Self, 3> {
            wrap(kernels::launch_bias_residual_add(
                cube(residual), cube(matmul_out), cube(bias),
            ))
        }

        fn fused_bias_residual_add_layernorm(
            residual: Tensor<Self, 3>,
            matmul_out: Tensor<Self, 3>,
            bias: Tensor<Self, 1>,
            ln_weight: Tensor<Self, 1>,
            ln_bias: Tensor<Self, 1>,
            eps: f32,
        ) -> (Tensor<Self, 3>, Tensor<Self, 3>) {
            let last_dim = residual.dims()[2];
            if last_dim % 32 == 0 {
                let (s, n) = kernels::launch_bias_residual_add_layernorm(
                    cube(residual), cube(matmul_out), cube(bias),
                    cube(ln_weight), cube(ln_bias), eps,
                );
                (wrap(s), wrap(n))
            } else {
                bias_residual_add_layernorm_standard(residual, matmul_out, bias, ln_weight, ln_bias, eps)
            }
        }

        fn fused_flash_attention(
            q: Tensor<Self, 4>,
            k_t: Tensor<Self, 4>,
            v: Tensor<Self, 4>,
        ) -> Tensor<Self, 4> {
            // Decomposed: Q×K_T → fused_softmax → attn×V
            // k_t is already [B, H, dh, S] so no swap_dims copy needed.
            // The autotuned matmul kernels outperform our custom flash attention
            // kernel for S≤1057 because they use register tiling.
            let scores = q.matmul(k_t);  // [B, H, S, S]
            let attn = Self::fused_softmax(scores, 3);
            attn.matmul(v)  // [B, H, S, dh]
        }
    }
}

// ── Wgpu (fusion wrapper) and NdArray → burn-ops fallback ─────────────────

macro_rules! impl_standard {
    ($ty:ty) => {
        impl FusedOps for $ty {
            fn fused_layernorm(
                x: Tensor<Self, 3>,
                weight: Tensor<Self, 1>,
                bias: Tensor<Self, 1>,
                eps: f32,
            ) -> Tensor<Self, 3> {
                layernorm_standard(x, weight, bias, eps)
            }
            fn fused_add_layernorm(
                a: Tensor<Self, 3>,
                b: Tensor<Self, 3>,
                weight: Tensor<Self, 1>,
                bias: Tensor<Self, 1>,
                eps: f32,
            ) -> (Tensor<Self, 3>, Tensor<Self, 3>) {
                add_layernorm_standard(a, b, weight, bias, eps)
            }
            fn fused_softmax(x: Tensor<Self, 4>, dim: usize) -> Tensor<Self, 4> {
                softmax_standard(x, dim)
            }
            fn fused_gelu(x: Tensor<Self, 3>) -> Tensor<Self, 3> {
                gelu_standard(x)
            }
            fn fused_bias_gelu(x: Tensor<Self, 3>, bias: Tensor<Self, 1>) -> Tensor<Self, 3> {
                bias_gelu_standard(x, bias)
            }
            fn fused_split_qkv_scaled(
                qkv: Tensor<Self, 3>,
                bias: Tensor<Self, 1>,
                n_heads: usize,
                head_dim: usize,
                scale: f32,
            ) -> (Tensor<Self, 4>, Tensor<Self, 4>, Tensor<Self, 4>) {
                split_qkv_scaled_standard(qkv, bias, n_heads, head_dim, scale)
            }
            fn fused_merge_heads(
                x: Tensor<Self, 4>,
                n_heads: usize,
                head_dim: usize,
            ) -> Tensor<Self, 3> {
                merge_heads_standard(x, n_heads, head_dim)
            }
            fn fused_bias_residual_add(
                residual: Tensor<Self, 3>,
                matmul_out: Tensor<Self, 3>,
                bias: Tensor<Self, 1>,
            ) -> Tensor<Self, 3> {
                bias_residual_add_standard(residual, matmul_out, bias)
            }
            fn fused_bias_residual_add_layernorm(
                residual: Tensor<Self, 3>,
                matmul_out: Tensor<Self, 3>,
                bias: Tensor<Self, 1>,
                ln_weight: Tensor<Self, 1>,
                ln_bias: Tensor<Self, 1>,
                eps: f32,
            ) -> (Tensor<Self, 3>, Tensor<Self, 3>) {
                bias_residual_add_layernorm_standard(residual, matmul_out, bias, ln_weight, ln_bias, eps)
            }
            fn fused_flash_attention(
                q: Tensor<Self, 4>,
                k_t: Tensor<Self, 4>,
                v: Tensor<Self, 4>,
            ) -> Tensor<Self, 4> {
                flash_attention_standard(q, k_t, v)
            }
        }
    };
}

#[cfg(feature = "wgpu")]
impl_standard!(burn::backend::Wgpu<f32, i32>);
#[cfg(feature = "wgpu")]
impl_standard!(burn::backend::Wgpu<half::f16, i32>);
#[cfg(feature = "ndarray")]
impl_standard!(burn::backend::NdArray<f32>);
