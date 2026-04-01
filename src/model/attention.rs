/// Multi-Head Self-Attention for ST-EEGFormer.
///
/// With `wgpu-kernels`:
///   - Fused QKV split+scale (1 dispatch vs ~7: copy dispatches + mul_scalar)
///   - Fused softmax (1 dispatch vs ~5, supports any sequence length)
///   - Fused merge heads (1 dispatch vs 1 copy dispatch)

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    pub qkv:      Linear<B>,
    pub proj:     Linear<B>,
    pub n_heads:  usize,
    pub head_dim: usize,
    pub scale:    f32,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn new(
        dim: usize,
        n_heads: usize,
        qkv_bias: bool,
        device: &B::Device,
    ) -> Self {
        let head_dim = dim / n_heads;
        Self {
            qkv:  LinearConfig::new(dim, dim * 3).with_bias(qkv_bias).init(device),
            proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }
}

// ── Standard forward ──────────────────────────────────────────────────────
#[cfg(not(feature = "wgpu-kernels"))]
impl<B: Backend> MultiHeadSelfAttention<B> {
    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);
        let dim = h * dh;

        let qkv = self.qkv.forward(x);
        let q = qkv.clone().narrow(2, 0, dim)
            .reshape([b, s, h, dh]).swap_dims(1, 2);
        let k = qkv.clone().narrow(2, dim, dim)
            .reshape([b, s, h, dh]).swap_dims(1, 2);
        let v = qkv.narrow(2, dim * 2, dim)
            .reshape([b, s, h, dh]).swap_dims(1, 2);

        let q = q.mul_scalar(self.scale);
        let attn = burn::tensor::activation::softmax(q.matmul(k.swap_dims(2, 3)), 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).flatten(2, 3);
        self.proj.forward(out)
    }
}

// ── Fused forward (CubeCL kernels) ───────────────────────────────────────
#[cfg(feature = "wgpu-kernels")]
impl<B: Backend + super::FusedOps> MultiHeadSelfAttention<B> {
    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (h, dh) = (self.n_heads, self.head_dim);

        // QKV matmul (no bias) — bias is fused into split kernel
        // Use 3D batched matmul: [B, S, D] × [D, 3D] → [B, S, 3D]
        // Avoids 2 reshape dispatches vs the [B*S, D] approach.
        let qkv_w = self.qkv.weight.val() // [D, 3D]
            .unsqueeze_dim::<3>(0);       // [1, D, 3D] — broadcasts over batch
        let qkv = x.matmul(qkv_w);       // [B, S, 3D]
        let qkv_bias = self.qkv.bias.as_ref().expect("qkv must have bias").val();

        // Fused: bias + split + scale + K transpose in 1 dispatch.
        // Q: [B, H, S, dh], K_T: [B, H, dh, S] (pre-transposed!), V: [B, H, S, dh]
        // K is output transposed to avoid a separate swap_dims + into_contiguous copy.
        let (q, k_t, v) = B::fused_split_qkv_scaled(qkv, qkv_bias, h, dh, self.scale);

        // Attention: Q × K^T → softmax → attn × V
        // k_t is already [B, H, dh, S], no swap_dims copy needed!
        // Use fused flash attention when available (1 dispatch vs 3)
        let out = B::fused_flash_attention(q, k_t, v);  // [B, H, S, dh]

        // Fused merge heads: [B, H, S, dh] → [B, S, D] (1 dispatch)
        let out = B::fused_merge_heads(out, h, dh);

        // Projection: matmul only (no bias), bias fused with residual in encoder block
        // 3D batched: [B, S, D] × [1, D, D] → [B, S, D]
        let w = self.proj.weight.val()  // [D, D]
            .unsqueeze_dim::<3>(0);     // [1, D, D]
        out.matmul(w)
    }

    /// Get projection bias for fused_bias_residual_add.
    pub fn proj_bias(&self) -> Tensor<B, 1> {
        self.proj.bias.as_ref().expect("proj must have bias").val()
    }
}
