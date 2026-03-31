/// Multi-Head Self-Attention for ST-EEGFormer.
///
/// Python (via timm Block): standard ViT attention with qkv_bias=True.
///   qkv = Linear(dim, 3*dim) → reshape → [3, B, H, S, D]
///   attn = softmax(q @ k^T / sqrt(d)) @ v
///   output = proj(concat_heads(attn))
///
/// No rotary embeddings — ST-EEGFormer uses additive positional encodings.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    pub qkv:      Linear<B>,
    pub proj:     Linear<B>,
    pub n_heads:  usize,
    pub head_dim: usize,
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
        }
    }

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);
        let dim = h * dh;

        // QKV projection: [B, S, 3*dim]
        let qkv = self.qkv.forward(x);

        // Split into Q, K, V: each [B, S, dim]
        let q = qkv.clone().narrow(2, 0, dim);
        let k = qkv.clone().narrow(2, dim, dim);
        let v = qkv.narrow(2, dim * 2, dim);

        // Reshape to multi-head: [B, S, H, D] → [B, H, S, D]
        let q = q.reshape([b, s, h, dh]).swap_dims(1, 2);
        let k = k.reshape([b, s, h, dh]).swap_dims(1, 2);
        let v = v.reshape([b, s, h, dh]).swap_dims(1, 2);

        // Scaled dot-product attention
        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v);  // [B, H, S, D]

        // Reshape back: [B, S, dim]
        let out = out.swap_dims(1, 2).reshape([b, s, dim]);
        self.proj.forward(out)
    }
}
