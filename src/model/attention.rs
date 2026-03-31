/// Multi-Head Self-Attention for ST-EEGFormer.
///
/// Manual QK^T + softmax + V implementation:
///   - cubecl flash-attention hardcodes causal=true (wrong for bidirectional encoder)
///   - For S=193 tokens, naive SDPA is fast enough — no flash-attention needed
///   - Scale fused into Q (one mul_scalar vs post-matmul div)

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;

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

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);
        let dim = h * dh;

        // Fused QKV projection: [B, S, 3*dim]
        let qkv = self.qkv.forward(x);

        // Split via narrow (view-level, no copy)
        let q = qkv.clone().narrow(2, 0, dim)
            .reshape([b, s, h, dh]).swap_dims(1, 2);
        let k = qkv.clone().narrow(2, dim, dim)
            .reshape([b, s, h, dh]).swap_dims(1, 2);
        let v = qkv.narrow(2, dim * 2, dim)
            .reshape([b, s, h, dh]).swap_dims(1, 2);

        // Fuse scale into Q (saves one kernel dispatch vs post-matmul div)
        let q = q.mul_scalar(self.scale);

        // Attention: softmax(Q·K^T) · V
        let attn = softmax(q.matmul(k.swap_dims(2, 3)), 3);
        let out = attn.matmul(v);

        // [B, H, S, dh] → [B, S, dim]
        let out = out.swap_dims(1, 2).flatten(2, 3);
        self.proj.forward(out)
    }
}
