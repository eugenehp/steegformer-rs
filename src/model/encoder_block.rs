/// Transformer Encoder Block for ST-EEGFormer.
///
/// Python (via timm `Block`):
///   x = x + attn(norm1(x))
///   x = x + mlp(norm2(x))
///
/// Pre-norm architecture (same as standard ViT).

use burn::prelude::*;
use crate::model::norm::SteegLayerNorm;
use crate::model::attention::MultiHeadSelfAttention;
use crate::model::feedforward::FeedForward;

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    pub norm1: SteegLayerNorm<B>,
    pub attn:  MultiHeadSelfAttention<B>,
    pub norm2: SteegLayerNorm<B>,
    pub mlp:   FeedForward<B>,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn new(
        dim:       usize,
        n_heads:   usize,
        mlp_ratio: f64,
        qkv_bias:  bool,
        norm_eps:  f64,
        device:    &B::Device,
    ) -> Self {
        let hidden_dim = (dim as f64 * mlp_ratio) as usize;
        Self {
            norm1: SteegLayerNorm::new(dim, norm_eps, device),
            attn:  MultiHeadSelfAttention::new(dim, n_heads, qkv_bias, device),
            norm2: SteegLayerNorm::new(dim, norm_eps, device),
            mlp:   FeedForward::new(dim, hidden_dim, device),
        }
    }

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = x.clone() + self.attn.forward(self.norm1.forward(x.clone()));
        h.clone() + self.mlp.forward(self.norm2.forward(h))
    }
}
