/// Transformer Encoder Block for ST-EEGFormer.
///
/// Pre-norm architecture:
///   x = x + attn(norm1(x))
///   x = x + mlp(norm2(x))
///
/// With `wgpu-kernels` — fused pipeline:
///   - LayerNorm: fused single-pass CubeCL kernel
///   - QKV split+scale: fused kernel (eliminates copy dispatches)
///   - Softmax: fused kernel (any seq length)
///   - Merge heads: fused kernel
///   - GELU: fused kernel

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
}

// ── Standard forward (no fused kernels) ───────────────────────────────────
#[cfg(not(feature = "wgpu-kernels"))]
impl<B: Backend> EncoderBlock<B> {
    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = x.clone() + self.attn.forward(self.norm1.forward(x.clone()));
        h.clone() + self.mlp.forward(self.norm2.forward(h))
    }
}

// ── Fused forward (CubeCL kernels) ────────────────────────────────────────
#[cfg(feature = "wgpu-kernels")]
impl<B: Backend + super::FusedOps> EncoderBlock<B> {
    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Fused LayerNorm (norm1)
        let n1 = fused_layernorm::<B>(&self.norm1, x.clone());

        // Attention: returns matmul output without proj bias
        let attn_matmul = self.attn.forward(n1);
        let proj_bias = self.attn.proj_bias();

        // Fused: residual + attn_matmul + proj_bias + layernorm(sum)
        // 1 dispatch instead of 2 (bias_residual_add + layernorm)
        let ln2_weight = self.norm2.inner.gamma.val();
        let ln2_bias = self.norm2.inner.beta.as_ref().expect("norm2 must have bias").val();
        let (h_sum, n2) = B::fused_bias_residual_add_layernorm(
            x, attn_matmul, proj_bias,
            ln2_weight, ln2_bias, self.norm2.eps as f32,
        );

        // Fused MLP (with fused bias+GELU) — returns FC2 matmul without bias
        let mlp_matmul = self.mlp.forward_no_fc2_bias(n2);
        let fc2_bias = self.mlp.fc2.bias.as_ref().expect("fc2 must have bias").val();

        // Fused: residual + FC2_matmul + FC2_bias in 1 dispatch
        B::fused_bias_residual_add(h_sum, mlp_matmul, fc2_bias)
    }

    /// Forward where caller provides pre-computed norm1(x).
    /// Returns block output.
    pub fn forward_with_norm1(&self, x: Tensor<B, 3>, n1: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn_matmul = self.attn.forward(n1);
        let proj_bias = self.attn.proj_bias();

        let ln2_weight = self.norm2.inner.gamma.val();
        let ln2_bias = self.norm2.inner.beta.as_ref().expect("norm2 must have bias").val();
        let (h_sum, n2) = B::fused_bias_residual_add_layernorm(
            x, attn_matmul, proj_bias,
            ln2_weight, ln2_bias, self.norm2.eps as f32,
        );

        let mlp_matmul = self.mlp.forward_no_fc2_bias(n2);
        let fc2_bias = self.mlp.fc2.bias.as_ref().expect("fc2 must have bias").val();

        B::fused_bias_residual_add(h_sum, mlp_matmul, fc2_bias)
    }

    /// Forward where caller provides pre-computed norm1(x).
    /// Fuses FC2 residual add with next block's norm1 using provided weights.
    /// Returns (block_output, norm1_of_output).
    pub fn forward_with_norm1_chain(
        &self,
        x: Tensor<B, 3>,
        n1: Tensor<B, 3>,
        next_ln_weight: Tensor<B, 1>,
        next_ln_bias: Tensor<B, 1>,
        next_ln_eps: f32,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let attn_matmul = self.attn.forward(n1);
        let proj_bias = self.attn.proj_bias();

        let ln2_weight = self.norm2.inner.gamma.val();
        let ln2_bias = self.norm2.inner.beta.as_ref().expect("norm2 must have bias").val();
        let (h_sum, n2) = B::fused_bias_residual_add_layernorm(
            x, attn_matmul, proj_bias,
            ln2_weight, ln2_bias, self.norm2.eps as f32,
        );

        let mlp_matmul = self.mlp.forward_no_fc2_bias(n2);
        let fc2_bias = self.mlp.fc2.bias.as_ref().expect("fc2 must have bias").val();

        // Fused: residual + FC2 + FC2_bias + next_block_norm1
        B::fused_bias_residual_add_layernorm(
            h_sum, mlp_matmul, fc2_bias,
            next_ln_weight, next_ln_bias, next_ln_eps,
        )
    }
}

/// Extract weight/bias from LayerNorm and call fused kernel.
#[cfg(feature = "wgpu-kernels")]
fn fused_layernorm<B: Backend + super::FusedOps>(
    ln: &SteegLayerNorm<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let weight = ln.inner.gamma.val();
    let bias = ln.inner.beta.as_ref().expect("LayerNorm must have bias").val();
    let eps = ln.eps as f32;
    B::fused_layernorm(x, weight, bias, eps)
}

/// Fused add + layernorm: computes (a+b, layernorm(a+b)) in one dispatch.
#[cfg(feature = "wgpu-kernels")]
fn fused_add_layernorm_block<B: Backend + super::FusedOps>(
    ln: &SteegLayerNorm<B>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let weight = ln.inner.gamma.val();
    let bias = ln.inner.beta.as_ref().expect("LayerNorm must have bias").val();
    let eps = ln.eps as f32;
    B::fused_add_layernorm(a, b, weight, bias, eps)
}
