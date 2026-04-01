/// Feed-Forward Network (MLP) for ST-EEGFormer transformer blocks.
///
/// Python (via timm Block): Linear(dim, hidden_dim) → GELU → Linear(hidden_dim, dim)
///
/// With `wgpu-kernels`: GELU uses a fused CubeCL kernel (1 dispatch vs ~3).

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(dim, hidden_dim).with_bias(true).init(device),
            fc2: LinearConfig::new(hidden_dim, dim).with_bias(true).init(device),
        }
    }
}

// ── Standard forward ──────────────────────────────────────────────────────
#[cfg(not(feature = "wgpu-kernels"))]
impl<B: Backend> FeedForward<B> {
    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = burn::tensor::activation::gelu(self.fc1.forward(x));
        self.fc2.forward(h)
    }
}

// ── Fused forward (CubeCL GELU) ──────────────────────────────────────────
#[cfg(feature = "wgpu-kernels")]
impl<B: Backend + super::FusedOps> FeedForward<B> {
    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // FC1: 3D batched matmul without bias, then fused bias+GELU in one kernel
        let w = self.fc1.weight.val()   // [dim, hidden_dim]
            .unsqueeze_dim::<3>(0);     // [1, dim, hidden_dim]
        let h = x.matmul(w);            // [B, S, hidden_dim]
        let fc1_bias = self.fc1.bias.as_ref().expect("fc1 must have bias").val();
        let h = B::fused_bias_gelu(h, fc1_bias);
        self.fc2.forward(h)
    }

    /// Forward returning FC2 matmul output WITHOUT bias (for fused_bias_residual_add).
    pub fn forward_no_fc2_bias(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // FC1: 3D batched matmul
        let w1 = self.fc1.weight.val()  // [dim, hidden_dim]
            .unsqueeze_dim::<3>(0);     // [1, dim, hidden_dim]
        let h = x.matmul(w1);           // [B, S, hidden_dim]
        let fc1_bias = self.fc1.bias.as_ref().expect("fc1 must have bias").val();
        let h = B::fused_bias_gelu(h, fc1_bias);

        // FC2: 3D batched matmul, no bias
        let w2 = self.fc2.weight.val()  // [hidden_dim, dim]
            .unsqueeze_dim::<3>(0);     // [1, hidden_dim, dim]
        h.matmul(w2)                    // [B, S, dim]
    }
}
