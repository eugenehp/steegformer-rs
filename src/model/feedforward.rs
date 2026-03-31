/// Feed-Forward Network (MLP) for ST-EEGFormer transformer blocks.
///
/// Python (via timm Block): Linear(dim, hidden_dim) → GELU → Linear(hidden_dim, dim)
///
/// timm's Block uses `Mlp` which by default has no internal LayerNorm (unlike LUNA).

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;

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

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = gelu(self.fc1.forward(x));
        self.fc2.forward(h)
    }
}
