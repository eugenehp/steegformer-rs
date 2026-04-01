/// LayerNorm wrapper for ST-EEGFormer.

use burn::prelude::*;
use burn::nn::{LayerNorm, LayerNormConfig};

#[derive(Module, Debug)]
pub struct SteegLayerNorm<B: Backend> {
    pub inner: LayerNorm<B>,
    /// Stored eps for fused kernel access (LayerNorm.epsilon is private).
    #[module(skip)]
    pub eps: f64,
}

impl<B: Backend> SteegLayerNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            inner: LayerNormConfig::new(dim).with_epsilon(eps).init(device),
            eps,
        }
    }

    /// Generic forward for any rank (always uses burn's LayerNorm).
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(x)
    }
}
