/// EEG Patch Embedding for ST-EEGFormer.
///
/// Python: `PatchEmbedEEG` in models_mae_eeg.py / models_vit_eeg.py
///
/// Input:  (B, C, T) — multi-channel EEG
/// Step 1: Unfold each channel into non-overlapping patches of size `patch_size`
///         → (B, C * num_patches, patch_size)
/// Step 2: Linear projection: patch_size → embed_dim
///         → (B, C * num_patches, embed_dim)
///
/// The "patchify" order is:
///   For each time patch t: for each channel c: patch[t, c]
///   i.e., shape (B, num_patches, C, patch_size) → flatten to (B, num_patches * C, embed_dim)
///
/// This creates a 2D spatio-temporal token sequence where both time and
/// channel information is encoded via additive positional embeddings.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct PatchEmbedEEG<B: Backend> {
    pub proj: Linear<B>,
    pub patch_size: usize,
    pub embed_dim:  usize,
}

impl<B: Backend> PatchEmbedEEG<B> {
    pub fn new(patch_size: usize, embed_dim: usize, device: &B::Device) -> Self {
        Self {
            proj: LinearConfig::new(patch_size, embed_dim).with_bias(true).init(device),
            patch_size,
            embed_dim,
        }
    }

    /// Patchify and embed EEG signals.
    ///
    /// x: [B, C, T] → returns [B, num_patches, C, embed_dim]
    ///
    /// The intermediate (before flattening) shape is kept so that the caller
    /// can build separate temporal and channel positional indices.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, c, t] = x.dims();
        let num_patches = t / self.patch_size;

        // Patchify: [B, C, T] → [B, num_patches, C, patch_size]
        let patches = self.patchify(x, b, c, num_patches);

        // Linear projection: [B, num_patches, C, patch_size] → [B, num_patches, C, embed_dim]
        self.proj.forward(patches)
    }

    /// Patchify without embedding (used for loss computation).
    ///
    /// x: [B, C, T] → [B, num_patches, C, patch_size]
    pub fn patchify(&self, x: Tensor<B, 3>, b: usize, c: usize, num_patches: usize) -> Tensor<B, 4> {
        // [B, C, T] → [B, C, num_patches, patch_size]
        let x = x.reshape([b, c, num_patches, self.patch_size]);

        // → [B, num_patches, C, patch_size]
        x.swap_dims(1, 2)
    }
}
