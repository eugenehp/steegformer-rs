/// Data preparation for ST-EEGFormer inference.
///
/// ST-EEGFormer input:
///   - Signal: [B, C, T] — EEG data at 128 Hz
///   - Channel indices: [B, C] — integer indices into the channel embedding table
///
/// Unlike LUNA, ST-EEGFormer does NOT use 3D electrode positions.
/// Channel identity is encoded purely through learned embeddings.

use burn::prelude::*;

/// A single prepared input batch for the ST-EEGFormer model.
pub struct InputBatch<B: Backend> {
    /// EEG signal: [1, C, T].
    pub signal: Tensor<B, 3>,
    /// Channel embedding indices: [1, C] (Int tensor).
    pub channel_indices: Tensor<B, 2, Int>,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of time samples.
    pub n_samples: usize,
}

/// Build an InputBatch from raw arrays.
pub fn build_batch<B: Backend>(
    signal: Vec<f32>,          // [C, T] row-major
    channel_indices: Vec<i64>, // [C] indices into embedding table
    n_channels: usize,
    n_samples: usize,
    device: &B::Device,
) -> InputBatch<B> {
    let signal = Tensor::<B, 2>::from_data(
        TensorData::new(signal, vec![n_channels, n_samples]), device,
    ).unsqueeze_dim::<3>(0);  // [1, C, T]

    let channel_indices = Tensor::<B, 1, Int>::from_data(
        TensorData::new(channel_indices, vec![n_channels]), device,
    ).unsqueeze_dim::<2>(0);  // [1, C]

    InputBatch {
        signal,
        channel_indices,
        n_channels,
        n_samples,
    }
}

/// Build an InputBatch from channel name strings.
///
/// Looks up channel vocabulary indices automatically.
pub fn build_batch_named<B: Backend>(
    signal: Vec<f32>,         // [C, T] row-major
    channel_names: &[&str],   // e.g. ["Fz", "C3", "C4", ...]
    n_samples: usize,
    device: &B::Device,
) -> InputBatch<B> {
    let n_channels = channel_names.len();
    let indices = crate::channel_vocab::channel_indices_unwrap(channel_names);
    build_batch(signal, indices, n_channels, n_samples, device)
}

/// Channel-wise z-score normalisation.
pub fn channel_wise_normalize<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);  // [B, C, 1]
    let diff = x.clone() - mean.clone();
    let var = (diff.clone() * diff.clone()).mean_dim(2);
    let std = (var + 1e-8).sqrt();
    (x - mean) / std
}
