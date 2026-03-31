/// Positional Encodings for ST-EEGFormer.
///
/// ST-EEGFormer uses two additive positional encodings:
///
/// 1. **Temporal Positional Encoding** — fixed sinusoidal (like original Transformer).
///    Each time-patch index gets a sinusoidal embedding. This is a buffer (not learned).
///
/// 2. **Channel Positional Embedding** — learned `nn.Embedding(145, embed_dim)`.
///    Each EEG channel index maps to a learned embedding vector.
///    Initialized to zeros so the model starts without channel bias.

use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig};

/// Fixed sinusoidal temporal positional encoding.
///
/// Python: `TemporalPositionalEncoding` in models_mae_eeg.py.
///
/// pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
/// pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
///
/// Stored as a [max_len, d_model] buffer.
#[derive(Debug)]
pub struct TemporalPositionalEncoding<B: Backend> {
    /// Pre-computed sinusoidal embeddings: [max_len, d_model].
    pub pe: Tensor<B, 2>,
    pub d_model: usize,
    pub max_len: usize,
}

impl<B: Backend> TemporalPositionalEncoding<B> {
    pub fn new(d_model: usize, max_len: usize, device: &B::Device) -> Self {
        let mut pe_data = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
                pe_data[pos * d_model + 2 * i] = angle.sin() as f32;
                pe_data[pos * d_model + 2 * i + 1] = angle.cos() as f32;
            }
        }

        let pe = Tensor::<B, 2>::from_data(
            TensorData::new(pe_data, vec![max_len, d_model]),
            device,
        );

        Self { pe, d_model, max_len }
    }

    /// Get temporal embeddings for the given sequence indices.
    ///
    /// seq_indices: [B, S] (integer indices into the PE table)
    /// Returns: [B, S, d_model]
    pub fn forward(&self, seq_indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, s] = seq_indices.dims();

        // Flatten indices, gather from PE table, reshape back
        let flat_indices = seq_indices.reshape([b * s]);

        // Manual gather: index into pe[flat_indices, :]
        // pe is [max_len, d_model], flat_indices is [b*s]
        let gathered = self.pe.clone()
            .select(0, flat_indices);  // [b*s, d_model]

        gathered.reshape([b, s, self.d_model])
    }

    /// Get the CLS token positional encoding (index 0).
    pub fn get_cls_token(&self) -> Tensor<B, 1> {
        self.pe.clone().narrow(0, 0, 1).reshape([self.d_model])
    }

    /// Get pre-tiled temporal embeddings for a given seq/channel layout.
    ///
    /// Returns: [seq_total, d_model] where seq_total = seq * ch_all.
    /// Each time position is repeated ch_all times.
    /// Avoids creating Int index tensors on every forward call.
    pub fn get_tiled(
        &self,
        seq: usize,
        ch_all: usize,
        d_model: usize,
        _device: &B::Device,
    ) -> Tensor<B, 2> {
        // Slice [0..seq] from PE buffer
        let temp_emb = self.pe.clone().narrow(0, 0, seq);  // [seq, D]

        // Tile across channels: [seq, D] → [seq, ch_all, D] → [seq*ch_all, D]
        temp_emb
            .unsqueeze_dim::<3>(1)
            .expand([seq, ch_all, d_model])
            .reshape([seq * ch_all, d_model])
    }
}

/// Learned channel positional embedding.
///
/// Python: `ChannelPositionalEmbed` in models_mae_eeg.py.
///   nn.Embedding(145, embed_dim), initialized to zeros.
#[derive(Module, Debug)]
pub struct ChannelPositionalEmbed<B: Backend> {
    pub embedding: Embedding<B>,
}

impl<B: Backend> ChannelPositionalEmbed<B> {
    pub fn new(max_channels: usize, embed_dim: usize, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(max_channels, embed_dim).init(device);
        // Note: Python initializes to zeros. The weight loading step will
        // overwrite with pretrained values. For fresh init, use set_zeros().
        Self { embedding }
    }

    /// Get channel embeddings for the given channel indices.
    ///
    /// channel_indices: [B, C] (integer indices)
    /// Returns: [B, C, embed_dim]
    pub fn forward(&self, channel_indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(channel_indices)
    }
}
