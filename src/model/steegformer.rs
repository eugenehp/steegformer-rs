/// ST-EEGFormer — full encoder model (burn 0.20.1)
///
/// Python: `VisionTransformer` class in models_vit_eeg.py (for fine-tuning/inference)
///         `MaskedAutoencoderViT` class in models_mae_eeg.py (for pre-training)
///
/// Architecture (encoder only, for inference):
///   1. PatchEmbedEEG — split each channel into temporal patches, linear project
///   2. Additive positional encodings:
///      - Sinusoidal temporal encoding (which time patch)
///      - Learned channel embedding (which EEG electrode)
///   3. Prepend [CLS] token
///   4. N × Transformer encoder blocks (pre-norm, multi-head self-attention + FFN)
///   5. Output: [CLS] token embedding (or global average pool)

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};

use crate::model::patch_embed::PatchEmbedEEG;
use crate::model::positional::{TemporalPositionalEncoding, ChannelPositionalEmbed};
use crate::model::encoder_block::EncoderBlock;
use crate::model::norm::SteegLayerNorm;
use crate::config::ModelConfig;

// ── STEEGFormer Model ─────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct STEEGFormer<B: Backend> {
    /// Patch embedding: per-channel temporal patching + linear projection.
    pub patch_embed: PatchEmbedEEG<B>,
    /// Learned [CLS] token: [1, 1, embed_dim].
    pub cls_token: Param<Tensor<B, 3>>,
    /// Learned channel positional embeddings.
    pub channel_embed: ChannelPositionalEmbed<B>,
    /// Transformer encoder blocks.
    pub blocks: Vec<EncoderBlock<B>>,
    /// Final layer norm (used when global_pool=false).
    pub norm: Option<SteegLayerNorm<B>>,
    /// Global pool norm (used when global_pool=true).
    pub fc_norm: Option<SteegLayerNorm<B>>,
    /// Classification head (optional).
    pub head: Option<Linear<B>>,

    // Config
    pub embed_dim:   usize,
    pub patch_size:  usize,
    pub global_pool: bool,
}

/// Fixed sinusoidal temporal encoding is NOT a Burn Module (it's a buffer),
/// so we store it outside the module system.
pub struct STEEGFormerWithPE<B: Backend> {
    pub model: STEEGFormer<B>,
    pub temporal_pe: TemporalPositionalEncoding<B>,
}

impl<B: Backend> STEEGFormer<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> STEEGFormerWithPE<B> {
        let patch_embed = PatchEmbedEEG::new(cfg.patch_size, cfg.embed_dim, device);
        let channel_embed = ChannelPositionalEmbed::new(cfg.max_channels, cfg.embed_dim, device);
        let temporal_pe = TemporalPositionalEncoding::new(cfg.embed_dim, 512, device);

        let cls_token = Param::initialized(
            ParamId::new(),
            Tensor::zeros([1, 1, cfg.embed_dim], device),
        );

        let blocks = (0..cfg.depth)
            .map(|_| EncoderBlock::new(
                cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
                true, // qkv_bias
                cfg.norm_eps, device,
            ))
            .collect();

        let (norm, fc_norm) = if cfg.global_pool {
            (None, Some(SteegLayerNorm::new(cfg.embed_dim, cfg.norm_eps, device)))
        } else {
            (Some(SteegLayerNorm::new(cfg.embed_dim, cfg.norm_eps, device)), None)
        };

        let head = if cfg.num_classes > 0 {
            Some(LinearConfig::new(cfg.embed_dim, cfg.num_classes).with_bias(true).init(device))
        } else {
            None
        };

        let model = STEEGFormer {
            patch_embed,
            cls_token,
            channel_embed,
            blocks,
            norm,
            fc_norm,
            head,
            embed_dim: cfg.embed_dim,
            patch_size: cfg.patch_size,
            global_pool: cfg.global_pool,
        };

        STEEGFormerWithPE { model, temporal_pe }
    }
}

impl<B: Backend> STEEGFormerWithPE<B> {
    /// Forward pass: extract features (no classification head).
    ///
    /// eeg: [B, C, T] — raw EEG signal
    /// chan_idx: [B, C] — channel embedding indices
    ///
    /// Returns: [B, embed_dim] — CLS token or global-pooled representation
    pub fn forward_features(
        &self,
        eeg: Tensor<B, 3>,
        chan_idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [b, _c, _t] = eeg.dims();

        // 1) Patch embed: [B, C, T] → [B, Seq, Ch_all, embed_dim]
        let x = self.model.patch_embed.forward(eeg.clone());
        let [_, seq, ch_all, dmodel] = x.dims();
        let seq_total = seq * ch_all;

        // Flatten to [B, Seq_total, embed_dim]
        let mut x = x.reshape([b, seq_total, dmodel]);

        // 2a) Channel embeddings: [B, Ch_all] → [B, Ch_all, D]
        let ch_emb_small = self.model.channel_embed.forward(chan_idx);  // [B, Ch_all, D]

        // Tile across Seq: [B, Ch_all, D] → [B, Seq, Ch_all, D] → [B, Seq_total, D]
        let ch_emb = ch_emb_small
            .unsqueeze_dim::<4>(1)                    // [B, 1, Ch_all, D]
            .expand([b, seq, ch_all, dmodel])         // [B, Seq, Ch_all, D]
            .reshape([b, seq_total, dmodel]);

        // 2b) Temporal embeddings
        // Build seq indices: [1, Seq] with values 0..Seq-1
        // Python forward_encoder (optimized) uses torch.arange(Seq) → 0-indexed
        let device = eeg.device();
        let seq_idx_data: Vec<i64> = (0..seq as i64).collect();
        let temp_idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(seq_idx_data, vec![seq]),
            &device,
        ).unsqueeze_dim::<2>(0);  // [1, Seq]

        let temp_emb_small = self.temporal_pe.forward(temp_idx);  // [1, Seq, D]
        let temp_emb_small = temp_emb_small.squeeze::<2>();     // [Seq, D]

        // Tile across Ch_all: [Seq, D] → [Seq, Ch_all, D] → [Seq_total, D]
        let temp_emb_flat = temp_emb_small
            .unsqueeze_dim::<3>(1)                    // [Seq, 1, D]
            .expand([seq, ch_all, dmodel])            // [Seq, Ch_all, D]
            .reshape([seq_total, dmodel]);

        // Broadcast to batch: [1, Seq_total, D] → [B, Seq_total, D]
        let tp_emb = temp_emb_flat
            .unsqueeze_dim::<3>(0)                    // [1, Seq_total, D]
            .expand([b, seq_total, dmodel]);

        // 2c) Add positional encodings
        x = x + tp_emb + ch_emb;

        // 3) Prepend [CLS] token
        let cls_pe = self.temporal_pe.get_cls_token();  // [D]
        let cls_token = self.model.cls_token.val() + cls_pe.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        let cls_tokens = cls_token.expand([b, 1, dmodel]);
        x = Tensor::cat(vec![cls_tokens, x], 1);  // [B, 1 + Seq_total, D]

        // 4) Transformer blocks
        for blk in &self.model.blocks {
            x = blk.forward(x);
        }

        // 5) Output extraction
        if self.model.global_pool {
            // Global average pool (skip CLS token)
            let x = x.narrow(1, 1, seq_total);  // [B, Seq_total, D]
            x.mean_dim(1).reshape([b, dmodel])   // [B, D]
        } else {
            // Use CLS token
            let x = self.model.norm.as_ref().unwrap().forward(x);
            x.narrow(1, 0, 1).reshape([b, dmodel])  // [B, D]
        }
    }

    /// Forward pass with classification head.
    ///
    /// eeg: [B, C, T] — raw EEG signal
    /// chan_idx: [B, C] — channel embedding indices
    ///
    /// Returns: [B, num_classes] or [B, embed_dim] if no head
    pub fn forward(
        &self,
        eeg: Tensor<B, 3>,
        chan_idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let features = self.forward_features(eeg, chan_idx);
        if let Some(ref head) = self.model.head {
            head.forward(features)
        } else {
            features
        }
    }

    /// Get embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.model.embed_dim
    }
}
