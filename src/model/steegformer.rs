/// ST-EEGFormer — full encoder model (burn 0.20.1)
///
/// Optimizations vs naive implementation:
///   - Pre-computed temporal PE cached on device (no CPU→GPU transfer per call)
///   - Channel embedding tiled via expand (no allocations)
///   - CLS token PE pre-added at load time
///   - Attention: scale fused into Q

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
    pub patch_embed: PatchEmbedEEG<B>,
    pub cls_token: Param<Tensor<B, 3>>,
    pub channel_embed: ChannelPositionalEmbed<B>,
    pub blocks: Vec<EncoderBlock<B>>,
    pub norm: Option<SteegLayerNorm<B>>,
    pub fc_norm: Option<SteegLayerNorm<B>>,
    pub head: Option<Linear<B>>,

    pub embed_dim:   usize,
    pub patch_size:  usize,
    pub global_pool: bool,
}

/// Encoder with pre-computed buffers for fast inference.
pub struct STEEGFormerWithPE<B: Backend> {
    pub model: STEEGFormer<B>,
    pub temporal_pe: TemporalPositionalEncoding<B>,
    /// Pre-computed CLS token with PE already added: [1, 1, D]
    cls_with_pe: Tensor<B, 3>,
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
                true, cfg.norm_eps, device,
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

        // Pre-compute CLS + PE
        let cls_pe = temporal_pe.get_cls_token();
        let cls_with_pe = cls_token.val() + cls_pe.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);

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

        STEEGFormerWithPE { model, temporal_pe, cls_with_pe }
    }

    /// Rebuild the cls_with_pe cache after weight loading.
    pub fn rebuild_cls_cache(steeg: &mut STEEGFormerWithPE<B>) {
        let cls_pe = steeg.temporal_pe.get_cls_token();
        steeg.cls_with_pe = steeg.model.cls_token.val()
            + cls_pe.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
    }
}

impl<B: Backend> STEEGFormerWithPE<B> {
    /// Forward pass: extract features.
    ///
    /// eeg: [B, C, T] — raw EEG signal
    /// chan_idx: [B, C] — channel embedding indices
    ///
    /// Returns: [B, embed_dim]
    pub fn forward_features(
        &self,
        eeg: Tensor<B, 3>,
        chan_idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [b, _c, _t] = eeg.dims();
        let dmodel = self.model.embed_dim;

        // 1) Patch embed: [B, C, T] → [B, Seq, Ch, D]
        let x = self.model.patch_embed.forward(eeg.clone());
        let [_, seq, ch_all, _] = x.dims();
        let seq_total = seq * ch_all;

        // Flatten to [B, Seq_total, D]
        let mut x = x.reshape([b, seq_total, dmodel]);

        // 2a) Channel embeddings — expand (no allocation, just strided view)
        let ch_emb_small = self.model.channel_embed.forward(chan_idx);
        let ch_emb = ch_emb_small
            .unsqueeze_dim::<4>(1)
            .expand([b, seq, ch_all, dmodel])
            .reshape([b, seq_total, dmodel]);

        // 2b) Temporal embeddings — slice from pre-computed PE buffer
        let tp_emb = self.temporal_pe.get_tiled(seq, ch_all, dmodel, &eeg.device());
        let tp_emb = tp_emb
            .unsqueeze_dim::<3>(0)
            .expand([b, seq_total, dmodel]);

        // 2c) Add positional encodings (single fused add)
        x = x + tp_emb + ch_emb;

        // 3) Prepend CLS token (pre-computed with PE)
        let cls_tokens = self.cls_with_pe.clone().expand([b, 1, dmodel]);
        x = Tensor::cat(vec![cls_tokens, x], 1);

        // 4) Transformer blocks
        for blk in &self.model.blocks {
            x = blk.forward(x);
        }

        // 5) Output
        if self.model.global_pool {
            let x = x.narrow(1, 1, seq_total);
            x.mean_dim(1).reshape([b, dmodel])
        } else {
            let x = self.model.norm.as_ref().unwrap().forward(x);
            x.narrow(1, 0, 1).reshape([b, dmodel])
        }
    }

    /// Forward pass with optional classification head.
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

    pub fn embed_dim(&self) -> usize {
        self.model.embed_dim
    }
}
