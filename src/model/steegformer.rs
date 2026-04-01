/// ST-EEGFormer — full encoder model (burn 0.20.1)
///
/// Optimizations vs naive implementation:
///   - Pre-computed temporal PE cached on device (no CPU→GPU transfer per call)
///   - Channel embedding tiled via expand (no allocations)
///   - CLS token PE pre-added at load time
///   - Attention: scale fused into Q
///
/// With `wgpu-kernels`:
///   - LayerNorm: fused single-pass CubeCL kernel (17× calls, saves ~136 dispatches)
///   - Softmax:   fused CubeCL kernel (8× calls, saves ~32 dispatches)
///   - GELU:      fused CubeCL kernel (8× calls, saves ~16 dispatches)
///   Total: ~330 dispatches → ~145 dispatches (~56% reduction)

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};

use crate::model::patch_embed::PatchEmbedEEG;
use crate::model::positional::{TemporalPositionalEncoding, ChannelPositionalEmbed};
use crate::model::encoder_block::EncoderBlock;
use crate::model::norm::SteegLayerNorm;
use crate::config::ModelConfig;

// ── STEEGFormer Model ─────────────────────────────────────────────────────

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
    pub cls_with_pe: Tensor<B, 3>,
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

// ── Standard forward (no fused kernels) ───────────────────────────────────
#[cfg(not(feature = "wgpu-kernels"))]
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
        forward_features_impl(self, eeg, chan_idx)
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

// ── Fused forward (CubeCL kernels) ────────────────────────────────────────
#[cfg(feature = "wgpu-kernels")]
impl<B: Backend + super::FusedOps> STEEGFormerWithPE<B> {
    pub fn forward_features(
        &self,
        eeg: Tensor<B, 3>,
        chan_idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        forward_features_impl(self, eeg, chan_idx)
    }

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

// ── Shared implementation (body is identical for both paths) ──────────────

macro_rules! define_forward_features {
    ($($bound:tt)*) => {
        fn forward_features_impl<B: Backend $($bound)*>(
            this: &STEEGFormerWithPE<B>,
            eeg: Tensor<B, 3>,
            chan_idx: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2> {
            let [b, _c, _t] = eeg.dims();
            let dmodel = this.model.embed_dim;

            // 1) Patch embed: [B, C, T] → [B, Seq, Ch, D]
            let x = this.model.patch_embed.forward(eeg.clone());
            let [_, seq, ch_all, _] = x.dims();
            let seq_total = seq * ch_all;

            // 2) Add positional encodings using 4D broadcast (avoids expand+reshape copies)
            //    x:      [B, Seq, Ch, D]
            //    ch_emb: [B,  1,  Ch, D] — broadcasts over Seq
            //    tp_emb: [1, Seq,  1, D] — broadcasts over B and Ch
            //    Burn's fusion engine combines this into a single kernel dispatch.
            let ch_emb = this.model.channel_embed.forward(chan_idx)
                .unsqueeze_dim::<4>(1);  // [B, 1, Ch, D]
            let tp_emb = this.temporal_pe.pe.clone()
                .narrow(0, 0, seq)          // [Seq, D]
                .unsqueeze_dim::<3>(1)      // [Seq, 1, D]
                .unsqueeze_dim::<4>(0);     // [1, Seq, 1, D]
            let x = (x + ch_emb + tp_emb)
                .reshape([b, seq_total, dmodel]);

            // 3) Prepend CLS token (pre-computed with PE)
            let cls_tokens = this.cls_with_pe.clone().expand([b, 1, dmodel]);
            let mut x = Tensor::cat(vec![cls_tokens, x], 1);

            // 4) Transformer blocks
            for blk in &this.model.blocks {
                x = blk.forward(x);
            }

            // 5) Output
            if this.model.global_pool {
                let x = x.narrow(1, 1, seq_total);
                x.mean_dim(1).reshape([b, dmodel])
            } else {
                let x = this.model.norm.as_ref().unwrap().forward(x);
                x.narrow(1, 0, 1).reshape([b, dmodel])
            }
        }
    };
}

#[cfg(not(feature = "wgpu-kernels"))]
define_forward_features!();

/// Fused version that chains blocks: the last dispatch of block N fuses with
/// norm1 of block N+1, saving 1 layernorm dispatch per block boundary (7 saves).
#[cfg(feature = "wgpu-kernels")]
macro_rules! define_forward_features_fused {
    () => {
        fn forward_features_impl<B: Backend + super::FusedOps>(
            this: &STEEGFormerWithPE<B>,
            eeg: Tensor<B, 3>,
            chan_idx: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2> {
            let [b, _c, t] = eeg.dims();
            let dmodel = this.model.embed_dim;
            let patch_size = this.model.patch_size;
            let n_channels = chan_idx.dims()[1];
            let num_patches = t / patch_size;
            let seq_total = num_patches * n_channels;

            // 1-3) Patch embed + positional encoding + CLS prepend
            let x = this.model.patch_embed.forward(eeg.clone());
            let [_, seq, ch_all, _] = x.dims();
            let ch_emb = this.model.channel_embed.forward(chan_idx)
                .unsqueeze_dim::<4>(1);
            let tp_emb = this.temporal_pe.pe.clone()
                .narrow(0, 0, seq)
                .unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(0);
            let x = (x + ch_emb + tp_emb)
                .reshape([b, seq_total, dmodel]);
            let cls_tokens = this.cls_with_pe.clone().expand([b, 1, dmodel]);
            let mut x = Tensor::cat(vec![cls_tokens, x], 1);

            // 4) Transformer blocks — chained with cross-block norm fusion
            let n_blocks = this.model.blocks.len();
            if n_blocks > 0 {
                // Compute norm1 for first block
                let blk0 = &this.model.blocks[0];
                let ln_w = blk0.norm1.inner.gamma.val();
                let ln_b = blk0.norm1.inner.beta.as_ref().unwrap().val();
                let mut n1 = B::fused_layernorm(x.clone(), ln_w, ln_b, blk0.norm1.eps as f32);

                for i in 0..n_blocks - 1 {
                    let blk = &this.model.blocks[i];
                    let next_blk = &this.model.blocks[i + 1];
                    let next_ln_w = next_blk.norm1.inner.gamma.val();
                    let next_ln_b = next_blk.norm1.inner.beta.as_ref().unwrap().val();
                    let next_eps = next_blk.norm1.eps as f32;

                    // Forward with chained norm: fuses FC2 residual with next norm1
                    let (out, next_n1) = blk.forward_with_norm1_chain(
                        x, n1, next_ln_w, next_ln_b, next_eps,
                    );
                    x = out;
                    n1 = next_n1;
                }

                // Last block — no next block to chain with
                x = this.model.blocks[n_blocks - 1].forward_with_norm1(x, n1);
            }

            // 5) Output — use fused layernorm for final norm
            if this.model.global_pool {
                let x = x.narrow(1, 1, seq_total);
                x.mean_dim(1).reshape([b, dmodel])
            } else {
                let norm = this.model.norm.as_ref().unwrap();
                let ln_w = norm.inner.gamma.val();
                let ln_b = norm.inner.beta.as_ref().unwrap().val();
                let x = B::fused_layernorm(x, ln_w, ln_b, norm.eps as f32);
                x.narrow(1, 0, 1).reshape([b, dmodel])
            }
        }
    };
}

#[cfg(feature = "wgpu-kernels")]
define_forward_features_fused!();
