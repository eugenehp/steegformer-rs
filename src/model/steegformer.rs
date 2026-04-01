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

/// Pre-cached weight tensors for zero-overhead forward pass.
/// Avoids repeated `Param::val()` calls (Arc clone) and `unsqueeze` per forward.
pub struct WeightCache<B: Backend> {
    /// Per-block cached weights
    pub blocks: Vec<BlockWeightCache<B>>,
    /// Final norm weights (if CLS pool)
    pub final_norm_weight: Option<Tensor<B, 1>>,
    pub final_norm_bias: Option<Tensor<B, 1>>,
    pub final_norm_eps: f32,
}

pub struct BlockWeightCache<B: Backend> {
    pub qkv_w: Tensor<B, 3>,      // [1, D, 3D] — pre-unsqueezed for 3D matmul
    pub qkv_bias: Tensor<B, 1>,   // [3D]
    pub proj_w: Tensor<B, 3>,     // [1, D, D]
    pub proj_bias: Tensor<B, 1>,  // [D]
    pub fc1_w: Tensor<B, 3>,      // [1, D, ff]
    pub fc1_bias: Tensor<B, 1>,   // [ff]
    pub fc2_w: Tensor<B, 3>,      // [1, ff, D]
    pub fc2_bias: Tensor<B, 1>,   // [D]
    pub norm1_weight: Tensor<B, 1>,
    pub norm1_bias: Tensor<B, 1>,
    pub norm1_eps: f32,
    pub norm2_weight: Tensor<B, 1>,
    pub norm2_bias: Tensor<B, 1>,
    pub norm2_eps: f32,
    pub n_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

/// Encoder with pre-computed buffers for fast inference.
pub struct STEEGFormerWithPE<B: Backend> {
    pub model: STEEGFormer<B>,
    pub temporal_pe: TemporalPositionalEncoding<B>,
    /// Pre-computed CLS token with PE already added: [1, 1, D]
    pub cls_with_pe: Tensor<B, 3>,
    /// Cached weight tensors (built lazily on first forward or after weight loading)
    pub weight_cache: Option<WeightCache<B>>,
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

        let mut steeg = STEEGFormerWithPE { model, temporal_pe, cls_with_pe, weight_cache: None };
        Self::build_weight_cache(&mut steeg);
        steeg
    }

    /// Rebuild the cls_with_pe cache after weight loading.
    pub fn rebuild_cls_cache(steeg: &mut STEEGFormerWithPE<B>) {
        let cls_pe = steeg.temporal_pe.get_cls_token();
        steeg.cls_with_pe = steeg.model.cls_token.val()
            + cls_pe.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
    }

    /// Build the weight cache for zero-overhead forward calls.
    pub fn build_weight_cache(steeg: &mut STEEGFormerWithPE<B>) {
        let blocks: Vec<BlockWeightCache<B>> = steeg.model.blocks.iter().map(|blk| {
            BlockWeightCache {
                qkv_w: blk.attn.qkv.weight.val().unsqueeze_dim::<3>(0),
                qkv_bias: blk.attn.qkv.bias.as_ref().unwrap().val(),
                proj_w: blk.attn.proj.weight.val().unsqueeze_dim::<3>(0),
                proj_bias: blk.attn.proj.bias.as_ref().unwrap().val(),
                fc1_w: blk.mlp.fc1.weight.val().unsqueeze_dim::<3>(0),
                fc1_bias: blk.mlp.fc1.bias.as_ref().unwrap().val(),
                fc2_w: blk.mlp.fc2.weight.val().unsqueeze_dim::<3>(0),
                fc2_bias: blk.mlp.fc2.bias.as_ref().unwrap().val(),
                norm1_weight: blk.norm1.inner.gamma.val(),
                norm1_bias: blk.norm1.inner.beta.as_ref().unwrap().val(),
                norm1_eps: blk.norm1.eps as f32,
                norm2_weight: blk.norm2.inner.gamma.val(),
                norm2_bias: blk.norm2.inner.beta.as_ref().unwrap().val(),
                norm2_eps: blk.norm2.eps as f32,
                n_heads: blk.attn.n_heads,
                head_dim: blk.attn.head_dim,
                scale: blk.attn.scale,
            }
        }).collect();

        let (final_norm_weight, final_norm_bias, final_norm_eps) =
            if let Some(ref norm) = steeg.model.norm {
                (Some(norm.inner.gamma.val()), Some(norm.inner.beta.as_ref().unwrap().val()), norm.eps as f32)
            } else if let Some(ref norm) = steeg.model.fc_norm {
                (Some(norm.inner.gamma.val()), Some(norm.inner.beta.as_ref().unwrap().val()), norm.eps as f32)
            } else {
                (None, None, 1e-6)
            };

        steeg.weight_cache = Some(WeightCache {
            blocks,
            final_norm_weight,
            final_norm_bias,
            final_norm_eps,
        });
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

/// Fused version with weight caching and cross-block norm fusion.
/// Uses pre-cached weight tensors to avoid per-call Param::val() + unsqueeze overhead.
#[cfg(feature = "wgpu-kernels")]
fn forward_features_impl<B: Backend + super::FusedOps>(
    this: &STEEGFormerWithPE<B>,
    eeg: Tensor<B, 3>,
    chan_idx: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [b, _c, _t] = eeg.dims();
    let dmodel = this.model.embed_dim;

    // 1-3) Patch embed + positional encoding + CLS prepend
    let x = this.model.patch_embed.forward(eeg.clone());
    let [_, seq, ch_all, _] = x.dims();
    let seq_total = seq * ch_all;
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

    // 4) Transformer blocks — use weight cache if available
    if let Some(ref wc) = this.weight_cache {
        // === Cached path: no Param::val() calls, pre-unsqueezed weights ===
        let n_blocks = wc.blocks.len();
        if n_blocks > 0 {
            let blk0 = &wc.blocks[0];
            let mut n1 = B::fused_layernorm(x.clone(), blk0.norm1_weight.clone(), blk0.norm1_bias.clone(), blk0.norm1_eps);

            for i in 0..n_blocks - 1 {
                let bw = &wc.blocks[i];
                let next_bw = &wc.blocks[i + 1];
                let (out, next_n1) = forward_block_cached::<B>(
                    x, n1, bw,
                    Some((&next_bw.norm1_weight, &next_bw.norm1_bias, next_bw.norm1_eps)),
                );
                x = out;
                n1 = next_n1.unwrap();
            }

            let (out, _) = forward_block_cached::<B>(
                x, n1, &wc.blocks[n_blocks - 1], None,
            );
            x = out;
        }

        // 5) Output
        if this.model.global_pool {
            let x = x.narrow(1, 1, seq_total);
            x.mean_dim(1).reshape([b, dmodel])
        } else {
            let x = B::fused_layernorm(x,
                wc.final_norm_weight.clone().unwrap(),
                wc.final_norm_bias.clone().unwrap(),
                wc.final_norm_eps,
            );
            x.narrow(1, 0, 1).reshape([b, dmodel])
        }
    } else {
        // === Fallback: extract weights on each call (no cache built yet) ===
        let n_blocks = this.model.blocks.len();
        if n_blocks > 0 {
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
                let (out, next_n1) = blk.forward_with_norm1_chain(
                    x, n1, next_ln_w, next_ln_b, next_eps,
                );
                x = out;
                n1 = next_n1;
            }
            x = this.model.blocks[n_blocks - 1].forward_with_norm1(x, n1);
        }

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
}

/// Forward a single transformer block using cached weights.
/// Returns (block_output, optional_next_norm1).
#[cfg(feature = "wgpu-kernels")]
pub fn forward_block_cached<B: Backend + super::FusedOps>(
    x: Tensor<B, 3>,
    n1: Tensor<B, 3>,
    bw: &BlockWeightCache<B>,
    next_ln: Option<(&Tensor<B, 1>, &Tensor<B, 1>, f32)>,
) -> (Tensor<B, 3>, Option<Tensor<B, 3>>) {
    // Attention sublayer (using cached pre-unsqueezed weights)
    let qkv = n1.matmul(bw.qkv_w.clone());  // [B, S, 3D]
    let (q, k_t, v) = B::fused_split_qkv_scaled(qkv, bw.qkv_bias.clone(), bw.n_heads, bw.head_dim, bw.scale);
    let out = B::fused_flash_attention(q, k_t, v);
    let out = B::fused_merge_heads(out, bw.n_heads, bw.head_dim);
    let attn_matmul = out.matmul(bw.proj_w.clone());

    // Bias + residual + LayerNorm(norm2)
    let (h_sum, n2) = B::fused_bias_residual_add_layernorm(
        x, attn_matmul, bw.proj_bias.clone(),
        bw.norm2_weight.clone(), bw.norm2_bias.clone(), bw.norm2_eps,
    );

    // FFN sublayer (using cached pre-unsqueezed weights)
    let h = n2.matmul(bw.fc1_w.clone());
    let h = B::fused_bias_gelu(h, bw.fc1_bias.clone());
    let mlp_matmul = h.matmul(bw.fc2_w.clone());

    // Final residual + optional next-block norm1 fusion
    if let Some((next_w, next_b, next_eps)) = next_ln {
        let (out, next_n1) = B::fused_bias_residual_add_layernorm(
            h_sum, mlp_matmul, bw.fc2_bias.clone(),
            next_w.clone(), next_b.clone(), next_eps,
        );
        (out, Some(next_n1))
    } else {
        let out = B::fused_bias_residual_add(h_sum, mlp_matmul, bw.fc2_bias.clone());
        (out, None)
    }
}
