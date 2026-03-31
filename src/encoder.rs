//! Standalone ST-EEGFormer encoder — produce latent EEG embeddings.
//!
//! The encoder produces:
//! - For embeddings: [B, embed_dim] CLS token or global-pooled representation
//! - For classification: [B, num_classes] logits

use std::{path::Path, time::Instant};

use anyhow::Context;
use burn::prelude::*;

use crate::{
    config::{DataConfig, ModelConfig},
    data::{InputBatch, channel_wise_normalize},
    model::steegformer::STEEGFormerWithPE,
    weights::load_model,
};

/// Per-segment embedding produced by ST-EEGFormer.
pub struct SegmentEmbedding {
    /// Output values: row-major f32.
    /// - Embedding mode: [embed_dim]
    /// - Classification mode: [num_classes]
    pub output: Vec<f32>,
    /// Shape of the output.
    pub shape: Vec<usize>,
    pub n_channels: usize,
}

/// Collection of per-segment outputs.
pub struct EncodingResult {
    pub segments: Vec<SegmentEmbedding>,
    pub ms_preproc: f64,
    pub ms_encode: f64,
}

impl EncodingResult {
    /// Save to safetensors file.
    pub fn save_safetensors(&self, path: &str) -> anyhow::Result<()> {
        use safetensors::{Dtype, View};
        use std::borrow::Cow;

        struct RawTensor { data: Vec<u8>, shape: Vec<usize>, dtype: Dtype }
        impl View for RawTensor {
            fn dtype(&self)    -> Dtype         { self.dtype }
            fn shape(&self)    -> &[usize]      { &self.shape }
            fn data(&self)     -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize         { self.data.len() }
        }

        let f32_bytes = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| f.to_le_bytes()).collect()
        };

        let mut keys: Vec<String> = Vec::new();
        let mut tensors: Vec<RawTensor> = Vec::new();

        for (i, seg) in self.segments.iter().enumerate() {
            keys.push(format!("output_{i}"));
            tensors.push(RawTensor {
                data: f32_bytes(&seg.output),
                shape: seg.shape.clone(),
                dtype: Dtype::F32,
            });
        }

        let n = self.segments.len() as f32;
        keys.push("n_samples".into());
        tensors.push(RawTensor {
            data: f32_bytes(&[n]),
            shape: vec![1],
            dtype: Dtype::F32,
        });

        let pairs: Vec<(&str, RawTensor)> = keys.iter()
            .map(|s| s.as_str())
            .zip(tensors)
            .collect();
        let bytes = safetensors::serialize(pairs, None)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ── STEEGFormerEncoder ────────────────────────────────────────────────────────

/// High-level encoder for EEG signal processing.
pub struct STEEGFormerEncoder<B: Backend> {
    steeg:     STEEGFormerWithPE<B>,
    pub model_cfg: ModelConfig,
    pub data_cfg:  DataConfig,
    device:    B::Device,
}

impl<B: Backend> STEEGFormerEncoder<B> {
    /// Load model from config.json and weights safetensors.
    pub fn load(
        config_path:  &Path,
        weights_path: &Path,
        device:       B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("config: {}", config_path.display()))?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: ModelConfig = serde_json::from_value(
            hf_val.get("model").cloned().unwrap_or(hf_val.clone())
        ).context("parsing model config")?;

        let t = Instant::now();
        let steeg = load_model::<B>(
            &model_cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            &device,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((Self { steeg, model_cfg, data_cfg: DataConfig::default(), device }, ms))
    }

    /// Load from a ModelConfig directly (no config.json file needed).
    pub fn load_from_config(
        cfg: ModelConfig,
        weights_path: &Path,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let t = Instant::now();
        let steeg = load_model::<B>(
            &cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            &device,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((Self { steeg, model_cfg: cfg, data_cfg: DataConfig::default(), device }, ms))
    }

    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        format!(
            "ST-EEGFormer  embed_dim={}  depth={}  heads={}  patch={}  classes={}  pool={}",
            c.embed_dim, c.depth, c.num_heads, c.patch_size, c.num_classes,
            if c.global_pool { "global" } else { "cls" },
        )
    }

    /// Run inference on a prepared InputBatch.
    pub fn run_batch(&self, batch: &InputBatch<B>) -> anyhow::Result<SegmentEmbedding> {
        // Channel-wise z-score normalisation
        let signal = channel_wise_normalize(batch.signal.clone());

        let output = self.steeg.forward(signal, batch.channel_indices.clone());

        let shape = output.dims().to_vec();
        let output_vec = output
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("output→vec: {e:?}"))?;

        Ok(SegmentEmbedding {
            output: output_vec,
            shape: shape[1..].to_vec(),  // remove batch dim
            n_channels: batch.n_channels,
        })
    }

    /// Run on multiple batches.
    pub fn run_batches(&self, batches: &[InputBatch<B>]) -> anyhow::Result<Vec<SegmentEmbedding>> {
        batches.iter().map(|b| self.run_batch(b)).collect()
    }

    pub fn device(&self) -> &B::Device { &self.device }
}
