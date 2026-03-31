/// Load pretrained ST-EEGFormer weights from a safetensors file.
///
/// ST-EEGFormer checkpoints store PyTorch state_dict keys.
/// The MAE pre-trained encoder weights have the following key patterns:
///
///   patch_embed.proj.weight [embed_dim, patch_size]
///   patch_embed.proj.bias   [embed_dim]
///   cls_token               [1, 1, embed_dim]
///   enc_channel_emd.channel_transformation.weight [145, embed_dim]
///   blocks.{i}.norm1.weight [D]
///   blocks.{i}.norm1.bias   [D]
///   blocks.{i}.attn.qkv.weight  [3*D, D]
///   blocks.{i}.attn.qkv.bias    [3*D]
///   blocks.{i}.attn.proj.weight [D, D]
///   blocks.{i}.attn.proj.bias   [D]
///   blocks.{i}.norm2.weight [D]
///   blocks.{i}.norm2.bias   [D]
///   blocks.{i}.mlp.fc1.weight [ff_dim, D]
///   blocks.{i}.mlp.fc1.bias   [ff_dim]
///   blocks.{i}.mlp.fc2.weight [D, ff_dim]
///   blocks.{i}.mlp.fc2.bias   [D]
///   norm.weight [D]
///   norm.bias   [D]

use std::collections::HashMap;
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;

use crate::model::steegformer::STEEGFormerWithPE;
use crate::config::ModelConfig;

// ── Raw tensor map ────────────────────────────────────────────────────────────

pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    /// Load all tensors from a safetensors file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let n_tensors = st.len();
        let mut tensors = HashMap::with_capacity(n_tensors);

        for (raw_key, view) in st.tensors() {
            // Strip common prefixes from PyTorch checkpoints
            let key = raw_key
                .strip_prefix("model.")
                .or_else(|| raw_key.strip_prefix("module."))
                .unwrap_or(raw_key.as_str())
                .to_string();

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by key, removing it from the map.
    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;

        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }

        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data, shape),
            device,
        ))
    }

    /// Check if a key exists.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Print all keys and shapes.
    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:80}  {s:?}");
        }
    }
}

// ── Weight assignment helpers ─────────────────────────────────────────────────

/// PyTorch [out, in] → burn [in, out] (transpose for Linear weight).
fn set_linear_wb<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>, b: Tensor<B, 1>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = linear.bias {
        linear.bias = Some(bias.clone().map(|_| b));
    }
}

fn set_layernorm<B: Backend>(norm: &mut crate::model::norm::SteegLayerNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    norm.inner.gamma = norm.inner.gamma.clone().map(|_| w);
    if let Some(ref beta) = norm.inner.beta {
        norm.inner.beta = Some(beta.clone().map(|_| b));
    }
}

// ── Full model loader ─────────────────────────────────────────────────────────

/// Load an ST-EEGFormer encoder from a safetensors file.
pub fn load_model<B: Backend>(
    cfg: &ModelConfig,
    weights_path: &str,
    device: &B::Device,
) -> anyhow::Result<STEEGFormerWithPE<B>> {
    let mut wm = WeightMap::from_file(weights_path)?;
    eprintln!("Loading {} weight tensors...", wm.tensors.len());
    load_model_from_wm(cfg, &mut wm, device)
}

/// Load from a pre-loaded WeightMap.
pub fn load_model_from_wm<B: Backend>(
    cfg: &ModelConfig,
    wm: &mut WeightMap,
    device: &B::Device,
) -> anyhow::Result<STEEGFormerWithPE<B>> {
    let mut steeg = crate::model::steegformer::STEEGFormer::new(cfg, device);
    load_encoder_weights(wm, &mut steeg, device)?;
    Ok(steeg)
}

fn load_encoder_weights<B: Backend>(
    wm: &mut WeightMap,
    steeg: &mut STEEGFormerWithPE<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let model = &mut steeg.model;
    let mut cls_loaded = false;

    // ── Patch embedding ─────────────────────────────────────────────────────
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("patch_embed.proj.weight", device),
        wm.take::<B, 1>("patch_embed.proj.bias", device),
    ) {
        set_linear_wb(&mut model.patch_embed.proj, w, b);
    }

    // ── CLS token ───────────────────────────────────────────────────────────
    if let Ok(t) = wm.take::<B, 3>("cls_token", device) {
        model.cls_token = model.cls_token.clone().map(|_| t);
        cls_loaded = true;
    }

    // ── Channel embedding ───────────────────────────────────────────────────
    // Python key: enc_channel_emd.channel_transformation.weight [145, D]
    if let Ok(w) = wm.take::<B, 2>("enc_channel_emd.channel_transformation.weight", device) {
        model.channel_embed.embedding.weight =
            model.channel_embed.embedding.weight.clone().map(|_| w);
    }

    // ── Transformer blocks ──────────────────────────────────────────────────
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let p = format!("blocks.{i}");

        // norm1
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.norm1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.norm1.bias"), device),
        ) { set_layernorm(&mut block.norm1, w, b); }

        // attn.qkv
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.attn.qkv.weight"), device),
            wm.take::<B, 1>(&format!("{p}.attn.qkv.bias"), device),
        ) { set_linear_wb(&mut block.attn.qkv, w, b); }

        // attn.proj
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.attn.proj.weight"), device),
            wm.take::<B, 1>(&format!("{p}.attn.proj.bias"), device),
        ) { set_linear_wb(&mut block.attn.proj, w, b); }

        // norm2
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.norm2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.norm2.bias"), device),
        ) { set_layernorm(&mut block.norm2, w, b); }

        // mlp.fc1
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.fc1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.fc1.bias"), device),
        ) { set_linear_wb(&mut block.mlp.fc1, w, b); }

        // mlp.fc2
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.fc2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.fc2.bias"), device),
        ) { set_linear_wb(&mut block.mlp.fc2, w, b); }
    }

    // ── Final norm ──────────────────────────────────────────────────────────
    if let Some(ref mut norm) = model.norm {
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>("norm.weight", device),
            wm.take::<B, 1>("norm.bias", device),
        ) { set_layernorm(norm, w, b); }
    }

    // ── FC norm (global pool) ───────────────────────────────────────────────
    if let Some(ref mut fc_norm) = model.fc_norm {
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>("fc_norm.weight", device),
            wm.take::<B, 1>("fc_norm.bias", device),
        ) { set_layernorm(fc_norm, w, b); }
    }

    // ── Classification head ─────────────────────────────────────────────────
    if let Some(ref mut head) = model.head {
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>("head.weight", device),
            wm.take::<B, 1>("head.bias", device),
        ) { set_linear_wb(head, w, b); }
    }

    // Rebuild CLS+PE cache if CLS token was loaded
    if cls_loaded {
        crate::model::steegformer::STEEGFormer::rebuild_cls_cache(steeg);
    }

    Ok(())
}
