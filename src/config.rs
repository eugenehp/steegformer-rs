/// Model configuration for ST-EEGFormer inference.
///
/// Field names correspond to the Python model constructor arguments.
/// ST-EEGFormer is designed for **128 Hz** EEG, with **6-second** segments
/// (768 samples) and up to **142 channels**.

// ── ModelConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    /// Patch size in time-samples (default 16).
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Transformer embedding dimension.
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,

    /// Number of transformer encoder blocks.
    #[serde(default = "default_depth")]
    pub depth: usize,

    /// Number of attention heads per transformer block.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,

    /// MLP expansion ratio inside transformer blocks (default 4.0).
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,

    /// Maximum number of EEG channels supported.
    #[serde(default = "default_max_channels")]
    pub max_channels: usize,

    /// Number of output classes. 0 = encoder-only (embeddings).
    #[serde(default)]
    pub num_classes: usize,

    /// Whether to use global average pooling (true) or CLS token (false).
    #[serde(default)]
    pub global_pool: bool,

    /// Layer normalisation epsilon (default 1e-6).
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
}

fn default_patch_size()   -> usize { 16 }
fn default_embed_dim()    -> usize { 1024 }
fn default_depth()        -> usize { 24 }
fn default_num_heads()    -> usize { 16 }
fn default_mlp_ratio()    -> f64   { 4.0 }
fn default_max_channels() -> usize { 145 }
fn default_norm_eps()     -> f64   { 1e-6 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            patch_size:   default_patch_size(),
            embed_dim:    default_embed_dim(),
            depth:        default_depth(),
            num_heads:    default_num_heads(),
            mlp_ratio:    default_mlp_ratio(),
            max_channels: default_max_channels(),
            num_classes:  0,
            global_pool:  false,
            norm_eps:     default_norm_eps(),
        }
    }
}

impl ModelConfig {
    /// Head dimension: embed_dim / num_heads.
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    /// FFN hidden dimension: embed_dim * mlp_ratio.
    pub fn ffn_hidden_dim(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio) as usize
    }

    /// Small variant: 8 layers, 512-dim, 8 heads (~30M params).
    pub fn small() -> Self {
        Self {
            patch_size: 16, embed_dim: 512, depth: 8, num_heads: 8,
            mlp_ratio: 4.0, max_channels: 145, num_classes: 0,
            global_pool: false, norm_eps: 1e-6,
        }
    }

    /// Base variant: 12 layers, 768-dim, 12 heads (~86M params).
    pub fn base() -> Self {
        Self {
            patch_size: 16, embed_dim: 768, depth: 12, num_heads: 12,
            mlp_ratio: 4.0, max_channels: 145, num_classes: 0,
            global_pool: false, norm_eps: 1e-6,
        }
    }

    /// Large variant: 24 layers, 1024-dim, 16 heads (~300M+ params).
    pub fn large() -> Self {
        Self {
            patch_size: 16, embed_dim: 1024, depth: 24, num_heads: 16,
            mlp_ratio: 4.0, max_channels: 145, num_classes: 0,
            global_pool: false, norm_eps: 1e-6,
        }
    }
}

// ── DataConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Sampling rate (Hz). ST-EEGFormer is designed for 128 Hz.
    pub sample_rate: f32,
    /// Segment duration in seconds. Pre-trained for 6-second segments.
    pub segment_dur: f32,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            sample_rate: 128.0,
            segment_dur: 6.0,
        }
    }
}

impl DataConfig {
    /// Number of time samples per segment: sample_rate * segment_dur.
    pub fn segment_samples(&self) -> usize {
        (self.sample_rate * self.segment_dur) as usize
    }
}
