//! # steegformer — ST-EEGFormer EEG Foundation Model inference in Rust
//!
//! Pure-Rust inference for the ST-EEGFormer (Spatio-Temporal EEG Transformer)
//! EEG foundation model, built on [Burn 0.20](https://burn.dev).
//!
//! ST-EEGFormer is a ViT-based foundation model pre-trained via Masked
//! Autoencoder (MAE) reconstruction on raw EEG signals. It won 1st place
//! in the NeurIPS 2025 EEG Foundation Challenge and was accepted at ICLR 2026.
//!
//! Key properties:
//! - Designed for **128 Hz** EEG data
//! - Supports up to **142 channels** with learned channel embeddings
//! - Pre-trained on **6-second** segments (768 samples)
//! - Patch size: **16 samples** (0.125s temporal resolution)
//! - Sinusoidal temporal + learned channel positional encodings
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use steegformer::STEEGFormerEncoder;
//!
//! let (encoder, ms) = STEEGFormerEncoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//!
//! let batch = steegformer::build_batch_named::<B>(
//!     signal, &["Fz", "C3", "C4", "Pz"], n_samples, &device,
//! );
//! let result = encoder.run_batch(&batch)?;
//! ```

pub mod channel_vocab;
pub mod config;
pub mod data;
pub mod encoder;
pub mod model;
pub mod weights;

// Flat re-exports
pub use encoder::{STEEGFormerEncoder, SegmentEmbedding, EncodingResult};
pub use config::{ModelConfig, DataConfig};
pub use data::{InputBatch, build_batch, build_batch_named, channel_wise_normalize};
pub use channel_vocab::{
    CHANNEL_VOCAB, VOCAB_SIZE, EMBEDDING_TABLE_SIZE,
    channel_index, channel_indices, channel_indices_unwrap,
    BCI_COMP_IV_2A, STANDARD_10_20,
};
