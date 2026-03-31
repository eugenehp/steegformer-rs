/// Basic forward-pass smoke test: ensure the model runs without panicking.

use burn::backend::NdArray as B;
use burn::prelude::*;
use steegformer::{ModelConfig, data, channel_vocab};
use steegformer::model::steegformer::STEEGFormer;

#[test]
fn test_forward_small() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let cfg = ModelConfig::small();
    let steeg = STEEGFormer::<B>::new(&cfg, &device);

    let n_channels = 4;
    let n_samples = 128;  // 1 second @ 128 Hz
    let signal = vec![0.01f32; n_channels * n_samples];
    let channel_names: &[&str] = &["Fz", "C3", "C4", "Pz"];
    let batch = data::build_batch_named::<B>(signal, channel_names, n_samples, &device);

    let output = steeg.forward(batch.signal, batch.channel_indices);
    let shape = output.dims();
    assert_eq!(shape[0], 1, "batch dim should be 1");
    assert_eq!(shape[1], cfg.embed_dim, "output dim should be embed_dim");
}

#[test]
fn test_channel_vocab_lookup() {
    // Ensure all BCI IV-2a channels are in the vocab
    for ch in channel_vocab::BCI_COMP_IV_2A {
        assert!(channel_vocab::channel_index(ch).is_some(),
            "Channel {} not in vocab", ch);
    }
}

#[test]
fn test_patch_embed_shape() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let pe = steegformer::model::patch_embed::PatchEmbedEEG::<B>::new(16, 512, &device);

    let x = Tensor::<B, 3>::ones([1, 4, 768], &device);
    let y = pe.forward(x);
    let [b, seq, ch, d] = y.dims();
    assert_eq!(b, 1);
    assert_eq!(seq, 768 / 16);  // 48 time patches
    assert_eq!(ch, 4);
    assert_eq!(d, 512);
}

#[test]
fn test_temporal_pe_shape() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let tpe = steegformer::model::positional::TemporalPositionalEncoding::<B>::new(512, 100, &device);

    let indices = Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![1i64, 2, 3, 4, 5], vec![5]), &device,
    ).unsqueeze_dim::<2>(0);  // [1, 5]

    let emb = tpe.forward(indices);
    assert_eq!(emb.dims(), [1, 5, 512]);
}

#[test]
fn test_different_channel_counts() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let cfg = ModelConfig::small();
    let steeg = STEEGFormer::<B>::new(&cfg, &device);

    // Test with 2 channels
    for n_ch in [2, 8, 19] {
        let n_samples = 256;
        let signal = vec![0.01f32; n_ch * n_samples];
        let channels: Vec<&str> = channel_vocab::STANDARD_10_20[..n_ch].to_vec();
        let batch = data::build_batch_named::<B>(signal, &channels, n_samples, &device);

        let output = steeg.forward(batch.signal, batch.channel_indices);
        assert_eq!(output.dims()[0], 1);
        assert_eq!(output.dims()[1], cfg.embed_dim);
    }
}
