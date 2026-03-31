/// Example: Embed — produce latent EEG embeddings with ST-EEGFormer.
///
/// Demonstrates:
///   - Loading the model
///   - Building input batches from channel names
///   - Extracting embeddings
///   - Saving results
///
/// Usage:
///   cargo run --example embed --release -- --weights model.safetensors --variant small

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use steegformer::{STEEGFormerEncoder, ModelConfig, data, channel_vocab};

#[derive(Parser, Debug)]
#[command(about = "ST-EEGFormer — latent embedding extraction")]
struct Args {
    #[arg(long, default_value = "small")]
    variant: String,
    #[arg(long)]
    weights: String,
    #[arg(long, default_value = "data/embeddings.safetensors")]
    output: String,
    #[arg(long, short = 'v')]
    verbose: bool,
}

// ── Backend dispatch ──────────────────────────────────────────────────────────
#[cfg(feature = "wgpu")]
type B = burn::backend::Wgpu;
#[cfg(feature = "wgpu")]
fn get_device() -> burn::backend::wgpu::WgpuDevice { burn::backend::wgpu::WgpuDevice::default() }

#[cfg(not(feature = "wgpu"))]
type B = burn::backend::NdArray;
#[cfg(not(feature = "wgpu"))]
fn get_device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let device = get_device();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ST-EEGFormer — Latent Embedding Extraction                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 1. Build config for variant
    let cfg = match args.variant.as_str() {
        "small" => ModelConfig::small(),
        "base"  => ModelConfig::base(),
        "large" => ModelConfig::large(),
        _ => anyhow::bail!("Unknown variant: {}. Use small, base, or large.", args.variant),
    };

    // 2. Load model
    println!("▸ Loading ST-EEGFormer-{} …", args.variant);
    let (encoder, ms_load) = STEEGFormerEncoder::<B>::load_from_config(
        cfg,
        Path::new(&args.weights),
        device.clone(),
    )?;
    println!("  {}  ({ms_load:.0} ms)\n", encoder.describe());

    // 3. Generate synthetic EEG segments
    let channel_names = channel_vocab::BCI_COMP_IV_2A;
    let n_channels = channel_names.len();
    let n_samples = 768;  // 6s @ 128 Hz
    let n_segments = 3;

    println!("▸ Generating {} synthetic EEG segments ({} ch × {} samples each)\n",
        n_segments, n_channels, n_samples);

    let mut all_outputs: Vec<steegformer::SegmentEmbedding> = Vec::new();

    for seg_idx in 0..n_segments {
        let signal = generate_synthetic_eeg(n_channels, n_samples, 128.0 + seg_idx as f32);
        let batch = data::build_batch_named::<B>(signal, channel_names, n_samples, &device);

        let t = Instant::now();
        let result = encoder.run_batch(&batch)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let vals = &result.output;
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        let std: f64 = (vals.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / vals.len() as f64).sqrt();
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("  Segment {seg_idx}: shape={:?}  mean={mean:+.4}  std={std:.4}  [{min:+.3}, {max:+.3}]  {ms:.1}ms",
            result.shape);

        if args.verbose && seg_idx == 0 {
            println!("\n    First 5 output values: {:?}", &vals[..5.min(vals.len())]);
        }

        all_outputs.push(result);
    }

    // 4. Save
    let encoding = steegformer::EncodingResult {
        segments: all_outputs,
        ms_preproc: 0.0,
        ms_encode: t0.elapsed().as_secs_f64() * 1000.0,
    };

    if let Some(p) = Path::new(&args.output).parent() { std::fs::create_dir_all(p)?; }
    encoding.save_safetensors(&args.output)?;
    println!("\n▸ Saved {} segments → {}", n_segments, args.output);

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Total: {ms_total:.0} ms");
    Ok(())
}

/// Generate synthetic EEG: sum of sine waves + noise.
fn generate_synthetic_eeg(n_channels: usize, n_samples: usize, sfreq: f32) -> Vec<f32> {
    let mut signal = vec![0.0f32; n_channels * n_samples];
    for ch in 0..n_channels {
        let alpha_freq = 9.0 + (ch as f32 * 0.3);
        let beta_freq = 18.0 + (ch as f32 * 0.5);
        let theta_freq = 5.0 + (ch as f32 * 0.2);
        let mut noise_state: u32 = (ch as u32 + 1) * 0xDEAD_BEEF;
        for t in 0..n_samples {
            let time = t as f32 / sfreq;
            let alpha = (2.0 * std::f32::consts::PI * alpha_freq * time).sin() * 20e-6;
            let beta  = (2.0 * std::f32::consts::PI * beta_freq * time).sin() * 5e-6;
            let theta = (2.0 * std::f32::consts::PI * theta_freq * time).sin() * 15e-6;
            noise_state ^= noise_state << 13;
            noise_state ^= noise_state >> 17;
            noise_state ^= noise_state << 5;
            let noise = (noise_state as f32 / u32::MAX as f32 - 0.5) * 2e-6;
            signal[ch * n_samples + t] = alpha + beta + theta + noise;
        }
    }
    signal
}
