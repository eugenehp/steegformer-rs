/// ST-EEGFormer benchmark — measure inference latency on CPU and GPU.
///
/// Usage:
///   cargo run --example benchmark --release -- --weights tests/vectors/encoder_small.safetensors
///   cargo run --example benchmark --release -- --weights tests/vectors/encoder_small.safetensors --json
///   cargo run --example benchmark --release --no-default-features --features metal -- --weights ... --json

use std::time::Instant;

use burn::prelude::*;
use clap::Parser;
use steegformer_rs::{ModelConfig, data, channel_vocab};
use steegformer_rs::model::steegformer::STEEGFormer;
use steegformer_rs::weights::{WeightMap, load_model_from_wm};

// ── Backend dispatch ──────────────────────────────────────────────────────────
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::{Wgpu as B, wgpu::WgpuDevice};
    pub fn device() -> WgpuDevice { WgpuDevice::DefaultDevice }
    #[cfg(feature = "metal")]
    pub const NAME: &str = "GPU (wgpu — Metal / MSL)";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "GPU (wgpu — Vulkan / SPIR-V)";
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    pub const NAME: &str = "GPU (wgpu — WGSL)";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "CPU (NdArray + Apple Accelerate)";
    #[cfg(feature = "openblas-system")]
    pub const NAME: &str = "CPU (NdArray + OpenBLAS)";
    #[cfg(not(any(feature = "blas-accelerate", feature = "openblas-system")))]
    pub const NAME: &str = "CPU (NdArray + Rayon)";
}

use backend::B;

#[derive(Parser, Debug)]
#[command(about = "ST-EEGFormer — inference latency benchmark")]
struct Args {
    /// Comma-separated list of variants to benchmark.
    #[arg(long, default_value = "small")]
    variants: String,
    /// Path to safetensors weights (required for 'small' if present).
    #[arg(long)]
    weights: Option<String>,
    /// Number of warmup runs.
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Number of timed runs.
    #[arg(long, default_value_t = 10)]
    runs: usize,
    /// Output JSON.
    #[arg(long, default_value_t = false)]
    json: bool,
}

/// Generate deterministic synthetic EEG.
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

fn config_for_variant(variant: &str) -> ModelConfig {
    match variant {
        "small" => ModelConfig::small(),
        "base"  => ModelConfig::base(),
        "large" => ModelConfig::large(),
        _ => panic!("Unknown variant: {variant}"),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let json_mode = args.json;
    let device = backend::device();

    if !json_mode {
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  ST-EEGFormer — Inference Benchmark                         ║");
        eprintln!("╚══════════════════════════════════════════════════════════════╝\n");
        eprintln!("  Backend: {}", backend::NAME);
    }

    let variants: Vec<&str> = args.variants.split(',').map(|s| s.trim()).collect();

    // Standard test input: 22 BCI channels × 768 samples (6s @ 128Hz)
    let channel_names = channel_vocab::BCI_COMP_IV_2A;
    let n_channels = channel_names.len();  // 22
    let n_samples = 768;

    let signal = generate_synthetic_eeg(n_channels, n_samples, 128.0);
    let indices = channel_vocab::channel_indices_unwrap(channel_names);

    let mut json_results: Vec<serde_json::Value> = Vec::new();

    for variant in &variants {
        let cfg = config_for_variant(variant);

        if !json_mode {
            eprintln!("━━━ ST-EEGFormer-{} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", variant);
            eprintln!("  Config: D={}, depth={}, heads={}, patch={}",
                cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.patch_size);
        }

        // Load model
        let t = Instant::now();
        let steeg = if let Some(ref wp) = args.weights {
            let mut wm = WeightMap::from_file(wp)?;
            load_model_from_wm(&cfg, &mut wm, &device)?
        } else {
            STEEGFormer::new(&cfg, &device)
        };
        let ms_load = t.elapsed().as_secs_f64() * 1000.0;

        if !json_mode {
            eprintln!("  Load: {ms_load:.0} ms");
        }

        // Build batch
        let batch = data::build_batch::<B>(
            signal.clone(), indices.clone(), n_channels, n_samples, &device,
        );
        let signal_norm = data::channel_wise_normalize(batch.signal.clone());

        // Warmup
        if !json_mode {
            eprintln!("\n  ▸ Standard input: {}ch × {} samples", n_channels, n_samples);
        }
        for _ in 0..args.warmup {
            let _ = steeg.forward(signal_norm.clone(), batch.channel_indices.clone());
        }

        // Timed runs
        let mut times = Vec::with_capacity(args.runs);
        for _ in 0..args.runs {
            let t = Instant::now();
            let _ = steeg.forward(signal_norm.clone(), batch.channel_indices.clone());
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mean_ms = times.iter().sum::<f64>() / times.len() as f64;
        let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = times.iter().cloned().fold(0.0f64, f64::max);
        let std_ms = (times.iter().map(|t| (t - mean_ms).powi(2)).sum::<f64>() / times.len() as f64).sqrt();

        if !json_mode {
            eprintln!("    mean={mean_ms:.1}ms  min={min_ms:.1}ms  max={max_ms:.1}ms  std={std_ms:.1}ms  (n={})", args.runs);
        }

        // Channel scaling
        if !json_mode {
            eprintln!("\n  ▸ Channel scaling (fixed T={}):", n_samples);
            eprintln!("    {:>6}  {:>10}", "Chans", "Mean (ms)");
        }

        let channel_counts = [4, 8, 16, 22, 32, 64];
        let mut channel_scaling: Vec<serde_json::Value> = Vec::new();

        for &nc in &channel_counts {
            // Use first nc channels from the vocab
            if nc > channel_vocab::VOCAB_SIZE { continue; }
            let ch_names: Vec<&str> = channel_vocab::CHANNEL_VOCAB[..nc].to_vec();
            let ch_indices: Vec<i64> = (0..nc as i64).collect();
            let sig = generate_synthetic_eeg(nc, n_samples, 128.0);
            let b = data::build_batch::<B>(sig, ch_indices, nc, n_samples, &device);
            let sn = data::channel_wise_normalize(b.signal.clone());

            // Warmup
            let _ = steeg.forward(sn.clone(), b.channel_indices.clone());

            let mut t_vec = Vec::new();
            for _ in 0..3.max(args.runs / 2) {
                let t = Instant::now();
                let _ = steeg.forward(sn.clone(), b.channel_indices.clone());
                t_vec.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let avg = t_vec.iter().sum::<f64>() / t_vec.len() as f64;
            let cs_min = t_vec.iter().cloned().fold(f64::INFINITY, f64::min);
            let cs_max = t_vec.iter().cloned().fold(0.0f64, f64::max);

            if !json_mode {
                eprintln!("    {:>6}  {:>7.1} ms", nc, avg);
            }

            channel_scaling.push(serde_json::json!({
                "channels": nc,
                "mean_ms": (avg * 100.0).round() / 100.0,
                "min_ms": (cs_min * 100.0).round() / 100.0,
                "max_ms": (cs_max * 100.0).round() / 100.0,
                "runs": t_vec,
            }));
        }

        // Sequence length scaling
        if !json_mode {
            eprintln!("\n  ▸ Sequence scaling (fixed C={}):", n_channels);
            eprintln!("    {:>8}  {:>10}", "Samples", "Mean (ms)");
        }

        let sample_counts = [128, 256, 512, 768, 1024];
        let mut seq_scaling: Vec<serde_json::Value> = Vec::new();

        for &ns in &sample_counts {
            let sig = generate_synthetic_eeg(n_channels, ns, 128.0);
            let b = data::build_batch::<B>(sig, indices.clone(), n_channels, ns, &device);
            let sn = data::channel_wise_normalize(b.signal.clone());

            let _ = steeg.forward(sn.clone(), b.channel_indices.clone());

            let mut t_vec = Vec::new();
            for _ in 0..3.max(args.runs / 2) {
                let t = Instant::now();
                let _ = steeg.forward(sn.clone(), b.channel_indices.clone());
                t_vec.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let avg = t_vec.iter().sum::<f64>() / t_vec.len() as f64;

            if !json_mode {
                eprintln!("    {:>8}  {:>7.1} ms  ({:.1}s @ 128Hz)", ns, avg, ns as f64 / 128.0);
            }

            seq_scaling.push(serde_json::json!({
                "samples": ns,
                "duration_s": ns as f64 / 128.0,
                "mean_ms": (avg * 100.0).round() / 100.0,
                "runs": t_vec,
            }));
        }

        if !json_mode { eprintln!(); }

        json_results.push(serde_json::json!({
            "variant": variant,
            "backend": backend::NAME,
            "config": {
                "embed_dim": cfg.embed_dim,
                "depth": cfg.depth,
                "num_heads": cfg.num_heads,
                "patch_size": cfg.patch_size,
                "mlp_ratio": cfg.mlp_ratio,
            },
            "load_ms": (ms_load * 100.0).round() / 100.0,
            "inference": {
                "channels": n_channels,
                "samples": n_samples,
                "warmup": args.warmup,
                "runs": args.runs,
                "mean_ms": (mean_ms * 100.0).round() / 100.0,
                "min_ms": (min_ms * 100.0).round() / 100.0,
                "max_ms": (max_ms * 100.0).round() / 100.0,
                "std_ms": (std_ms * 100.0).round() / 100.0,
                "all_ms": times,
            },
            "channel_scaling": channel_scaling,
            "sequence_scaling": seq_scaling,
        }));
    }

    if json_mode {
        println!("{}", serde_json::to_string_pretty(&json_results)?);
    }

    Ok(())
}
