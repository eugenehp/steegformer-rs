/// ST-EEGFormer benchmark — fused CPU forward pass vs standard Burn path.
///
/// Usage:
///   cargo run --example benchmark_fused --release --features "ndarray,blas-accelerate" -- \
///     --weights model.safetensors

use std::time::Instant;

use burn::prelude::*;
use burn::backend::NdArray as B;
use clap::Parser;
use steegformer::{ModelConfig, data, channel_vocab};
use steegformer::model::steegformer::STEEGFormer;
use steegformer::model::cpu_fused::{
    self, extract_model_weights, forward_fused, ScratchBuffers, ModelWeightsRaw,
};
use steegformer::weights::{WeightMap, load_model_from_wm};

#[derive(Parser, Debug)]
#[command(about = "ST-EEGFormer — fused CPU benchmark")]
struct Args {
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    runs: usize,
    #[arg(long, default_value_t = false)]
    json: bool,
}

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

fn bench_standard(
    steeg: &steegformer::model::steegformer::STEEGFormerWithPE<B>,
    signal_norm: &Tensor<B, 3>,
    chan_idx: &Tensor<B, 2, Int>,
    warmup: usize,
    runs: usize,
) -> Vec<f64> {
    // Warmup
    for _ in 0..warmup {
        let out = steeg.forward(signal_norm.clone(), chan_idx.clone());
        let _ = out.into_data().to_vec::<f32>();
    }

    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let out = steeg.forward(signal_norm.clone(), chan_idx.clone());
        let _ = out.into_data().to_vec::<f32>();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times
}

fn bench_fused(
    weights: &ModelWeightsRaw,
    signal: &[f32],
    channel_indices: &[i64],
    n_channels: usize,
    n_samples: usize,
    warmup: usize,
    runs: usize,
) -> Vec<f64> {
    let dim = weights.dim;
    let n_heads = weights.n_heads;
    let ff_dim = weights.ff_dim;
    let num_patches = n_samples / weights.patch_size;
    let seq_total = num_patches * n_channels;
    let seq_with_cls = seq_total + 1;

    let mut scratch = ScratchBuffers::new(seq_with_cls, dim, n_heads, ff_dim);

    // Pre-normalize signal
    let mut signal_norm = signal.to_vec();
    cpu_fused::channel_normalize(&mut signal_norm, n_channels, n_samples);

    // Warmup
    for _ in 0..warmup {
        let _ = forward_fused(&signal_norm, channel_indices, weights, &mut scratch);
    }

    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = forward_fused(&signal_norm, channel_indices, weights, &mut scratch);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times
}

fn print_stats(label: &str, times: &[f64]) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(0.0f64, f64::max);
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    eprintln!("  {label:<40} mean={mean:>7.1}ms  min={min:>7.1}ms  max={max:>7.1}ms  std={std:.1}ms");
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let cfg = ModelConfig::small();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  ST-EEGFormer — Fused CPU Benchmark                         ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load model
    let steeg = if let Some(ref wp) = args.weights {
        let mut wm = WeightMap::from_file(wp)?;
        load_model_from_wm(&cfg, &mut wm, &device)?
    } else {
        STEEGFormer::new(&cfg, &device)
    };

    // Extract weights for fused path
    let raw_weights = extract_model_weights(&steeg);

    let channel_names = channel_vocab::BCI_COMP_IV_2A;
    let n_channels = channel_names.len();
    let indices = channel_vocab::channel_indices_unwrap(channel_names);

    // ── Standard input benchmark ─────────────────────────────────────────
    let n_samples = 768;
    let signal = generate_synthetic_eeg(n_channels, n_samples, 128.0);

    eprintln!("━━━ Standard input: {}ch × {} samples (S={}) ━━━",
        n_channels, n_samples, n_samples / cfg.patch_size * n_channels + 1);

    // Standard path
    let batch = data::build_batch::<B>(
        signal.clone(), indices.clone(), n_channels, n_samples, &device,
    );
    let signal_norm = data::channel_wise_normalize(batch.signal.clone());
    let std_times = bench_standard(&steeg, &signal_norm, &batch.channel_indices, args.warmup, args.runs);
    print_stats("Standard (Burn NdArray)", &std_times);

    // Fused path
    let fused_times = bench_fused(
        &raw_weights, &signal, &indices, n_channels, n_samples,
        args.warmup, args.runs,
    );
    print_stats("Fused (raw f32 + BLAS)", &fused_times);

    let std_mean = std_times.iter().sum::<f64>() / std_times.len() as f64;
    let fused_mean = fused_times.iter().sum::<f64>() / fused_times.len() as f64;
    eprintln!("\n  Speedup: {:.1}×\n", std_mean / fused_mean);

    // ── Verify numerical parity ──────────────────────────────────────────
    eprintln!("━━━ Numerical parity check ━━━");
    let std_out = steeg.forward(signal_norm.clone(), batch.channel_indices.clone());
    let std_vec = std_out.into_data().to_vec::<f32>().unwrap();

    let mut signal_norm_raw = signal.clone();
    cpu_fused::channel_normalize(&mut signal_norm_raw, n_channels, n_samples);
    let dim = cfg.embed_dim;
    let n_heads = cfg.num_heads;
    let ff_dim = (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize;
    let num_patches = n_samples / cfg.patch_size;
    let seq_total = num_patches * n_channels;
    let seq_with_cls = seq_total + 1;
    let mut scratch = ScratchBuffers::new(seq_with_cls, dim, n_heads, ff_dim);
    let fused_vec = forward_fused(&signal_norm_raw, &indices, &raw_weights, &mut scratch);

    let rmse = (std_vec.iter().zip(fused_vec.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / std_vec.len() as f32).sqrt();
    let max_diff = std_vec.iter().zip(fused_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("  Standard output (first 5): {:?}", &std_vec[..5.min(std_vec.len())]);
    eprintln!("  Fused output    (first 5): {:?}", &fused_vec[..5.min(fused_vec.len())]);
    eprintln!("  RMSE: {rmse:.6}");
    eprintln!("  Max diff: {max_diff:.6}");
    eprintln!();

    // ── Channel scaling ──────────────────────────────────────────────────
    eprintln!("━━━ Channel scaling (T={}) ━━━", n_samples);
    eprintln!("  {:>6}  {:>12}  {:>12}  {:>8}", "Chans", "Standard", "Fused", "Speedup");

    for &nc in &[4, 8, 16, 22, 32, 64] {
        if nc > channel_vocab::VOCAB_SIZE { continue; }
        let ch_indices: Vec<i64> = (0..nc as i64).collect();
        let sig = generate_synthetic_eeg(nc, n_samples, 128.0);

        // Standard
        let b = data::build_batch::<B>(sig.clone(), ch_indices.clone(), nc, n_samples, &device);
        let sn = data::channel_wise_normalize(b.signal.clone());
        let st = bench_standard(&steeg, &sn, &b.channel_indices, 3, 5);
        let st_mean = st.iter().sum::<f64>() / st.len() as f64;

        // Fused
        let ft = bench_fused(&raw_weights, &sig, &ch_indices, nc, n_samples, 3, 5);
        let ft_mean = ft.iter().sum::<f64>() / ft.len() as f64;

        eprintln!("  {:>6}  {:>9.1} ms  {:>9.1} ms  {:>6.1}×", nc, st_mean, ft_mean, st_mean / ft_mean);
    }
    eprintln!();

    // ── Sequence scaling ─────────────────────────────────────────────────
    eprintln!("━━━ Sequence scaling (C={}) ━━━", n_channels);
    eprintln!("  {:>8}  {:>12}  {:>12}  {:>8}", "Samples", "Standard", "Fused", "Speedup");

    for &ns in &[128, 256, 512, 768, 1024] {
        let sig = generate_synthetic_eeg(n_channels, ns, 128.0);

        // Standard
        let b = data::build_batch::<B>(sig.clone(), indices.clone(), n_channels, ns, &device);
        let sn = data::channel_wise_normalize(b.signal.clone());
        let st = bench_standard(&steeg, &sn, &b.channel_indices, 3, 5);
        let st_mean = st.iter().sum::<f64>() / st.len() as f64;

        // Fused
        let ft = bench_fused(&raw_weights, &sig, &indices, n_channels, ns, 3, 5);
        let ft_mean = ft.iter().sum::<f64>() / ft.len() as f64;

        eprintln!("  {:>8}  {:>9.1} ms  {:>9.1} ms  {:>6.1}×", ns, st_mean, ft_mean, st_mean / ft_mean);
    }

    if args.json {
        println!("{}", serde_json::json!({
            "standard_mean_ms": std_mean,
            "fused_mean_ms": fused_mean,
            "speedup": std_mean / fused_mean,
            "rmse": rmse,
            "max_diff": max_diff,
        }));
    }

    Ok(())
}
