/// ST-EEGFormer EEG inference — thin CLI.
///
/// Build — CPU (default):
///   cargo build --release
///
/// Build — GPU:
///   cargo build --release --no-default-features --features wgpu
///
/// Usage:
///   infer --weights <st> --config <json>

use std::{path::Path, time::Instant};
use clap::Parser;
use steegformer::{STEEGFormerEncoder, data, channel_vocab};

// ── Backend ───────────────────────────────────────────────────────────────────
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::{Wgpu as B, wgpu::WgpuDevice as Device};
    pub fn device() -> Device { Device::DefaultDevice }
    #[cfg(feature = "metal")]
    pub const NAME: &str = "GPU (wgpu — Metal / MSL shaders)";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "GPU (wgpu — Vulkan / SPIR-V shaders)";
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    pub const NAME: &str = "GPU (wgpu — WGSL shaders)";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "CPU (NdArray + Apple Accelerate)";
    #[cfg(feature = "openblas-system")]
    pub const NAME: &str = "CPU (NdArray + OpenBLAS)";
    #[cfg(not(any(feature = "blas-accelerate", feature = "openblas-system")))]
    pub const NAME: &str = "CPU (NdArray + Rayon)";
}

use backend::{B, device};

// ── CLI ───────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(about = "ST-EEGFormer EEG model inference (Burn 0.20.1)")]
struct Args {
    /// Safetensors weights file.
    #[arg(long)]
    weights: String,

    /// config.json.
    #[arg(long)]
    config: String,

    /// Model variant (small, base, large). Used if no config file.
    #[arg(long, default_value = "large")]
    variant: String,

    /// Print details.
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t0   = Instant::now();
    let dev  = device();

    println!("Backend : {}", backend::NAME);

    // Load model
    let (encoder, ms_weights) = STEEGFormerEncoder::<B>::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        dev.clone(),
    )?;

    println!("Model   : {}  ({ms_weights:.0} ms)", encoder.describe());

    // Example: create a dummy input for testing
    let channel_names: Vec<&str> = channel_vocab::STANDARD_10_20.to_vec();
    let n_channels = channel_names.len();
    let n_samples = 768;  // 6s @ 128 Hz

    // Dummy signal (zeros)
    let signal = vec![0.0f32; n_channels * n_samples];
    let indices = channel_vocab::channel_indices_unwrap(&channel_names);

    let batch = data::build_batch::<B>(
        signal, indices, n_channels, n_samples, &dev,
    );

    let t_inf = Instant::now();
    let result = encoder.run_batch(&batch)?;
    let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;

    println!("Output  : shape={:?}  ({ms_infer:.1} ms)", result.shape);

    if args.verbose {
        let mean: f64 = result.output.iter().map(|&v| v as f64).sum::<f64>() / result.output.len() as f64;
        let std: f64 = (result.output.iter().map(|&v| {
            let d = v as f64 - mean; d * d
        }).sum::<f64>() / result.output.len() as f64).sqrt();
        println!("  mean={mean:+.4}  std={std:.4}");
    }

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("── Timing ───────────────────────────────────────────────────────");
    println!("  Weights  : {ms_weights:.0} ms");
    println!("  Infer    : {ms_infer:.0} ms");
    println!("  Total    : {ms_total:.0} ms");

    Ok(())
}
