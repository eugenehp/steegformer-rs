/// Example: Reconstruct — basic model loading and forward pass.
///
/// Usage:
///   cargo run --example reconstruct --release -- --weights model.safetensors

use std::time::Instant;
use clap::Parser;

use steegformer::{ModelConfig, data, channel_vocab};
use steegformer::model::steegformer::STEEGFormer;

#[derive(Parser, Debug)]
#[command(about = "ST-EEGFormer — model loading and forward pass test")]
struct Args {
    #[arg(long, default_value = "small")]
    variant: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, short = 'v')]
    verbose: bool,
}

// ── Backend ──────────────────────────────────────────────────────────────────
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
    let device = get_device();
    let t0 = Instant::now();

    let cfg = match args.variant.as_str() {
        "small" => ModelConfig::small(),
        "base"  => ModelConfig::base(),
        "large" => ModelConfig::large(),
        _ => anyhow::bail!("Unknown variant: {}", args.variant),
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ST-EEGFormer — Forward Pass Test                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create model (random weights if no weights file provided)
    let steeg = if let Some(ref weights_path) = args.weights {
        println!("▸ Loading weights from {weights_path} …");
        let mut wm = steegformer::weights::WeightMap::from_file(weights_path)?;
        if args.verbose { wm.print_keys(); }
        steegformer::weights::load_model_from_wm(&cfg, &mut wm, &device)?
    } else {
        println!("▸ Initializing with random weights (no --weights provided)");
        STEEGFormer::new(&cfg, &device)
    };

    println!("  embed_dim={}, depth={}, heads={}, patch={}\n",
        cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.patch_size);

    // Create test input
    let channel_names = channel_vocab::BCI_COMP_IV_2A;
    let n_channels = channel_names.len();
    let n_samples = 768;  // 6s @ 128 Hz

    println!("▸ Running forward pass: {} channels × {} samples", n_channels, n_samples);

    let signal = vec![0.01f32; n_channels * n_samples];
    let batch = data::build_batch_named::<B>(signal, channel_names, n_samples, &device);

    let t_inf = Instant::now();
    let signal_norm = data::channel_wise_normalize(batch.signal.clone());
    let output = steeg.forward(signal_norm, batch.channel_indices.clone());
    let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;

    let shape = output.dims();
    println!("  Output shape: {:?}  ({ms_infer:.1} ms)", shape);

    let output_vec = output.into_data().to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let mean: f64 = output_vec.iter().map(|&v| v as f64).sum::<f64>() / output_vec.len() as f64;
    let std: f64 = (output_vec.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / output_vec.len() as f64).sqrt();
    println!("  mean={mean:+.6}  std={std:.6}");

    if args.verbose {
        println!("  First 10 values: {:?}", &output_vec[..10.min(output_vec.len())]);
    }

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Total: {ms_total:.0} ms");
    Ok(())
}
