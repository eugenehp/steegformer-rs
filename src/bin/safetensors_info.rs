/// Print safetensors file info: keys, shapes, dtypes.

use clap::Parser;
use safetensors::SafeTensors;

#[derive(Parser, Debug)]
#[command(about = "Inspect safetensors file")]
struct Args {
    /// Path to safetensors file.
    path: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.path)?;
    let st = SafeTensors::deserialize(&bytes)?;

    let mut total_params = 0usize;
    let mut entries: Vec<(String, Vec<usize>, safetensors::Dtype)> = st.tensors()
        .into_iter()
        .map(|(key, view)| {
            let shape = view.shape().to_vec();
            let dtype = view.dtype();
            (key, shape, dtype)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    for (key, shape, dtype) in &entries {
        let numel: usize = shape.iter().product();
        total_params += numel;
        println!("  {key:80}  {shape:?}  {dtype:?}");
    }

    println!("\n  {} tensors, {:.2}M parameters", entries.len(),
        total_params as f64 / 1e6);
    println!("  File size: {:.2} MB", bytes.len() as f64 / 1e6);
    Ok(())
}
