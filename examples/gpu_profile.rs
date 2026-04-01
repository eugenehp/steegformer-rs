/// GPU dispatch profiling — measures time per operation type.
///
/// Usage:
///   cargo run --example gpu_profile --release --no-default-features --features wgpu-kernels

use std::time::Instant;
use burn::prelude::*;
use steegformer::{ModelConfig, channel_vocab};
use steegformer::model::steegformer::{STEEGFormer, WeightCache, BlockWeightCache};

type B = burn::backend::wgpu::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
use steegformer::model::FusedOps;

fn device() -> burn::backend::wgpu::WgpuDevice { burn::backend::wgpu::WgpuDevice::DefaultDevice }

fn sync(t: &Tensor<B, 3>) { let _ = t.to_data().to_vec::<f32>(); }
fn sync2(t: &Tensor<B, 2>) { let _ = t.to_data().to_vec::<f32>(); }
fn sync4(t: &Tensor<B, 4>) { let _ = t.to_data().to_vec::<f32>(); }

fn bench<F: FnMut()>(label: &str, warmup: usize, runs: usize, mut f: F) -> f64 {
    for _ in 0..warmup { f(); }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!("  {label:<45} mean={mean:>7.2}ms  min={min:>7.2}ms");
    mean
}

fn main() {
    let dev = device();
    let cfg = ModelConfig::small();
    let steeg = STEEGFormer::new(&cfg, &dev);
    let wc = steeg.weight_cache.as_ref().unwrap();

    let n_channels = 22usize;
    let n_samples = 768usize;
    let d = cfg.embed_dim;
    let num_patches = n_samples / cfg.patch_size;
    let seq_total = num_patches * n_channels;
    let s = seq_total + 1; // with CLS

    eprintln!("S={s}, D={d}, H={}, dh={}", cfg.num_heads, d / cfg.num_heads);
    eprintln!();

    // Create test tensors on GPU
    let x3 = Tensor::<B, 3>::random([1, s, d], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    let bw = &wc.blocks[0];

    let warmup = 5;
    let runs = 10;

    eprintln!("━━━ Per-operation profiling ━━━");

    // 1. LayerNorm
    bench("fused_layernorm", warmup, runs, || {
        let out = B::fused_layernorm(x3.clone(), bw.norm1_weight.clone(), bw.norm1_bias.clone(), bw.norm1_eps);
        sync(&out);
    });

    // 2. QKV matmul
    bench("QKV matmul [S,D]×[1,D,3D]", warmup, runs, || {
        let out = x3.clone().matmul(bw.qkv_w.clone());
        sync(&out);
    });

    // 3. Split QKV
    let qkv = x3.clone().matmul(bw.qkv_w.clone());
    bench("fused_split_qkv_scaled", warmup, runs, || {
        let (q, k, v) = B::fused_split_qkv_scaled(qkv.clone(), bw.qkv_bias.clone(), bw.n_heads, bw.head_dim, bw.scale);
        sync4(&q);
    });

    // 4. Attention (decomposed: matmul + softmax + matmul)
    let (q, k, v) = B::fused_split_qkv_scaled(qkv.clone(), bw.qkv_bias.clone(), bw.n_heads, bw.head_dim, bw.scale);
    let _ = q.to_data(); // sync
    bench("Q×K_T matmul [B,H,S,dh]×[B,H,dh,S]", warmup, runs, || {
        let scores = q.clone().matmul(k.clone().swap_dims(2, 3));
        sync4(&scores);
    });

    bench("fused_softmax [B,H,S,S]", warmup, runs, || {
        let scores = q.clone().matmul(k.clone().swap_dims(2, 3));
        let attn = B::fused_softmax(scores, 3);
        sync4(&attn);
    });

    bench("attn×V matmul [B,H,S,S]×[B,H,S,dh]", warmup, runs, || {
        let scores = q.clone().matmul(k.clone().swap_dims(2, 3));
        let attn = B::fused_softmax(scores, 3);
        let out = attn.matmul(v.clone());
        sync4(&out);
    });

    // 5. Full attention (Q×K_T → softmax → attn×V)
    bench("full attention (3 dispatches)", warmup, runs, || {
        let scores = q.clone().matmul(k.clone().swap_dims(2, 3));
        let attn = B::fused_softmax(scores, 3);
        let out = attn.matmul(v.clone());
        sync4(&out);
    });

    // 6. Merge heads
    let attn_out = Tensor::<B, 4>::random([1, cfg.num_heads, s, d/cfg.num_heads], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    bench("fused_merge_heads", warmup, runs, || {
        let out = B::fused_merge_heads(attn_out.clone(), bw.n_heads, bw.head_dim);
        sync(&out);
    });

    // 7. Proj matmul
    bench("proj matmul [B,S,D]×[1,D,D]", warmup, runs, || {
        let out = x3.clone().matmul(bw.proj_w.clone());
        sync(&out);
    });

    // 8. Bias + residual + layernorm
    let proj_out = x3.clone().matmul(bw.proj_w.clone());
    bench("fused_bias_residual_add_layernorm", warmup, runs, || {
        let (s, n) = B::fused_bias_residual_add_layernorm(
            x3.clone(), proj_out.clone(), bw.proj_bias.clone(),
            bw.norm2_weight.clone(), bw.norm2_bias.clone(), bw.norm2_eps,
        );
        sync(&n);
    });

    // 9. FC1 matmul
    bench("FC1 matmul [B,S,D]×[1,D,ff]", warmup, runs, || {
        let out = x3.clone().matmul(bw.fc1_w.clone());
        sync(&out);
    });

    // 10. Bias+GELU
    let ff_dim = bw.fc1_w.dims()[2];
    let h = Tensor::<B, 3>::random([1, s, ff_dim], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    bench("fused_bias_gelu", warmup, runs, || {
        let out = B::fused_bias_gelu(h.clone(), bw.fc1_bias.clone());
        sync(&out);
    });

    // 11. FC2 matmul
    bench("FC2 matmul [B,S,ff]×[1,ff,D]", warmup, runs, || {
        let out = h.clone().matmul(bw.fc2_w.clone());
        sync(&out);
    });

    // 12. Bias + residual add
    bench("fused_bias_residual_add", warmup, runs, || {
        let out = B::fused_bias_residual_add(x3.clone(), x3.clone(), bw.fc2_bias.clone());
        sync(&out);
    });

    eprintln!();
    eprintln!("━━━ Full block profiling ━━━");

    // Full block (cached)
    let n1 = B::fused_layernorm(x3.clone(), bw.norm1_weight.clone(), bw.norm1_bias.clone(), bw.norm1_eps);
    let _ = n1.to_data();
    bench("forward_block_cached (full block)", warmup, runs, || {
        let (out, _next_n1) = steegformer::model::steegformer::forward_block_cached::<B>(
            x3.clone(), n1.clone(), bw, None,
        );
        sync(&out);
    });

    eprintln!();
    eprintln!("━━━ Full forward profiling ━━━");

    // Full forward
    let signal: Vec<f32> = (0..n_channels * n_samples).map(|i| (i as f32 * 0.001).sin()).collect();
    let indices = channel_vocab::channel_indices_unwrap(channel_vocab::BCI_COMP_IV_2A);
    let batch = steegformer::data::build_batch::<B>(signal, indices, n_channels, n_samples, &dev);
    let signal_norm = steegformer::data::channel_wise_normalize(batch.signal.clone());

    bench("full forward (22ch × 768)", warmup, runs, || {
        let out = steeg.forward(signal_norm.clone(), batch.channel_indices.clone());
        sync2(&out);
    });

    // Pre-transformer overhead
    bench("pre-transformer (patch+PE+CLS)", warmup, runs, || {
        let x = steeg.model.patch_embed.forward(signal_norm.clone());
        let [_, seq, ch_all, _] = x.dims();
        let ch_emb = steeg.model.channel_embed.forward(batch.channel_indices.clone())
            .unsqueeze_dim::<4>(1);
        let tp_emb = steeg.temporal_pe.pe.clone()
            .narrow(0, 0, seq)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(0);
        let x = (x + ch_emb + tp_emb)
            .reshape([1, seq * ch_all, d]);
        let cls = steeg.cls_with_pe.clone().expand([1, 1, d]);
        let x = Tensor::cat(vec![cls, x], 1);
        sync(&x);
    });
}
