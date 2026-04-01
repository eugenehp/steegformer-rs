//! Quick test: 2D vs 3D matmul performance on GPU

use std::time::Instant;
use burn::prelude::*;

type B = burn::backend::wgpu::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

fn device() -> burn::backend::wgpu::WgpuDevice { burn::backend::wgpu::WgpuDevice::DefaultDevice }
fn sync3(t: &Tensor<B, 3>) { let _ = t.to_data().to_vec::<f32>(); }
fn sync2(t: &Tensor<B, 2>) { let _ = t.to_data().to_vec::<f32>(); }

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
    eprintln!("  {label:<50} mean={mean:>6.2}ms  min={min:>6.2}ms");
    mean
}

fn main() {
    let dev = device();
    let s = 1057usize;
    let d = 512usize;
    let ff = 2048usize;

    let x3 = Tensor::<B, 3>::random([1, s, d], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    let w_qkv_3d = Tensor::<B, 3>::random([1, d, 3*d], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    let w_qkv_2d = Tensor::<B, 2>::random([d, 3*d], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    let w_fc1_3d = Tensor::<B, 3>::random([1, d, ff], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    let w_fc1_2d = Tensor::<B, 2>::random([d, ff], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);

    eprintln!("━━━ QKV matmul: [1,{s},{d}] × [?,{d},{}", 3*d);

    bench("3D batched: [1,S,D] × [1,D,3D]", 10, 10, || {
        let out = x3.clone().matmul(w_qkv_3d.clone());
        sync3(&out);
    });

    bench("2D reshape: [S,D] × [D,3D]", 10, 10, || {
        let x2 = x3.clone().reshape([s, d]);
        let out = x2.matmul(w_qkv_2d.clone());
        let out3 = out.reshape([1, s, 3*d]);
        sync3(&out3);
    });

    eprintln!();
    eprintln!("━━━ FC1 matmul: [1,{s},{d}] × [?,{d},{ff}]");

    bench("3D batched: [1,S,D] × [1,D,ff]", 10, 10, || {
        let out = x3.clone().matmul(w_fc1_3d.clone());
        sync3(&out);
    });

    bench("2D reshape: [S,D] × [D,ff]", 10, 10, || {
        let x2 = x3.clone().reshape([s, d]);
        let out = x2.matmul(w_fc1_2d.clone());
        let out3 = out.reshape([1, s, ff]);
        sync3(&out3);
    });

    eprintln!();
    eprintln!("━━━ Proj matmul: [1,{s},{d}] × [?,{d},{d}]");
    let w_proj_3d = Tensor::<B, 3>::random([1, d, d], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
    let w_proj_2d = Tensor::<B, 2>::random([d, d], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);

    bench("3D batched: [1,S,D] × [1,D,D]", 10, 10, || {
        let out = x3.clone().matmul(w_proj_3d.clone());
        sync3(&out);
    });

    bench("2D reshape: [S,D] × [D,D]", 10, 10, || {
        let x2 = x3.clone().reshape([s, d]);
        let out = x2.matmul(w_proj_2d.clone());
        let out3 = out.reshape([1, s, d]);
        sync3(&out3);
    });
}
