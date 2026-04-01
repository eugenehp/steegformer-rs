//! Dump generated MSL shader source for our CubeCL kernels.
//! cargo run --example dump_msl --release --no-default-features --features wgpu-kernels

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

// ── Our kernels (copied from kernels.rs to get access to compile) ────────

#[cube(launch)]
fn bias_gelu_kernel<F: Float>(
    x: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    bias_dim: u32,
    total: usize,
) {
    let base = ABSOLUTE_POS * 4;
    let coeff = F::new(1.702);
    if base < total {
        let pos0 = base;
        let bias_idx0 = (pos0 as u32) % bias_dim;
        let v0 = x[pos0] + bias[bias_idx0 as usize];
        out[pos0] = v0 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v0));
    }
    if base + 1 < total {
        let pos1 = base + 1;
        let bias_idx1 = (pos1 as u32) % bias_dim;
        let v1 = x[pos1] + bias[bias_idx1 as usize];
        out[pos1] = v1 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v1));
    }
    if base + 2 < total {
        let pos2 = base + 2;
        let bias_idx2 = (pos2 as u32) % bias_dim;
        let v2 = x[pos2] + bias[bias_idx2 as usize];
        out[pos2] = v2 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v2));
    }
    if base + 3 < total {
        let pos3 = base + 3;
        let bias_idx3 = (pos3 as u32) % bias_dim;
        let v3 = x[pos3] + bias[bias_idx3 as usize];
        out[pos3] = v3 / (F::new(1.0) + F::exp(F::new(0.0) - coeff * v3));
    }
}

#[cube(launch)]
fn layernorm_kernel<F: Float>(
    x: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    d: u32,
    eps: f32,
) {
    let row = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let n_per_lane = d / PLANE_DIM;
    let base = row * d + lane;
    let mut sum_val = F::new(0.0);
    let mut sum_sq = F::new(0.0);
    for i in 0u32..n_per_lane {
        let v = x[(base + i * PLANE_DIM) as usize];
        sum_val += v;
        sum_sq += v * v;
    }
    let total_sum = plane_sum(sum_val);
    let total_sq = plane_sum(sum_sq);
    let d_f = F::cast_from(d);
    let mean = total_sum / d_f;
    let variance = total_sq / d_f - mean * mean;
    let inv_std = F::powf(variance + F::cast_from(eps), F::new(-0.5));
    for i in 0u32..n_per_lane {
        let idx = (base + i * PLANE_DIM) as usize;
        let w_idx = (lane + i * PLANE_DIM) as usize;
        let normalized = (x[idx] - mean) * inv_std;
        out[idx] = normalized * weight[w_idx] + bias[w_idx];
    }
}

#[cube(launch)]
fn softmax_kernel<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<F>,
    d: u32,
) {
    let row = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let base = row * d;
    let n_per_lane = (d + PLANE_DIM - 1) / PLANE_DIM;
    let mut local_max = F::new(-65504.0);
    for i in 0u32..n_per_lane {
        let col = lane + i * PLANE_DIM;
        if col < d {
            local_max = F::max(local_max, x[(base + col) as usize]);
        }
    }
    let row_max = plane_max(local_max);
    let mut local_sum = F::new(0.0);
    for i in 0u32..n_per_lane {
        let col = lane + i * PLANE_DIM;
        if col < d {
            let idx = (base + col) as usize;
            let e = F::exp(x[idx] - row_max);
            out[idx] = e;
            local_sum += e;
        }
    }
    let total_sum = plane_sum(local_sum);
    let inv_sum = F::new(1.0) / total_sum;
    for i in 0u32..n_per_lane {
        let col = lane + i * PLANE_DIM;
        if col < d {
            let idx = (base + col) as usize;
            out[idx] = out[idx] * inv_sum;
        }
    }
}

fn main() {
    use cubecl::ir::CubeDim as IrCubeDim;
    
    // Use the MslCompiler directly
    let compiler = cubecl_cpp::MslCompiler::default();
    
    // Compile bias_gelu kernel
    eprintln!("═══ Compiling bias_gelu kernel ═══");
    let kernel_gelu = bias_gelu_kernel::create_kernel::<f32, WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim { x: 256, y: 1, z: 1 },
        TensorArg::<WgpuRuntime>::alias(0),
        TensorArg::<WgpuRuntime>::alias(1),
        TensorArg::<WgpuRuntime>::alias(2),
        ScalarArg::new(2048u32),
        ScalarArg::new(2164736usize),
    );
    
    // Get the kernel source through the compile infrastructure
    use cubecl::Compiler;
    let compiled = compiler.compile(
        kernel_gelu,
        &Default::default(),
        cubecl::ExecutionMode::Checked,
    );
    match compiled {
        Ok(c) => {
            println!("// ═══ BIAS_GELU MSL ═══");
            println!("{}", c.source);
            std::fs::write("/tmp/bias_gelu.metal", &c.source).ok();
        }
        Err(e) => eprintln!("Failed to compile bias_gelu: {e:?}"),
    }

    // Compile layernorm kernel
    let kernel_ln = layernorm_kernel::create_kernel::<f32, WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim { x: 32, y: 1, z: 1 },
        TensorArg::<WgpuRuntime>::alias(0),
        TensorArg::<WgpuRuntime>::alias(1),
        TensorArg::alias(2),
        TensorArg::alias(3),
        ScalarArg::new(512u32),
        ScalarArg::new(1e-6f32),
    );
    let compiled = compiler.compile(
        kernel_ln,
        &Default::default(),
        cubecl::ExecutionMode::Checked,
    );
    match compiled {
        Ok(c) => {
            println!("// ═══ LAYERNORM MSL ═══");
            println!("{}", c.source);
            std::fs::write("/tmp/layernorm.metal", &c.source).ok();
        }
        Err(e) => eprintln!("Failed to compile layernorm: {e:?}"),
    }

    // Compile softmax kernel
    let kernel_sm = softmax_kernel::create_kernel::<f32, WgpuRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim { x: 32, y: 1, z: 1 },
        TensorArg::<WgpuRuntime>::alias(0),
        TensorArg::alias(1),
        ScalarArg::new(1057u32),
    );
    let compiled = compiler.compile(
        kernel_sm,
        &Default::default(),
        cubecl::ExecutionMode::Checked,
    );
    match compiled {
        Ok(c) => {
            println!("// ═══ SOFTMAX MSL ═══");
            println!("{}", c.source);
            std::fs::write("/tmp/softmax.metal", &c.source).ok();
        }
        Err(e) => eprintln!("Failed to compile softmax: {e:?}"),
    }
    
    eprintln!("Shaders written to /tmp/*.metal");
}
