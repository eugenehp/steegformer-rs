pub mod norm;
pub mod feedforward;
pub mod attention;
pub mod encoder_block;
pub mod patch_embed;
pub mod positional;
pub mod steegformer;

#[cfg(feature = "wgpu-kernels")]
pub mod kernels;
pub mod fused;

#[cfg(feature = "blas-accelerate")]
pub mod cpu_fused;

pub use fused::FusedOps;
