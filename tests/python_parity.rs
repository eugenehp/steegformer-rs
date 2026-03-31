/// Python parity test: compare Rust output against Python-exported reference vectors.
///
/// Run `python scripts/export_parity_vectors.py` first to generate test vectors.

use burn::backend::NdArray as B;
use burn::prelude::*;
use steegformer::config::ModelConfig;
use steegformer::weights::{WeightMap, load_model_from_wm};

const VECTORS_PATH: &str = "tests/vectors/parity.safetensors";
const WEIGHTS_PATH: &str = "tests/vectors/encoder_small.safetensors";

fn load_parity_tensors() -> std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> {
    let bytes = std::fs::read(VECTORS_PATH).expect("Run: python scripts/export_parity_vectors.py");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut map = std::collections::HashMap::new();
    for (key, view) in st.tensors().into_iter() {
        let shape = view.shape().to_vec();
        let data: Vec<f32> = view.data().chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        map.insert(key.clone(), (data, shape));
    }
    map
}

fn tensor_from_ref<const N: usize>(
    refs: &std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> Tensor<B, N> {
    let (data, shape) = refs.get(key).unwrap_or_else(|| panic!("Missing key: {key}"));
    Tensor::<B, N>::from_data(TensorData::new(data.clone(), shape.clone()), device)
}

fn compare_tensors(
    name: &str,
    rust: &[f32],
    python: &[f32],
    atol: f32,
    rtol: f32,
) -> (f32, f32, f32) {
    assert_eq!(rust.len(), python.len(), "{name}: length mismatch {} vs {}", rust.len(), python.len());

    let mut max_abs_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut n_mismatch = 0usize;

    for (i, (&r, &p)) in rust.iter().zip(python.iter()).enumerate() {
        let abs_err = (r - p).abs();
        let threshold = atol + rtol * p.abs();
        if abs_err > threshold {
            n_mismatch += 1;
            if n_mismatch <= 3 {
                eprintln!("  {name}[{i}]: rust={r:.8} python={p:.8} diff={abs_err:.8}");
            }
        }
        max_abs_err = max_abs_err.max(abs_err);
        sum_sq_err += (abs_err as f64).powi(2);
    }

    let rmse = (sum_sq_err / rust.len() as f64).sqrt() as f32;

    // Pearson correlation
    let n = rust.len() as f64;
    let r_mean: f64 = rust.iter().map(|&v| v as f64).sum::<f64>() / n;
    let p_mean: f64 = python.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut cov = 0.0f64;
    let mut r_var = 0.0f64;
    let mut p_var = 0.0f64;
    for (&r, &p) in rust.iter().zip(python.iter()) {
        let rd = r as f64 - r_mean;
        let pd = p as f64 - p_mean;
        cov += rd * pd;
        r_var += rd * rd;
        p_var += pd * pd;
    }
    let pearson = if r_var > 0.0 && p_var > 0.0 {
        cov / (r_var.sqrt() * p_var.sqrt())
    } else {
        1.0  // both constant
    };

    eprintln!("  {name:40} RMSE={rmse:.8}  max_err={max_abs_err:.8}  r={pearson:.6}  mismatches={n_mismatch}/{}",
        rust.len());

    (rmse, max_abs_err, pearson as f32)
}

#[test]
fn test_patch_embed_parity() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let refs = load_parity_tensors();

    // Load model
    let cfg = ModelConfig::small();
    let mut wm = WeightMap::from_file(WEIGHTS_PATH).unwrap();
    let steeg = load_model_from_wm(&cfg, &mut wm, &device).unwrap();

    // Get input
    let input_signal: Tensor<B, 2> = tensor_from_ref(&refs, "input_signal", &device);
    let signal = input_signal.unsqueeze_dim::<3>(0);  // [1, C, T]

    // Run patch embed
    let x = steeg.model.patch_embed.forward(signal);
    let [b, seq, ch, d] = x.dims();
    let x_flat = x.reshape([b, seq * ch, d]);
    let rust_data = x_flat.into_data().to_vec::<f32>().unwrap();

    let (py_data, _) = refs.get("after_patch_embed").unwrap();
    let (rmse, max_err, pearson) = compare_tensors("patch_embed", &rust_data, py_data, 1e-5, 1e-4);

    assert!(rmse < 1e-5, "patch_embed RMSE too high: {rmse}");
    assert!(pearson > 0.999999, "patch_embed correlation too low: {pearson}");
}

#[test]
fn test_positional_encoding_parity() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let refs = load_parity_tensors();

    // Check temporal embedding
    let cfg = ModelConfig::small();
    let mut wm = WeightMap::from_file(WEIGHTS_PATH).unwrap();
    let steeg = load_model_from_wm(&cfg, &mut wm, &device).unwrap();

    // Python uses torch.arange(Seq) → 0-indexed, Seq=48 for 768/16
    let seq = 48usize;
    let ch_all = 4usize;
    let dmodel = 512usize;

    let seq_idx_data: Vec<i64> = (0..seq as i64).collect();
    let temp_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::new(seq_idx_data, vec![seq]), &device,
    ).unsqueeze_dim::<2>(0);

    let temp_emb = steeg.temporal_pe.forward(temp_idx);  // [1, Seq, D]
    let temp_emb = temp_emb.reshape([seq, dmodel]);

    // Tile across channels like Python does
    let temp_emb_tiled = temp_emb
        .unsqueeze_dim::<3>(1)
        .expand([seq, ch_all, dmodel])
        .reshape([seq * ch_all, dmodel]);

    let rust_data = temp_emb_tiled.into_data().to_vec::<f32>().unwrap();
    let (py_data, _) = refs.get("temporal_embedding").unwrap();
    let (rmse, _, pearson) = compare_tensors("temporal_embedding", &rust_data, py_data, 1e-6, 1e-5);

    assert!(rmse < 1e-6, "temporal_embedding RMSE too high: {rmse}");
    assert!(pearson > 0.999999, "temporal_embedding correlation too low: {pearson}");
}

#[test]
fn test_channel_embedding_parity() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let refs = load_parity_tensors();

    let cfg = ModelConfig::small();
    let mut wm = WeightMap::from_file(WEIGHTS_PATH).unwrap();
    let steeg = load_model_from_wm(&cfg, &mut wm, &device).unwrap();

    // Channel indices: [25, 130, 2, 1]
    let chan_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![25i64, 130, 2, 1], vec![4]), &device,
    ).unsqueeze_dim::<2>(0);  // [1, 4]

    let ch_emb = steeg.model.channel_embed.forward(chan_idx);  // [1, 4, 512]
    let ch_emb = ch_emb.reshape([4, 512]);

    // Tile across Seq (48 time patches)
    let seq = 48;
    let ch_all = 4;
    let dmodel = 512;
    let ch_emb_tiled = ch_emb
        .unsqueeze_dim::<3>(0)       // [1, 4, 512]
        .expand([seq, ch_all, dmodel])  // [48, 4, 512]
        .reshape([seq * ch_all, dmodel]);

    let rust_data = ch_emb_tiled.into_data().to_vec::<f32>().unwrap();
    let (py_data, _) = refs.get("channel_embedding").unwrap();
    let (rmse, _, pearson) = compare_tensors("channel_embedding", &rust_data, py_data, 1e-6, 1e-5);

    assert!(rmse < 1e-6, "channel_embedding RMSE too high: {rmse}");
    assert!(pearson > 0.999999, "channel_embedding correlation too low: {pearson}");
}

#[test]
fn test_full_encoder_parity() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let refs = load_parity_tensors();

    let cfg = ModelConfig::small();
    let mut wm = WeightMap::from_file(WEIGHTS_PATH).unwrap();
    let steeg = load_model_from_wm(&cfg, &mut wm, &device).unwrap();

    // Input
    let input_signal: Tensor<B, 2> = tensor_from_ref(&refs, "input_signal", &device);
    let signal = input_signal.unsqueeze_dim::<3>(0);  // [1, C, T]

    let chan_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![25i64, 130, 2, 1], vec![4]), &device,
    ).unsqueeze_dim::<2>(0);  // [1, 4]

    // Run full encoder
    let output = steeg.forward_features(signal, chan_idx);
    let rust_data = output.into_data().to_vec::<f32>().unwrap();

    // Compare against CLS token output (default: no global pool)
    let (py_data, _) = refs.get("cls_output").unwrap();
    let (rmse, max_err, pearson) = compare_tensors("cls_output", &rust_data, py_data, 1e-3, 1e-2);

    eprintln!("\n  ══ FULL ENCODER PARITY ══");
    eprintln!("  RMSE:    {rmse:.8}");
    eprintln!("  Max err: {max_err:.8}");
    eprintln!("  Pearson: {pearson:.8}");

    assert!(pearson > 0.99, "Full encoder Pearson r too low: {pearson}");
    assert!(rmse < 0.1, "Full encoder RMSE too high: {rmse}");
}

#[test]
fn test_after_block0_parity() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let refs = load_parity_tensors();

    let cfg = ModelConfig::small();
    let mut wm = WeightMap::from_file(WEIGHTS_PATH).unwrap();
    let steeg = load_model_from_wm(&cfg, &mut wm, &device).unwrap();

    // Reconstruct input up to block 0
    let input_signal: Tensor<B, 2> = tensor_from_ref(&refs, "input_signal", &device);
    let signal = input_signal.unsqueeze_dim::<3>(0);

    let chan_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![25i64, 130, 2, 1], vec![4]), &device,
    ).unsqueeze_dim::<2>(0);

    // Manually run through the model step by step
    let x = steeg.model.patch_embed.forward(signal.clone());
    let [b, seq, ch_all, dmodel] = x.dims();
    let seq_total = seq * ch_all;
    let mut x = x.reshape([b, seq_total, dmodel]);

    // Channel embedding
    let ch_emb_small = steeg.model.channel_embed.forward(chan_idx);
    let ch_emb = ch_emb_small
        .unsqueeze_dim::<4>(1)
        .expand([b, seq, ch_all, dmodel])
        .reshape([b, seq_total, dmodel]);

    // Temporal embedding (0-indexed)
    let seq_idx_data: Vec<i64> = (0..seq as i64).collect();
    let temp_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::new(seq_idx_data, vec![seq]), &signal.device(),
    ).unsqueeze_dim::<2>(0);
    let temp_emb_small = steeg.temporal_pe.forward(temp_idx).reshape([seq, dmodel]);
    let temp_emb_flat = temp_emb_small
        .unsqueeze_dim::<3>(1)
        .expand([seq, ch_all, dmodel])
        .reshape([seq_total, dmodel]);
    let tp_emb = temp_emb_flat
        .unsqueeze_dim::<3>(0)
        .expand([b, seq_total, dmodel]);

    x = x + tp_emb + ch_emb;

    // Check after_pos_encoding
    {
        let rust_data = x.clone().reshape([seq_total, dmodel]).into_data().to_vec::<f32>().unwrap();
        let (py_data, _) = refs.get("after_pos_encoding").unwrap();
        let (rmse, _, pearson) = compare_tensors("after_pos_encoding", &rust_data, py_data, 1e-5, 1e-4);
        assert!(rmse < 1e-5, "after_pos_encoding RMSE: {rmse}");
        assert!(pearson > 0.999999, "after_pos_encoding r: {pearson}");
    }

    // CLS token
    let cls_pe = steeg.temporal_pe.get_cls_token();
    let cls_token = steeg.model.cls_token.val()
        + cls_pe.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
    let cls_tokens = cls_token.expand([b, 1, dmodel]);
    x = Tensor::cat(vec![cls_tokens, x], 1);

    // Check after_cls_prepend
    {
        let rust_data = x.clone().reshape([1 + seq_total, dmodel]).into_data().to_vec::<f32>().unwrap();
        let (py_data, _) = refs.get("after_cls_prepend").unwrap();
        let (rmse, _, pearson) = compare_tensors("after_cls_prepend", &rust_data, py_data, 1e-5, 1e-4);
        assert!(rmse < 1e-5, "after_cls_prepend RMSE: {rmse}");
        assert!(pearson > 0.999999, "after_cls_prepend r: {pearson}");
    }

    // Block 0
    x = steeg.model.blocks[0].forward(x);

    {
        let rust_data = x.clone().reshape([1 + seq_total, dmodel]).into_data().to_vec::<f32>().unwrap();
        let (py_data, _) = refs.get("after_block_0").unwrap();
        let (rmse, max_err, pearson) = compare_tensors("after_block_0", &rust_data, py_data, 1e-4, 1e-3);
        assert!(pearson > 0.9999, "after_block_0 r: {pearson}");
        assert!(rmse < 0.001, "after_block_0 RMSE: {rmse}");
    }
}
