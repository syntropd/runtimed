//! Safetensors loading and name mapping integration tests.

use candle_core::Device;
use runtimed_model::weights::Weights;
use runtimed_model::LoraAdapter;
use safetensors::tensor::Dtype;
use safetensors::serialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("runtimed-safetensors-test-{name}.safetensors"))
}

#[test]
fn loads_safetensors_and_maps_names() {
    let dev = Device::Cpu;
    let mut tensors = HashMap::new();

    // 2x2 f32 embedding weight
    let emb_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 2], &emb_data).unwrap(),
    );

    // 2x2 f16 q_proj weight (1.0 is 0x3c00 in f16)
    let q_data: Vec<u8> = vec![
        0x00, 0x3c, 0x00, 0x00, // [1.0, 0.0]
        0x00, 0x00, 0x00, 0x3c, // [0.0, 1.0]
    ];
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        safetensors::tensor::TensorView::new(Dtype::F16, vec![2, 2], &q_data).unwrap(),
    );

    // 2x2 fp8_e4m3 gate_proj weight
    let fp8_data: Vec<u8> = vec![0x38, 0x38, 0x38, 0x38]; // 1.0 in E4M3
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        safetensors::tensor::TensorView::new(Dtype::F8_E4M3, vec![2, 2], &fp8_data).unwrap(),
    );

    let serialized = serialize(&tensors, &None).unwrap();
    let p = tmp("load_map");
    fs::write(&p, serialized).unwrap();

    let weights = Weights::load_safetensors(&p, &dev).unwrap();
    let _ = fs::remove_file(&p);

    assert!(weights.get("token_embd.weight").is_ok());
    assert!(weights.get("blk.0.attn_q.weight").is_ok());
    assert!(weights.get("blk.0.ffn_gate.weight").is_ok());

    let emb = weights.get("token_embd.weight").unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(emb, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let q = weights.get("blk.0.attn_q.weight").unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(q, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

    let gate = weights.get("blk.0.ffn_gate.weight").unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(gate, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
}

#[test]
fn loads_safetensors_lora_and_fuses() {
    let dev = Device::Cpu;
    // Base weight
    let mut base_tensors = HashMap::new();
    let w_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    base_tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 2], &w_data).unwrap(),
    );
    let base_ser = serialize(&base_tensors, &None).unwrap();
    let base_p = tmp("base_for_lora");
    fs::write(&base_p, base_ser).unwrap();
    let mut weights = Weights::load_safetensors(&base_p, &dev).unwrap();
    let _ = fs::remove_file(&base_p);

    // LoRA A and B: A is [1, 2], B is [2, 1]
    let mut lora_tensors = HashMap::new();
    let a_data: Vec<u8> = vec![1.0f32, 1.0]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let b_data: Vec<u8> = vec![2.0f32, 2.0]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    lora_tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        safetensors::tensor::TensorView::new(Dtype::F32, vec![1, 2], &a_data).unwrap(),
    );
    lora_tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 1], &b_data).unwrap(),
    );
    let lora_ser = serialize(&lora_tensors, &None).unwrap();
    let lora_p = tmp("adapter_model");
    let lora_cfg_p = lora_p.with_file_name("adapter_config.json");
    fs::write(&lora_p, lora_ser).unwrap();
    fs::write(
        &lora_cfg_p,
        r#"{"lora_alpha": 1.0, "r": 1, "peft_type": "LORA"}"#,
    )
    .unwrap();

    let adapter = LoraAdapter::load(&lora_p, &dev, "qwen2").unwrap();
    let fused = adapter.fuse_into(&mut weights).unwrap();
    assert_eq!(fused, vec!["blk.0.attn_q.weight"]);

    let _ = fs::remove_file(&lora_p);
    let _ = fs::remove_file(&lora_cfg_p);

    // Fused: [[1, 2], [3, 4]] + 1.0 * [[2], [2]] * [[1, 1]]
    // delta = [[2, 2], [2, 2]] -> result = [[3, 4], [5, 6]]
    let fused_w = weights.get("blk.0.attn_q.weight").unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(fused_w, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
}

#[test]
fn loads_safetensors_with_tied_embeddings() {
    let dev = Device::Cpu;
    let mut tensors = HashMap::new();

    let emb_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 2], &emb_data).unwrap(),
    );

    let serialized = serialize(&tensors, &None).unwrap();
    let p = tmp("tied_emb");
    fs::write(&p, serialized).unwrap();

    let weights = Weights::load_safetensors(&p, &dev).unwrap();
    let _ = fs::remove_file(&p);

    assert!(weights.get("token_embd.weight").is_ok());
    assert!(weights.get("output.weight").is_ok());
    let out = weights.get("output.weight").unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(out, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}

#[test]
fn fp8_e5m2_dequantization_lut_correctness() {
    use runtimed_model::weights::dequantize_fp8_e5m2;
    // 0x00 is +0.0, 0x80 is -0.0
    let res = dequantize_fp8_e5m2(&[0x00, 0x80]);
    assert_eq!(res[0], 0.0);
    assert_eq!(res[1], -0.0);

    // 0x3c is 1.0 (sign 0, exp 15 (0b01111), mant 0)
    let res = dequantize_fp8_e5m2(&[0x3c]);
    assert_eq!(res[0], 1.0);

    // 0x7c is +infinity (sign 0, exp 31, mant 0)
    let res = dequantize_fp8_e5m2(&[0x7c]);
    assert!(res[0].is_infinite() && res[0].is_sign_positive());
}
