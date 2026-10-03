//! LoRA adapter proofs: file validation plus live fusion.
//!
//! A tiny GGUF writer forges synthetic adapters (no fixtures needed).
//! Format checks run ungated; the fusion behavior test needs
//! `SYNTROP_TEST_GGUF_Q4K` (E2B weights to fuse into).

use candle_core::Device;
use runtimed_model::LoraAdapter;
use std::path::{Path, PathBuf};

fn w_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn w_meta_str(out: &mut Vec<u8>, k: &str, v: &str) {
    w_str(out, k);
    out.extend_from_slice(&8u32.to_le_bytes());
    w_str(out, v);
}

fn w_meta_f32(out: &mut Vec<u8>, k: &str, v: f32) {
    w_str(out, k);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

/// Forge an adapter GGUF: `pairs` = (base name, rank, in, out, fill).
/// `arch`/`alpha` go to metadata; `drop_b` omits B tensors (unpaired).
fn forge(path: &Path, arch: &str, alpha: f32, pairs: &[(&str, usize, usize, usize, f32)], drop_b: bool) {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    let n_tensors = pairs.len() * if drop_b { 1 } else { 2 };
    head.extend_from_slice(&(n_tensors as u64).to_le_bytes());
    head.extend_from_slice(&4u64.to_le_bytes());
    w_meta_str(&mut head, "general.type", "adapter");
    w_meta_str(&mut head, "general.architecture", arch);
    w_meta_str(&mut head, "adapter.type", "lora");
    w_meta_f32(&mut head, "adapter.lora.alpha", alpha);
    // Tensor infos first (offsets relative to 32-aligned data start).
    let mut infos = Vec::new();
    let mut data = Vec::new();
    let mut off = 0u64;
    for (base, rank, inp, out, fill) in pairs {
        // A: GGUF dims [in, rank] -> row-major [rank, in].
        let aname = format!("{base}.lora_a");
        w_str(&mut infos, &aname);
        infos.extend_from_slice(&2u32.to_le_bytes());
        infos.extend_from_slice(&(*inp as u64).to_le_bytes());
        infos.extend_from_slice(&(*rank as u64).to_le_bytes());
        infos.extend_from_slice(&0u32.to_le_bytes()); // F32
        infos.extend_from_slice(&off.to_le_bytes());
        let n = rank * inp;
        for i in 0..n {
            data.extend_from_slice(&((i as f32 + 1.0) * fill).to_le_bytes());
        }
        off += (n * 4) as u64;
        if !drop_b {
            let bname = format!("{base}.lora_b");
            w_str(&mut infos, &bname);
            infos.extend_from_slice(&2u32.to_le_bytes());
            infos.extend_from_slice(&(*rank as u64).to_le_bytes());
            infos.extend_from_slice(&(*out as u64).to_le_bytes());
            infos.extend_from_slice(&0u32.to_le_bytes());
            infos.extend_from_slice(&off.to_le_bytes());
            let n = out * rank;
            for i in 0..n {
                data.extend_from_slice(&((i as f32 + 1.0) * fill).to_le_bytes());
            }
            off += (n * 4) as u64;
        }
    }
    head.extend_from_slice(&infos);
    while head.len() % 32 != 0 {
        head.push(0);
    }
    head.extend_from_slice(&data);
    std::fs::write(path, head).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("runtimed-lora-test-{name}.gguf"))
}

fn load_err(p: &std::path::Path) -> String {
    match LoraAdapter::load(p, &Device::Cpu, "gemma4") {
        Ok(_) => panic!("expected load failure"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn rejects_wrong_arch() {
    let p = tmp("arch");
    forge(&p, "qwen2", 1.0, &[("blk.0.attn_q.weight", 1, 4, 4, 0.01)], false);
    let err = load_err(&p);
    assert!(err.contains("arch mismatch"), "{err}");
    std::fs::remove_file(&p).unwrap();
}

#[test]
fn rejects_unpaired_tensors() {
    let p = tmp("unpaired");
    forge(&p, "gemma4", 1.0, &[("blk.0.attn_q.weight", 1, 4, 4, 0.01)], true);
    let err = load_err(&p);
    assert!(err.contains("unpaired"), "{err}");
    std::fs::remove_file(&p).unwrap();
}

#[test]
fn rejects_embedding_lora() {
    let p = tmp("emb");
    forge(&p, "gemma4", 1.0, &[("token_embd.weight", 1, 4, 4, 0.01)], false);
    let err = load_err(&p);
    assert!(err.contains("embedding"), "{err}");
    std::fs::remove_file(&p).unwrap();
}

#[test]
fn fusion_shifts_logits_deterministically() {
    let gguf = match std::env::var("SYNTROP_TEST_GGUF_Q4K") {
        Ok(v) if Path::new(&v).exists() => PathBuf::from(v),
        _ => {
            eprintln!("skip: SYNTROP_TEST_GGUF_Q4K not set");
            return;
        }
    };
    let dev = Device::Cpu;
    let (mut model, _) = runtimed_model::Session::load(&gguf, &dev).unwrap();
    // E2B shapes: attn_q [1536, 1536], ffn_down [1536, 4096]? read live.
    let prompt = vec![2u32, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107];
    model.reset();
    let base = model.forward(&prompt, 0).unwrap();
    let base_row = runtimed_model::decode::generate::last_row(&base).unwrap().to_vec1::<f32>().unwrap();
    // Rank-2 adapter on two linears (E2B row-major: q [2048,1536], down [1536,6144]).
    let p = tmp("fuse");
    forge(
        &p,
        "gemma4",
        2.0,
        &[
            ("blk.0.attn_q.weight", 2, 1536, 2048, 1e-5),
            ("blk.0.ffn_down.weight", 2, 6144, 1536, 1e-5),
        ],
        false,
    );
    let adapter = LoraAdapter::load(&p, &Device::Cpu, "gemma4").unwrap();
    let fused = model.fuse_lora(&adapter).unwrap();
    assert_eq!(fused.len(), 2);
    std::fs::remove_file(&p).unwrap();
    model.reset();
    let lora = model.forward(&prompt, 0).unwrap();
    let lora_row = runtimed_model::decode::generate::last_row(&lora).unwrap().to_vec1::<f32>().unwrap();
    let maxdiff = base_row.iter().zip(lora_row.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(maxdiff > 1e-6, "fusion had no effect (maxdiff {maxdiff})");
    assert!(maxdiff < 5.0, "fusion exploded (maxdiff {maxdiff})");
    assert!(lora_row.iter().all(|x| x.is_finite()), "non-finite logits");
    // Determinism: same weights, same prompt -> identical row.
    model.reset();
    let again = model.forward(&prompt, 0).unwrap();
    let again_row = runtimed_model::decode::generate::last_row(&again).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(lora_row, again_row);
}
