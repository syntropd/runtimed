//! ArchConfig proofs from a forged metadata-only GGUF.
//!
//! Same tiny-writer trick as the LoRA tests: no fixtures needed. The
//! forged file carries the `qwen2.*` hyper-parameters plus two empty
//! F32 tensors (`token_embd.weight` sets the vocab width).

use runtimed_gguf::GgufFile;
use runtimed_model::{Arch, ArchConfig};
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

fn w_meta_u32(out: &mut Vec<u8>, k: &str, v: u32) {
    w_str(out, k);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

fn w_meta_f32(out: &mut Vec<u8>, k: &str, v: f32) {
    w_str(out, k);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

fn w_tensor(infos: &mut Vec<u8>, data: &mut Vec<u8>, off: &mut u64, name: &str, d0: u64, d1: u64) {
    w_str(infos, name);
    infos.extend_from_slice(&2u32.to_le_bytes());
    infos.extend_from_slice(&d0.to_le_bytes());
    infos.extend_from_slice(&d1.to_le_bytes());
    infos.extend_from_slice(&0u32.to_le_bytes()); // F32
    infos.extend_from_slice(&off.to_le_bytes());
    let n = (d0 * d1) as usize;
    data.extend_from_slice(&vec![0u8; n * 4]);
    *off += (n * 4) as u64;
}

/// Forge a qwen2 GGUF: 8 hyper-parameters plus embedding/output tables.
/// `with_head=false` drops `output.weight` (refused for qwen2).
fn forge(path: &Path, arch: &str, with_head: bool) {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(if with_head { 2u64 } else { 1u64 }).to_le_bytes());
    head.extend_from_slice(&8u64.to_le_bytes());
    w_meta_str(&mut head, "general.architecture", arch);
    w_meta_u32(&mut head, "qwen2.block_count", 2);
    w_meta_u32(&mut head, "qwen2.embedding_length", 8);
    w_meta_u32(&mut head, "qwen2.attention.head_count", 2);
    w_meta_u32(&mut head, "qwen2.attention.head_count_kv", 1);
    w_meta_f32(&mut head, "qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    w_meta_f32(&mut head, "qwen2.rope.freq_base", 1_000_000.0);
    w_meta_u32(&mut head, "qwen2.feed_forward_length", 16);
    let mut infos = Vec::new();
    let mut data = Vec::new();
    let mut off = 0u64;
    w_tensor(&mut infos, &mut data, &mut off, "token_embd.weight", 8, 16);
    if with_head {
        w_tensor(&mut infos, &mut data, &mut off, "output.weight", 8, 16);
    }
    head.extend_from_slice(&infos);
    while head.len() % 32 != 0 {
        head.push(0);
    }
    head.extend_from_slice(&data);
    std::fs::write(path, head).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("runtimed-config-test-{name}.gguf"))
}

#[test]
fn parses_qwen2_config_from_forged_file() {
    let path = tmp("qwen2");
    forge(&path, "qwen2", true);
    let file = GgufFile::open(&path).expect("parse forged");
    let cfg = ArchConfig::parse(&file).expect("config");
    let _ = std::fs::remove_file(&path);
    assert_eq!(cfg.arch, Arch::Qwen2);
    assert_eq!(cfg.n_layer, 2);
    assert_eq!(cfg.hidden, 8);
    assert_eq!(cfg.vocab, 16);
    assert!(!cfg.tie_lm_head);
    assert!(!cfg.has_qkv_bias);
    assert_eq!(cfg.layers.len(), 2);
    assert_eq!(cfg.layers[0].head_dim, 4);
    assert_eq!(cfg.layers[0].ffn, 16);
}

#[test]
fn rejects_unknown_arch() {
    let path = tmp("arch");
    forge(&path, "bert", true);
    let file = GgufFile::open(&path).expect("parse forged");
    let err = ArchConfig::parse(&file).err().expect("must fail").to_string();
    let _ = std::fs::remove_file(&path);
    assert!(err.contains("unsupported architecture"), "{err}");
}

#[test]
fn refuses_qwen2_without_output_weight() {
    let path = tmp("nohead");
    forge(&path, "qwen2", false);
    let file = GgufFile::open(&path).expect("parse forged");
    let err = ArchConfig::parse(&file).err().expect("must fail").to_string();
    let _ = std::fs::remove_file(&path);
    assert!(err.contains("without output.weight"), "{err}");
}
