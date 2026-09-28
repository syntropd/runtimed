//! Storage-dtype proofs: F32 resident on CPU, F16 on CUDA, same greedy pick.
//!
//! Gated on `SYNTROP_TEST_GGUF_Q4K` (E2B weights). The CUDA leg also needs
//! a GPU; without one (`Device::new_cuda(0)` fails) it skips honestly.

use candle_core::Device;
use runtimed_gguf::GgufFile;
use runtimed_model::Session;
use std::path::PathBuf;

fn gated() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SYNTROP_TEST_GGUF_Q4K").ok()?);
    if !path.exists() {
        eprintln!("skip: SYNTROP_TEST_GGUF_Q4K not set or missing");
        return None;
    }
    Some(path)
}

fn param_count(path: &PathBuf) -> usize {
    GgufFile::open(path)
        .expect("must parse")
        .tensors
        .iter()
        .map(|t| t.n_elements)
        .sum()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap()
}

#[test]
fn cpu_storage_is_f32_and_counted() {
    let Some(path) = gated() else { return };
    assert!(runtimed_model::Weights::ensure_current(&Device::Cpu).is_ok());
    let (model, _) = Session::load(&path, &Device::Cpu).unwrap();
    assert_eq!(model.resident_bytes(), param_count(&path) * 4);
}

#[test]
fn cuda_storage_is_f16_with_matching_greedy() {
    let Some(path) = gated() else { return };
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    assert!(runtimed_model::Weights::ensure_current(&dev).is_ok());
    let (mut gpu, _) = Session::load(&path, &dev).unwrap();
    assert_eq!(gpu.resident_bytes(), param_count(&path) * 2);
    // Greedy parity: F16 storage, F32 compute, same top-1 as CPU.
    let prompt = vec![2u32, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107];
    let (mut cpu, _) = Session::load(&path, &Device::Cpu).unwrap();
    gpu.reset();
    let grow = runtimed_model::decode::generate::last_row(&gpu.forward(&prompt, 0).unwrap())
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    cpu.reset();
    let crow = runtimed_model::decode::generate::last_row(&cpu.forward(&prompt, 0).unwrap())
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(argmax(&grow), argmax(&crow));
    assert!(grow.iter().all(|x| x.is_finite()));
}
