//! Integration tests for visual generation sampler, memfd rendering, and LoRA fusion.

use candle_core::{DType, Device, Tensor};
use runtimed_model::visual_gen::{
    LoraMatrixPair, VisualComputeLease, VisualGenConfig, VisualGenSampler,
};
use runtimed_model::weights::Weights;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_sampler_renders_valid_png() {
    let sampler = VisualGenSampler::new();
    let lease = VisualComputeLease::new("lease-test-123");
    let png = sampler
        .sample_1step("a futuristic server rack", 64, 64, 1337, &lease)
        .expect("sampling must succeed");

    assert_eq!(&png[1..4], b"PNG");
    let img = image::load_from_memory(&png).expect("valid png decode");
    assert_eq!(img.width(), 64);
    assert_eq!(img.height(), 64);
}

#[test]
fn test_sampler_renders_with_weights_fallback() {
    let dev = Device::Cpu;
    let weights = Arc::new(Weights::from_parts(dev, DType::F32, HashMap::new()));
    let sampler = VisualGenSampler::with_weights(VisualGenConfig::default(), weights);
    let lease = VisualComputeLease::new("lease-test-weights");
    let png = sampler
        .sample_1step("neural generative landscape", 32, 32, 888, &lease)
        .expect("sampling with weights must succeed");
    assert_eq!(&png[1..4], b"PNG");
}

#[test]
fn test_sampler_renders_to_sealed_memfd() {
    let sampler = VisualGenSampler::new();
    let lease = VisualComputeLease::new("lease-456");
    let (_fd, len) = sampler
        .generate_to_sealed_memfd("high throughput engine", 32, 32, 99, &lease)
        .expect("memfd generation must succeed");
    assert!(len > 0);
}

#[test]
fn test_sampler_multistep_and_lora_hooks() {
    let dev = Device::Cpu;
    let mut map = HashMap::new();
    let w0 = Tensor::ones((4, 4), DType::F32, &dev).expect("tensor");
    map.insert("linear.weight".into(), w0);
    let weights = Arc::new(Weights::from_parts(dev.clone(), DType::F32, map));

    let cfg = VisualGenConfig {
        steps: 4,
        ..Default::default()
    };
    let a = Tensor::ones((2, 4), DType::F32, &dev).expect("tensor a");
    let b = Tensor::ones((4, 2), DType::F32, &dev).expect("tensor b");
    let lora = LoraMatrixPair::new("linear.weight", a, b, 1.0).expect("lora");

    let sampler = VisualGenSampler::with_weights(cfg, weights).with_lora(lora);
    let lease = VisualComputeLease::new("active-lease-123");
    let png = sampler
        .sample_1step("futuristic city", 64, 64, 999, &lease)
        .expect("sampling must succeed");
    assert_eq!(&png[1..4], b"PNG");
}

#[test]
fn test_sampler_rejects_empty_prompt_and_bad_dims() {
    let sampler = VisualGenSampler::new();
    let lease = VisualComputeLease::new("lease-active");

    assert!(sampler.sample_1step("   ", 64, 64, 1, &lease).is_err());
    assert!(sampler.sample_1step("prompt", 0, 64, 1, &lease).is_err());
    assert!(sampler.sample_1step("prompt", 5000, 64, 1, &lease).is_err());
}
