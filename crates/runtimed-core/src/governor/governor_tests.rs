use super::device_headroom::{DeviceHeadroom, ModelKvSpec};
use super::gang_governor::{GangLinkType, MultiGpuHeadroomGovernor};
use crate::engine::ReasoningEffort;

fn test_kv_spec() -> ModelKvSpec {
    ModelKvSpec {
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        bytes_per_element: 2, // FP16
    }
}

#[test]
fn test_cpu_offload_veto() {
    let spec = test_kv_spec();
    let gpu_stage = DeviceHeadroom::calculate(
        "gpu0".into(),
        false,
        24,
        16 * 1024 * 1024 * 1024,
        &spec,
    );
    let cpu_stage = DeviceHeadroom::calculate(
        "cpu-host".into(),
        true,
        8,
        32 * 1024 * 1024 * 1024,
        &spec,
    );

    let governor = MultiGpuHeadroomGovernor::new(
        vec![gpu_stage, cpu_stage],
        GangLinkType::NVLink,
        0.0,
    );

    let res = governor.evaluate();
    assert!(res.cpu_veto);
    assert_eq!(res.effort, ReasoningEffort::None);
    assert_eq!(governor.resolve_effort(true), ReasoningEffort::None);
}

#[test]
fn test_asymmetric_vram_weakest_link() {
    let spec = test_kv_spec();
    // GPU 0: 24 GB, 16 layers
    let gpu0 = DeviceHeadroom::calculate(
        "gpu0".into(),
        false,
        16,
        24 * 1024 * 1024 * 1024,
        &spec,
    );
    // GPU 1: 4 GB, 16 layers (tight bottleneck)
    let gpu1 = DeviceHeadroom::calculate(
        "gpu1".into(),
        false,
        16,
        2 * 1024 * 1024 * 1024,
        &spec,
    );

    let governor = MultiGpuHeadroomGovernor::new(
        vec![gpu0.clone(), gpu1.clone()],
        GangLinkType::NVLink,
        0.0,
    );

    let res = governor.evaluate();
    assert!(!res.cpu_veto);
    assert_eq!(res.weakest_link_tokens, gpu1.max_tokens);
    assert!(res.weakest_link_tokens < gpu0.max_tokens);
}

#[test]
fn test_pcie_vs_nvlink_link_factors() {
    let spec = test_kv_spec();
    let gpu0 = DeviceHeadroom::calculate(
        "gpu0".into(),
        false,
        16,
        16 * 1024 * 1024 * 1024,
        &spec,
    );
    let gpu1 = DeviceHeadroom::calculate(
        "gpu1".into(),
        false,
        16,
        16 * 1024 * 1024 * 1024,
        &spec,
    );

    let gov_nvlink = MultiGpuHeadroomGovernor::new(
        vec![gpu0.clone(), gpu1.clone()],
        GangLinkType::NVLink,
        0.0,
    );
    let gov_pcie = MultiGpuHeadroomGovernor::new(
        vec![gpu0, gpu1],
        GangLinkType::PCIe,
        0.0,
    );

    let res_nv = gov_nvlink.evaluate();
    let res_pcie = gov_pcie.evaluate();

    assert_eq!(res_nv.beta_link, 1.0);
    assert_eq!(res_pcie.beta_link, 0.85);
    assert!(res_nv.effective_tokens > res_pcie.effective_tokens);
}

#[test]
fn test_psi_factor_nan_and_high_pressure_safety() {
    let spec = test_kv_spec();
    let gpu = DeviceHeadroom::calculate(
        "gpu0".into(),
        false,
        32,
        16 * 1024 * 1024 * 1024,
        &spec,
    );

    let gov_nan = MultiGpuHeadroomGovernor::new(vec![gpu.clone()], GangLinkType::NVLink, f32::NAN);
    let res_nan = gov_nan.evaluate();
    assert_eq!(res_nan.phi_psi, 1.0); // clamped to 0.0 PSI -> factor 1.0

    let gov_crit = MultiGpuHeadroomGovernor::new(vec![gpu], GangLinkType::NVLink, 75.0);
    let res_crit = gov_crit.evaluate();
    assert_eq!(res_crit.phi_psi, 0.0);
    assert_eq!(res_crit.effort, ReasoningEffort::None);
}
