//! Weight store: GGUF and native Safetensors weights decoded to Candle tensors.

mod embed_layer;
pub mod fp8_weight;
mod linear_layer;
mod load_gguf;
mod load_safetensors;
mod name_map;
mod store;

pub use fp8_weight::Fp8Weight;
pub use load_safetensors::{
    dequantize_fp8_e4m3, dequantize_fp8_e5m2, f32_to_fp8_e4m3, FP8_E4M3_LUT, FP8_E5M2_LUT,
};
pub use name_map::map_hf_tensor_name;
pub use store::Weights;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn weights_device_check() {
        assert!(Weights::ensure_current(&Device::Cpu).is_ok());
    }
}
