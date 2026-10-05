//! Intra-node pure Rust Tensor Parallelism for NVLink and PCIe.

pub mod column_linear;
pub mod dual_gpu;
pub mod ring_reduce;
pub mod row_linear;

pub use column_linear::ColumnParallelLinear;
pub use dual_gpu::{DualGpuAttention, DualGpuBlock, DualGpuContext, DualGpuMlp};
pub use ring_reduce::{ring_all_reduce, RingReducer};
pub use row_linear::RowParallelLinear;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_tensor_parallel_linear_equivalence() {
        let full_weight = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let col0 = ColumnParallelLinear::from_full_weight(&full_weight, None, 0, 2).unwrap();
        let col1 = ColumnParallelLinear::from_full_weight(&full_weight, None, 1, 2).unwrap();

        let x = Tensor::new(&[[1.0f32, 1.0]], &Device::Cpu).unwrap();
        let y0 = col0.forward(&x).unwrap();
        let y1 = col1.forward(&x).unwrap();

        let cat_y = Tensor::cat(&[&y0, &y1], 1).unwrap();
        let expected = Tensor::new(&[[3.0f32, 7.0]], &Device::Cpu).unwrap();
        assert_eq!(cat_y.to_vec2::<f32>().unwrap(), expected.to_vec2::<f32>().unwrap());

        let reduced = ring_all_reduce(&[y0.clone(), y0.clone()]).unwrap();
        assert_eq!(reduced.len(), 2);
    }
}
