//! Online block quantizer converting K and V chunks to FP8 / INT8.
//!
//! Keeps initial attention sink tokens (< 16) in FP16 to avoid numerical drift.

use super::block::CachePrecision;
use crate::error::Result;
use candle_core::{DType, Device, Tensor};

pub const SINK_TOKENS_COUNT: usize = 16;

/// Quantize a 4D tensor (e.g. `[batch, heads, tokens, dim]`) to FP8 (F8E4M3) with per-head scale vector.
pub fn quantize_fp8(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let dev = t.device().clone();
    let dims = t.dims();
    let (b, h, s, d) = match dims.len() {
        4 => (dims[0], dims[1], dims[2], dims[3]),
        _ => (1, 1, dims.iter().product::<usize>(), 1),
    };

    let t_f32 = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let raw_f32 = t_f32.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let stride_h = s * d;
    let mut scales = Vec::with_capacity(h);

    for head_idx in 0..h {
        let mut max_abs = 0.0f32;
        for batch_idx in 0..b {
            let offset = (batch_idx * h + head_idx) * stride_h;
            for &val in &raw_f32[offset..offset + stride_h] {
                let a = val.abs();
                if a > max_abs { max_abs = a; }
            }
        }
        let scale = (max_abs / 448.0).max(1e-8);
        scales.push(scale);
    }

    let mut fp8_bytes = Vec::with_capacity(raw_f32.len());
    for batch_idx in 0..b {
        for (head_idx, &s_val) in scales.iter().enumerate().take(h) {
            let offset = (batch_idx * h + head_idx) * stride_h;
            for &val in &raw_f32[offset..offset + stride_h] {
                fp8_bytes.push(crate::weights::f32_to_fp8_e4m3(val / s_val));
            }
        }
    }

    let scale_tensor = Tensor::from_vec(scales, (1, h, 1, 1), &dev)?;
    let fp8_tensor = Tensor::from_raw_buffer(&fp8_bytes, DType::F8E4M3, dims, &dev)?;
    Ok((fp8_tensor, scale_tensor))
}

/// Dequantize an FP8 (F8E4M3) tensor scaled by per-head scale vector to target precision.
pub fn dequantize_fp8(t: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let dev = t.device().clone();
    let unscaled = match t.to_dtype(DType::F32) {
        Ok(f) => f,
        Err(_) => t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_device(&dev)?,
    };
    let scale_dev = scale.to_dtype(DType::F32)?.to_device(&dev)?;
    let scaled = unscaled.contiguous()?.broadcast_mul(&scale_dev)?;
    if target_dtype == DType::F32 { Ok(scaled) } else { Ok(scaled.to_dtype(target_dtype)?) }
}

/// Quantize a 4D tensor to INT8 with per-head scale vector.
pub fn quantize_int8(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let dev = t.device().clone();
    let dims = t.dims();
    let (b, h, s, d) = match dims.len() {
        4 => (dims[0], dims[1], dims[2], dims[3]),
        _ => (1, 1, dims.iter().product::<usize>(), 1),
    };

    let t_f32 = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let raw_f32 = t_f32.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let stride_h = s * d;
    let mut scales = Vec::with_capacity(h);

    for head_idx in 0..h {
        let mut max_abs = 0.0f32;
        for batch_idx in 0..b {
            let offset = (batch_idx * h + head_idx) * stride_h;
            for &val in &raw_f32[offset..offset + stride_h] {
                let a = val.abs();
                if a > max_abs { max_abs = a; }
            }
        }
        let scale = (max_abs / 127.0).max(1e-8);
        scales.push(scale);
    }

    let mut int8_bytes = Vec::with_capacity(raw_f32.len());
    for batch_idx in 0..b {
        for (head_idx, &s_val) in scales.iter().enumerate().take(h) {
            let offset = (batch_idx * h + head_idx) * stride_h;
            for &val in &raw_f32[offset..offset + stride_h] {
                let scaled = (val / s_val).round().clamp(-128.0, 127.0) as i8;
                int8_bytes.push(scaled as u8);
            }
        }
    }

    let scale_tensor = Tensor::from_vec(scales, (1, h, 1, 1), &dev)?;
    let int8_tensor = Tensor::from_raw_buffer(&int8_bytes, DType::U8, dims, &dev)?;
    Ok((int8_tensor, scale_tensor))
}

/// Dequantize an INT8 tensor scaled by per-head scale vector to target precision.
pub fn dequantize_int8(t: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let dev = t.device().clone();
    let raw_u8 = t.to_device(&Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<u8>()?;
    let floats: Vec<f32> = raw_u8.into_iter().map(|b| (b as i8) as f32).collect();
    let unscaled = Tensor::from_vec(floats, t.dims(), &Device::Cpu)?.to_device(&dev)?;
    let scale_dev = scale.to_dtype(DType::F32)?.to_device(&dev)?;
    let scaled = unscaled.contiguous()?.broadcast_mul(&scale_dev)?;
    if target_dtype == DType::F32 { Ok(scaled) } else { Ok(scaled.to_dtype(target_dtype)?) }
}

/// Selectively quantize KV chunk based on sequence offset: keeps initial sink tokens (< 16) in FP16.
pub fn quantize_block_kv(
    token_offset: usize,
    k: &Tensor,
    v: &Tensor,
    requested_precision: CachePrecision,
) -> Result<(Tensor, Tensor, Tensor, Tensor, CachePrecision)> {
    if token_offset < SINK_TOKENS_COUNT || requested_precision == CachePrecision::Fp16 {
        let k_f16 = if k.dtype() == DType::F16 { k.clone() } else { k.to_dtype(DType::F16)? };
        let v_f16 = if v.dtype() == DType::F16 { v.clone() } else { v.to_dtype(DType::F16)? };
        let one_k = Tensor::ones(1, DType::F32, k.device())?;
        let one_v = Tensor::ones(1, DType::F32, v.device())?;
        return Ok((k_f16, v_f16, one_k, one_v, CachePrecision::Fp16));
    }

    match requested_precision {
        CachePrecision::Int8 => {
            let (k_q, k_s) = quantize_int8(k)?;
            let (v_q, v_s) = quantize_int8(v)?;
            Ok((k_q, v_q, k_s, v_s, CachePrecision::Int8))
        }
        _ => {
            let (k_q, k_s) = quantize_fp8(k)?;
            let (v_q, v_s) = quantize_fp8(v)?;
            Ok((k_q, v_q, k_s, v_s, CachePrecision::Fp8))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_quantization_and_dequantization_roundtrip() {
        let dev = Device::Cpu;
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let t = Tensor::from_vec(data.clone(), (1, 2, 2, 16), &dev).unwrap();

        let (q, scale) = quantize_fp8(&t).unwrap();
        assert_eq!(q.dtype(), DType::F8E4M3);
        assert_eq!(scale.dims(), &[1, 2, 1, 1]);

        let deq = dequantize_fp8(&q, &scale, DType::F32).unwrap();
        let deq_vec = deq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (orig, recon) in data.iter().zip(deq_vec.iter()) {
            assert!((orig - recon).abs() < 0.2, "orig={orig}, recon={recon}");
        }
    }

    #[test]
    fn test_attention_sink_preserves_fp16() {
        let dev = Device::Cpu;
        let k = Tensor::zeros((1, 2, 4, 16), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 4, 16), DType::F32, &dev).unwrap();

        // token_offset = 0 (< 16) must stay FP16
        let (kq, vq, _, _, prec) = quantize_block_kv(0, &k, &v, CachePrecision::Fp8).unwrap();
        assert_eq!(prec, CachePrecision::Fp16);
        assert_eq!(kq.dtype(), DType::F16);
        assert_eq!(vq.dtype(), DType::F16);

        // token_offset = 16 (>= 16) must quantize to FP8
        let (kq2, vq2, _, _, prec2) = quantize_block_kv(16, &k, &v, CachePrecision::Fp8).unwrap();
        assert_eq!(prec2, CachePrecision::Fp8);
        assert_eq!(kq2.dtype(), DType::F8E4M3);
        assert_eq!(vq2.dtype(), DType::F8E4M3);
    }

    #[test]
    fn test_int8_quantization_and_dequantization_roundtrip() {
        let dev = Device::Cpu;
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let t = Tensor::from_vec(data.clone(), (1, 2, 2, 16), &dev).unwrap();

        let (q, scale) = quantize_int8(&t).unwrap();
        assert_eq!(q.dtype(), DType::U8);
        assert_eq!(scale.dims(), &[1, 2, 1, 1]);

        let deq = dequantize_int8(&q, &scale, DType::F32).unwrap();
        let deq_vec = deq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (orig, recon) in data.iter().zip(deq_vec.iter()) {
            assert!((orig - recon).abs() < 0.2, "orig={orig}, recon={recon}");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_fp8_quantization_and_dequantization() {
        if let Ok(dev) = Device::new_cuda(0) {
            let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
            let t = Tensor::from_vec(data.clone(), (1, 2, 2, 16), &dev).unwrap();
            let (q, scale) = quantize_fp8(&t).unwrap();
            assert_eq!(q.dtype(), DType::F8E4M3);
            let deq = dequantize_fp8(&q, &scale, DType::F32).unwrap();
            let deq_vec = deq.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (orig, recon) in data.iter().zip(deq_vec.iter()) {
                assert!((orig - recon).abs() < 0.2, "orig={orig}, recon={recon}");
            }
        }
    }
}
