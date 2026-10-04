//! Ada Lovelace Native FP8 (FP8_E4M3) Hardware Tensor Core GEMM via cuBLASLt.

use crate::error::Result;
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")]
use crate::error::ModelError;

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::cublaslt::{result, sys};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::CudaStorageSlice;
#[cfg(feature = "cuda")]
use candle_core::Storage;

#[cfg(feature = "cuda")]
struct Handle(sys::cublasLtHandle_t);
#[cfg(feature = "cuda")]
impl Drop for Handle { fn drop(&mut self) { unsafe { let _ = result::destroy_handle(self.0); } } }

#[cfg(feature = "cuda")]
struct MatmulDesc(sys::cublasLtMatmulDesc_t);
#[cfg(feature = "cuda")]
impl Drop for MatmulDesc { fn drop(&mut self) { unsafe { let _ = result::destroy_matmul_desc(self.0); } } }

#[cfg(feature = "cuda")]
struct Layout(sys::cublasLtMatrixLayout_t);
#[cfg(feature = "cuda")]
impl Drop for Layout { fn drop(&mut self) { unsafe { let _ = result::destroy_matrix_layout(self.0); } } }

#[cfg(feature = "cuda")]
struct Pref(sys::cublasLtMatmulPreference_t);
#[cfg(feature = "cuda")]
impl Drop for Pref { fn drop(&mut self) { unsafe { let _ = result::destroy_matmul_pref(self.0); } } }

#[cfg(feature = "cuda")]
fn get_device_ptr(t: &Tensor, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let off = layout.start_offset();
    match &*storage {
        Storage::Cuda(c) => match &c.slice {
            CudaStorageSlice::F8E4M3(s) => Ok(s.slice(off..).device_ptr(stream).0),
            CudaStorageSlice::F16(s) => Ok(s.slice(off..).device_ptr(stream).0),
            CudaStorageSlice::F32(s) => Ok(s.slice(off..).device_ptr(stream).0),
            CudaStorageSlice::BF16(s) => Ok(s.slice(off..).device_ptr(stream).0),
            _ => Err(ModelError::Config("unsupported CUDA storage slice for FP8 GEMM".into())),
        },
        _ => Err(ModelError::Config("tensor not on CUDA device".into())),
    }
}

/// Fallback FP8 GEMM using dequantized matrix multiplication.
pub fn cpu_fp8_gemm_fallback(x: &Tensor, w: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dev = x.device().clone();
    let x_cpu = x.to_device(&Device::Cpu)?;
    let w_cpu = w.to_device(&Device::Cpu)?;
    let s_cpu = scale.to_device(&Device::Cpu)?;
    let w_f32 = w_cpu.to_dtype(DType::F32)?;
    let s_f32 = s_cpu.to_dtype(DType::F32)?;
    let s_f32 = if s_f32.dims().len() == 1 { s_f32.reshape((s_f32.dim(0)?, 1))? } else { s_f32 };
    let w_scaled = w_f32.broadcast_mul(&s_f32)?;
    let x_f32 = x_cpu.to_dtype(DType::F32)?;
    let dims = x_cpu.dims().to_vec();
    let k = *dims.last().unwrap_or(&0);
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let y = x_f32.reshape((rows, k))?.matmul(&w_scaled.t()?)?;
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(w.dim(0)?);
    let y = y.reshape(out_shape)?;
    let y = if y.dtype() != x.dtype() && x.dtype() != DType::F8E4M3 { y.to_dtype(x.dtype())? } else { y };
    Ok(y.to_device(&dev)?)
}

/// Execute FP8 GEMM: computes `x @ (w * scale)^T` using native Ada Lovelace Tensor Cores.
pub fn fp8_gemm(x: &Tensor, w: &Tensor, scale: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = x.device() {
        if w.device().same_device(&Device::Cuda(dev.clone())) {
            return fp8_gemm_cuda(x, w, scale, dev);
        }
    }
    cpu_fp8_gemm_fallback(x, w, scale)
}

#[cfg(feature = "cuda")]
fn fp8_gemm_cuda(x: &Tensor, w: &Tensor, scale: &Tensor, dev: &candle_core::CudaDevice) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let (in_dim, out_dim) = (*dims.last().unwrap_or(&0), w.dim(0)?);
    let rows: usize = dims[..dims.len() - 1].iter().product();

    let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
    let x_fp8 = if x_c.dtype() == DType::F8E4M3 {
        x_c
    } else {
        let x_f32 = if x_c.dtype() == DType::F32 { x_c.clone() } else { x_c.to_dtype(DType::F32)? };
        let slice = x_f32.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let bytes: Vec<u8> = slice.into_iter().map(crate::weights::f32_to_fp8_e4m3).collect();
        Tensor::from_raw_buffer(&bytes, DType::F8E4M3, x_c.dims(), &Device::Cuda(dev.clone()))?
    };
    let w_c = if w.is_contiguous() { w.clone() } else { w.contiguous()? };
    let s_c = if scale.is_contiguous() { scale.clone() } else { scale.contiguous()? };
    let s_f32 = if s_c.dtype() == DType::F32 { s_c.clone() } else { s_c.to_dtype(DType::F32)? };

    let out_dt = if x.dtype() == DType::F16 || x.dtype() == DType::F8E4M3 { DType::F16 } else { DType::F32 };
    let out = Tensor::zeros((rows, out_dim), out_dt, &Device::Cuda(dev.clone()))?;
    let stream = dev.cuda_stream();

    let one_t = Tensor::ones(1, DType::F32, &Device::Cuda(dev.clone()))?;
    let is_per_channel = s_f32.elem_count() > 1;

    let handle = result::create_handle().map(Handle).map_err(|e| ModelError::Config(e.to_string()))?;
    let desc = result::create_matmul_desc(
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cudaDataType_t::CUDA_R_32F,
    ).map(MatmulDesc).map_err(|e| ModelError::Config(e.to_string()))?;

    let transa = 1i32;
    unsafe {
        result::set_matmul_desc_attribute(
            desc.0,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            (&transa) as *const _ as *const _,
            std::mem::size_of::<i32>(),
        ).map_err(|e| ModelError::Config(e.to_string()))?;

        let a_scale_ptr = if is_per_channel {
            get_device_ptr(&one_t, &stream)?
        } else {
            get_device_ptr(&s_f32, &stream)?
        };
        result::set_matmul_desc_attribute(
            desc.0,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            (&a_scale_ptr) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        ).map_err(|e| ModelError::Config(e.to_string()))?;
    }

    let a_ly = result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_8F_E4M3, in_dim as u64, out_dim as u64, in_dim as i64)
        .map(Layout).map_err(|e| ModelError::Config(e.to_string()))?;
    let b_ly = result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_8F_E4M3, in_dim as u64, rows as u64, in_dim as i64)
        .map(Layout).map_err(|e| ModelError::Config(e.to_string()))?;
    let c_dt = if out_dt == DType::F16 { sys::cudaDataType_t::CUDA_R_16F } else { sys::cudaDataType_t::CUDA_R_32F };
    let c_ly = result::create_matrix_layout(c_dt, out_dim as u64, rows as u64, out_dim as i64)
        .map(Layout).map_err(|e| ModelError::Config(e.to_string()))?;

    let pref = result::create_matmul_pref().map(Pref).map_err(|e| ModelError::Config(e.to_string()))?;
    let ws_size = 4 * 1024 * 1024;
    unsafe {
        result::set_matmul_pref_attribute(
            pref.0,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&ws_size) as *const _ as *const _,
            std::mem::size_of::<usize>(),
        ).map_err(|e| ModelError::Config(e.to_string()))?;
    }

    let heuristic = match unsafe { result::get_matmul_algo_heuristic(handle.0, desc.0, a_ly.0, b_ly.0, c_ly.0, c_ly.0, pref.0) } {
        Ok(h) => h,
        Err(_) => return cpu_fp8_gemm_fallback(x, w, scale),
    };

    let ws = unsafe { stream.alloc::<u8>(ws_size).map_err(|e| ModelError::Config(e.to_string()))? };
    let (w_ptr, _) = ws.device_ptr(&stream);
    let (a_ptr, b_ptr, d_ptr) = (get_device_ptr(&w_c, &stream)?, get_device_ptr(&x_fp8, &stream)?, get_device_ptr(&out, &stream)?);
    let (alpha, beta) = (1.0f32, 0.0f32);
    let cu_stream = stream.cu_stream() as *mut sys::CUstream_st;

    unsafe {
        result::matmul(
            handle.0, desc.0, (&alpha) as *const _ as *const _, (&beta) as *const _ as *const _,
            a_ptr as *const _, a_ly.0, b_ptr as *const _, b_ly.0,
            d_ptr as *const _, c_ly.0, d_ptr as *mut _, c_ly.0,
            &heuristic.algo, w_ptr as *mut _, ws_size, cu_stream,
        ).map_err(|e| ModelError::Config(format!("cublasLt matmul: {e}")))?;
    }

    let out = if s_c.elem_count() > 1 {
        let s_flat = if s_c.dims().len() > 1 { s_c.flatten_all()? } else { s_c.clone() };
        let s_dt = if s_flat.dtype() == out.dtype() { s_flat } else { s_flat.to_dtype(out.dtype())? };
        out.broadcast_mul(&s_dt)?
    } else {
        out
    };

    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(out_dim);
    Ok(out.reshape(out_shape)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fp8_gemm_matches_reference() {
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let w = Tensor::new(&[[1.0f32, 0.0], [0.0, 2.0]], &Device::Cpu).unwrap();
        let scale = Tensor::new(&[2.0f32, 0.5f32], &Device::Cpu).unwrap();

        let y = fp8_gemm(&x, &w, &scale).unwrap();
        assert_eq!(y.dims(), &[2, 2]);
        let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(vals, vec![2.0, 2.0, 6.0, 4.0]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_fp8_gemm_runs_or_skips() {
        if let Ok(dev) = Device::new_cuda(0) {
            let m = 16;
            let n = 16;
            let k = 16;
            let x_bytes = vec![0x38u8; m * k];
            let w_bytes = vec![0x38u8; n * k];
            let x = Tensor::from_raw_buffer(&x_bytes, DType::F8E4M3, &[m, k], &dev).unwrap();
            let w = Tensor::from_raw_buffer(&w_bytes, DType::F8E4M3, &[n, k], &dev).unwrap();
            let s_vec: Vec<f32> = (1..=16).map(|i| i as f32).collect();
            let scale = Tensor::from_vec(s_vec, n, &dev).unwrap();
            let y = fp8_gemm(&x, &w, &scale).unwrap();
            assert_eq!(y.dims(), &[m, n]);
            let ref_y = cpu_fp8_gemm_fallback(&x, &w, &scale).unwrap();
            let y_vals = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let ref_vals = ref_y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (y_val, ref_val) in y_vals.iter().zip(ref_vals.iter()) {
                assert!((y_val - ref_val).abs() < 1e-1, "mismatch: got {y_val}, expected {ref_val}");
            }
        }
    }
}
