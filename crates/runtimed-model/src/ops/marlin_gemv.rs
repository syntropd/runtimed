//! Marlin INT4 GEMV kernel: 128-bit vectorized dequantization and matmul.

use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::{cudarc::driver::DevicePtr, CudaStorageSlice, DeviceId};
#[cfg(feature = "cuda")]
use candle_core::Storage;

/// Packed INT4 weight matrix for Marlin GEMV execution.
#[derive(Clone, Debug)]
pub struct MarlinWeight {
    pub packed: Tensor,
    pub scales: Tensor,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl MarlinWeight {
    pub fn new(packed: Tensor, scales: Tensor, in_dim: usize, out_dim: usize) -> Result<Self> {
        if packed.dtype() != DType::U32 { return Err(ModelError::Config(format!("marlin packed must be U32, got {:?}", packed.dtype()))); }
        let s_dims = scales.dims();
        if s_dims != [out_dim] && s_dims != [out_dim, 1] { return Err(ModelError::Shape { name: "marlin_scales".into(), expected: vec![out_dim], got: s_dims.to_vec() }); }
        if !packed.device().same_device(scales.device()) { return Err(ModelError::Config("packed and scales must be on the same device".into())); }
        let k_chunks = in_dim.div_ceil(32);
        if packed.elem_count() < k_chunks * out_dim * 4 { return Err(ModelError::Shape { name: "marlin_packed".into(), expected: vec![k_chunks, out_dim, 4], got: packed.dims().to_vec() }); }
        Ok(Self { packed, scales, in_dim, out_dim })
    }

    pub fn device(&self) -> &Device { self.packed.device() }

    pub fn resident_bytes(&self) -> usize {
        self.packed.elem_count() * self.packed.dtype().size_in_bytes() + self.scales.elem_count() * self.scales.dtype().size_in_bytes()
    }

    pub fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self { packed: self.packed.to_device(dev)?, scales: self.scales.to_device(dev)?, in_dim: self.in_dim, out_dim: self.out_dim })
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        let dev = self.packed.device().clone();
        let (p_cpu, s_cpu) = (self.packed.to_device(&Device::Cpu)?, self.scales.to_device(&Device::Cpu)?);
        let (p_vec, s_f32) = (p_cpu.flatten_all()?.to_vec1::<u32>()?, s_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
        let k_chunks = self.in_dim.div_ceil(32);
        let mut out = vec![0.0f32; self.out_dim * self.in_dim];

        for c in 0..k_chunks {
            for n in 0..self.out_dim {
                let (base, s) = ((c * self.out_dim + n) * 4, s_f32[n]);
                for (w_off, &w_val) in p_vec[base..base + 4].iter().enumerate() {
                    for nib in 0..8 {
                        let k = c * 32 + w_off * 8 + nib;
                        if k < self.in_dim { out[n * self.in_dim + k] = (((w_val >> (nib * 4)) & 0x0F) as i32 - 8) as f32 * s; }
                    }
                }
            }
        }
        Ok(Tensor::from_vec(out, (self.out_dim, self.in_dim), &Device::Cpu)?.to_device(&dev)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { marlin_gemv(x, self) }
}

/// Repacks a dense weight matrix into 128-bit Marlin INT4 tiles.
pub fn repack_marlin_tiles(weight: &Tensor, scales: &Tensor) -> Result<MarlinWeight> {
    let dev = weight.device().clone();
    let (w_cpu, s_cpu) = (weight.to_device(&Device::Cpu)?, scales.to_device(&Device::Cpu)?);
    let w_dims = w_cpu.dims();
    if w_dims.len() != 2 { return Err(ModelError::Config(format!("weight must be 2D, got {w_dims:?}"))); }
    let (out_dim, in_dim) = (w_dims[0], w_dims[1]);
    let s_f32 = s_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if s_f32.len() != out_dim { return Err(ModelError::Shape { name: "scales".into(), expected: vec![out_dim], got: vec![s_f32.len()] }); }
    let w_data = w_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let k_chunks = in_dim.div_ceil(32);
    let mut packed_vec = vec![0u32; k_chunks * out_dim * 4];

    for c in 0..k_chunks {
        for n in 0..out_dim {
            let inv_s = if s_f32[n].abs() > 1e-12 { 1.0 / s_f32[n] } else { 1.0 };
            let mut words = [0u32; 4];
            for (w_idx, word_slot) in words.iter_mut().enumerate() {
                let mut word = 0u32;
                for nib in 0..8 {
                    let k = c * 32 + w_idx * 8 + nib;
                    let q = if k < in_dim { ((w_data[n * in_dim + k] * inv_s).round() + 8.0).clamp(0.0, 15.0) as u32 } else { 8u32 };
                    word |= (q & 0x0F) << (nib * 4);
                }
                *word_slot = word;
            }
            let base = (c * out_dim + n) * 4;
            packed_vec[base..base + 4].copy_from_slice(&words);
        }
    }
    let packed = Tensor::from_vec(packed_vec, (k_chunks, out_dim, 4), &Device::Cpu)?.to_device(&dev)?;
    Ok(MarlinWeight { packed, scales: scales.clone(), in_dim, out_dim })
}

/// Fallback CPU dequantized matrix multiplication for Marlin weights.
pub fn cpu_marlin_gemv_fallback(x: &Tensor, w: &MarlinWeight) -> Result<Tensor> {
    let dev = x.device().clone();
    let (x_cpu, orig_dt) = (x.to_device(&Device::Cpu)?, x.dtype());
    let dims = x.dims().to_vec();
    let in_dim = *dims.last().unwrap_or(&0);
    if in_dim != w.in_dim { return Err(ModelError::Shape { name: "marlin_gemv".into(), expected: vec![w.in_dim], got: dims }); }
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(w.out_dim);
    if rows == 0 || w.out_dim == 0 { return Ok(Tensor::zeros(out_shape, orig_dt, &dev)?); }
    let (x_f32, w_deq) = (x_cpu.to_dtype(DType::F32)?, w.dequantize()?.to_device(&Device::Cpu)?);
    let y = x_f32.reshape((rows, in_dim))?.matmul(&w_deq.t()?)?.reshape(out_shape)?;
    let y = if orig_dt != DType::F32 { y.to_dtype(orig_dt)? } else { y };
    Ok(y.to_device(&dev)?)
}

/// Execute Marlin INT4 GEMV: evaluates `x @ (w * scale)^T`.
pub fn marlin_gemv(x: &Tensor, w: &MarlinWeight) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = x.device() {
        if w.device().same_device(&Device::Cuda(dev.clone())) { return marlin_gemv_cuda(x, w, dev); }
    }
    cpu_marlin_gemv_fallback(x, w)
}

#[cfg(feature = "cuda")]
fn get_device_ptr(t: &Tensor, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let off = layout.start_offset();
    match &*storage {
        Storage::Cuda(c) => match &c.slice {
            CudaStorageSlice::U32(s) => Ok(s.slice(off..).device_ptr(stream).0),
            CudaStorageSlice::F32(s) => Ok(s.slice(off..).device_ptr(stream).0),
            _ => Err(ModelError::Config("unsupported CUDA storage slice for Marlin GEMV".into())),
        },
        _ => Err(ModelError::Config("tensor not on CUDA device".into())),
    }
}

#[cfg(feature = "cuda")]
static MARLIN_MODULES: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<DeviceId, std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaModule>>>> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn marlin_gemv_cuda(x: &Tensor, w: &MarlinWeight, dev: &candle_core::CudaDevice) -> Result<Tensor> {
    crate::weights::Weights::ensure_current(&Device::Cuda(dev.clone()))?;
    let dims = x.dims().to_vec();
    let in_dim = *dims.last().unwrap_or(&0);
    if in_dim != w.in_dim { return Err(ModelError::Shape { name: "marlin_gemv".into(), expected: vec![w.in_dim], got: dims }); }
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(w.out_dim);
    let orig_dt = x.dtype();
    if rows == 0 || w.out_dim == 0 { return Ok(Tensor::zeros(out_shape, orig_dt, &Device::Cuda(dev.clone()))?); }

    let stream = dev.cuda_stream();
    let ctx = stream.context();
    let cache = MARLIN_MODULES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().map_err(|e| ModelError::Config(e.to_string()))?;
    let module = match guard.get(&dev.id()) {
        Some(m) => m.clone(),
        None => {
            let ptx = candle_core::cuda_backend::cudarc::nvrtc::Ptx::from_src(crate::ops::ptx_marlin::MARLIN_GEMV_PTX);
            let m = ctx.load_module(ptx).map_err(|e| ModelError::Config(format!("marlin ptx: {e}")))?;
            guard.insert(dev.id(), m.clone());
            m
        }
    };
    drop(guard);

    let func = module.load_function("marlin_gemv_kernel").map_err(|e| ModelError::Config(format!("marlin fn: {e}")))?;
    let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
    let x_f32 = if x_c.dtype() == DType::F32 { x_c } else { x_c.to_dtype(DType::F32)? };
    let x_2d = x_f32.reshape((rows, in_dim))?;
    let (x_in, k_padded) = if !in_dim.is_multiple_of(32) {
        let pad = Tensor::zeros((rows, 32 - (in_dim % 32)), DType::F32, &Device::Cuda(dev.clone()))?;
        (Tensor::cat(&[&x_2d, &pad], 1)?, in_dim + 32 - (in_dim % 32))
    } else { (x_2d, in_dim) };

    let p_c = if w.packed.is_contiguous() { w.packed.clone() } else { w.packed.contiguous()? };
    let s_c = if w.scales.is_contiguous() { w.scales.clone() } else { w.scales.contiguous()? };
    let s_f32 = if s_c.dtype() == DType::F32 { s_c } else { s_c.to_dtype(DType::F32)? };
    let out = Tensor::zeros((rows, w.out_dim), DType::F32, &Device::Cuda(dev.clone()))?;
    let (x_ptr, w_ptr) = (get_device_ptr(&x_in, &stream)?, get_device_ptr(&p_c, &stream)?);
    let (s_ptr, y_ptr) = (get_device_ptr(&s_f32, &stream)?, get_device_ptr(&out, &stream)?);
    let (k, n, k_chunks) = (k_padded as u32, w.out_dim as u32, (k_padded / 32) as u32);

    let mut kernel_params = [
        (&x_ptr) as *const _ as *mut std::ffi::c_void, (&w_ptr) as *const _ as *mut std::ffi::c_void,
        (&s_ptr) as *const _ as *mut std::ffi::c_void, (&y_ptr) as *const _ as *mut std::ffi::c_void,
        (&k) as *const _ as *mut std::ffi::c_void, (&n) as *const _ as *mut std::ffi::c_void,
        (&k_chunks) as *const _ as *mut std::ffi::c_void,
    ];

    unsafe {
        candle_core::cuda_backend::cudarc::driver::result::launch_kernel(
            func.cu_function(), (n.div_ceil(128), rows as u32, 1), (128, 1, 1), 0,
            stream.cu_stream(), &mut kernel_params,
        ).map_err(|e| ModelError::Config(format!("marlin launch: {e}")))?;
    }

    let out = out.reshape(out_shape)?;
    if orig_dt != DType::F32 { Ok(out.to_dtype(orig_dt)?) } else { Ok(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marlin_gemv_cpu_matches_dequant() {
        let dev = Device::Cpu;
        let w_raw = Tensor::new(&[[1.0f32, -2.0, 3.0, 0.0], [0.5, 1.5, -0.5, 2.0]], &dev).unwrap();
        let scales = Tensor::new(&[0.5f32, 0.25f32], &dev).unwrap();
        let mw = repack_marlin_tiles(&w_raw, &scales).unwrap();
        assert_eq!(mw.resident_bytes(), (2 * 4) * 4 + 2 * 4);
        let x = Tensor::new(&[[2.0f32, 1.0, -1.0, 3.0]], &dev).unwrap();
        let y = marlin_gemv(&x, &mw).unwrap();
        let deq = mw.dequantize().unwrap();
        let expected = x.matmul(&deq.t().unwrap()).unwrap();
        let y_vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let exp_vals = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (y_v, e_v) in y_vals.iter().zip(exp_vals.iter()) { assert!((y_v - e_v).abs() < 1e-4); }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_marlin_gemv_matches_dequant() {
        for ord in [0, 1] {
            if let Ok(dev) = Device::new_cuda(ord) {
                let (m, k, n) = (2, 50, 32);
                let w_vals: Vec<f32> = (0..(n * k)).map(|i| ((i * 7 + 3) % 15) as f32 - 7.0).collect();
                let w_raw = Tensor::from_vec(w_vals, (n, k), &dev).unwrap();
                let scales = Tensor::from_vec(vec![0.5f32; n], n, &dev).unwrap();
                let mw = repack_marlin_tiles(&w_raw, &scales).unwrap();
                let x_vals: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.1).sin()).collect();
                let x_3d = Tensor::from_vec(x_vals, (1, m, k), &dev).unwrap();
                let y = marlin_gemv(&x_3d, &mw).unwrap();
                assert_eq!(y.dims(), &[1, m, n]);
                let deq = mw.dequantize().unwrap();
                let exp = x_3d.reshape((m, k)).unwrap().matmul(&deq.t().unwrap()).unwrap().reshape((1, m, n)).unwrap();
                let y_vals = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let exp_vals = exp.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for (y_v, e_v) in y_vals.iter().zip(exp_vals.iter()) {
                    assert!((y_v - e_v).abs() < 1e-3, "device {ord} mismatch: got {y_v}, exp {e_v}");
                }
                let empty_x = Tensor::zeros((0, k), DType::F32, &dev).unwrap();
                assert_eq!(marlin_gemv(&empty_x, &mw).unwrap().dims(), &[0, n]);
            }
        }
    }
}
