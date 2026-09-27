//! Scalar dequantization to `f32`: passthrough, Q8_0, and the dispatcher.
//!
//! K-quant block decoders live in [`crate::dequant_k`]. Ported from ggml's
//! `src/ggml-quants.c`; block layouts in `src/ggml-common.h`.

use crate::dequant_k::{dequant_q4_k_block, dequant_q5_k_block, dequant_q6_k_block};
use crate::dtype::GgmlDtype;
use crate::error::{GgufError, Result};

/// IEEE-754 half → `f32`, subnormals included.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = 127 - 14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

pub(crate) fn read_f16(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// One Q8_0 block (34 bytes) → 32 floats: `y = d * q`.
pub fn dequant_q8_0_block(block: &[u8]) -> [f32; 32] {
    let d = read_f16(&block[0..2]);
    let mut out = [0.0f32; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = d * block[2 + i] as i8 as f32;
    }
    out
}


/// Whole-tensor decode. `n_elements` must be an exact block multiple.
pub fn dequant_tensor(dtype: GgmlDtype, bytes: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let Some((elems, block_bytes)) = dtype.block() else {
        return Err(GgufError::Unsupported(dtype));
    };
    if n_elements % elems != 0 {
        return Err(GgufError::Overflow(format!(
            "{n_elements} not a multiple of {elems}"
        )));
    }
    let need = n_elements / elems * block_bytes;
    if bytes.len() < need {
        return Err(GgufError::Truncated(bytes.len()));
    }
    let mut out = Vec::with_capacity(n_elements);
    match dtype {
        GgmlDtype::F32 => {
            for c in bytes[..need].chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        GgmlDtype::F16 => {
            for c in bytes[..need].chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        GgmlDtype::BF16 => {
            for c in bytes[..need].chunks_exact(2) {
                out.push(f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16));
            }
        }
        GgmlDtype::Q8_0 => {
            for b in bytes[..need].chunks_exact(34) {
                out.extend_from_slice(&dequant_q8_0_block(b));
            }
        }
        GgmlDtype::Q4K => {
            for b in bytes[..need].chunks_exact(144) {
                out.extend_from_slice(&dequant_q4_k_block(b));
            }
        }
        GgmlDtype::Q5K => {
            for b in bytes[..need].chunks_exact(176) {
                out.extend_from_slice(&dequant_q5_k_block(b));
            }
        }
        GgmlDtype::Q6K => {
            for b in bytes[..need].chunks_exact(210) {
                out.extend_from_slice(&dequant_q6_k_block(b));
            }
        }
        other => return Err(GgufError::Unsupported(other)),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_corners() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!((f16_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-30);
    }

    #[test]
    fn q8_0_scales_ints() {
        let mut block = vec![0x00, 0x3C]; // d = 1.0
        block.extend(0u8..32);
        let out = dequant_q8_0_block(&block);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as f32);
        }
    }

}