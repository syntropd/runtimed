//! Scalar dequantization to `f32`, ported from ggml's reference routines.
//!
//! Only what the engine needs: F32/F16 passthrough, Q8_0, Q4_K.
//! Reference: `dequantize_row_q8_0`, `dequantize_row_q4_K`,
//! `get_scale_min_k4` in ggml's `src/ggml-quants.c`.

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

fn read_f16(bytes: &[u8]) -> f32 {
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

/// Scale/min pair `j` (0..8) from a Q4_K super-block's 12 scale bytes.
fn scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// One Q4_K super-block (144 bytes) → 256 floats.
pub fn dequant_q4_k_block(block: &[u8]) -> [f32; 256] {
    let d = read_f16(&block[0..2]);
    let min = read_f16(&block[2..4]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut out = [0.0f32; 256];
    let mut o = 0;
    let mut is = 0;
    for chunk in 0..4 {
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let (sc, m) = scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, min * m as f32);
        let (sc, m) = scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, min * m as f32);
        for l in 0..32 {
            out[o] = d1 * (q[l] & 0xF) as f32 - m1;
            o += 1;
        }
        for l in 0..32 {
            out[o] = d2 * (q[l] >> 4) as f32 - m2;
            o += 1;
        }
        is += 2;
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

    #[test]
    fn q4_k_nibbles_and_high_bits() {
        // d = 1.0, dmin = 0.0, scales[0] carries high bits for group 4.
        let mut block = vec![0x00, 0x3C, 0x00, 0x00];
        block.extend([65u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        block.extend([0x11u8; 128]); // low nibble 1, high nibble 1
        let out = dequant_q4_k_block(&block);
        assert_eq!(out.len(), 256);
        assert!(out[0..32].iter().all(|&v| v == 1.0));
        assert!(out[32..64].iter().all(|&v| v == 1.0));
        // group 4: sc = (1 & 0xF) | ((65 >> 6) << 4) = 17
        assert!(out[128..160].iter().all(|&v| v == 17.0));
    }
}
