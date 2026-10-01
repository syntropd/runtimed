//! K-quant block decoders (Q4_K, Q5_K, Q6_K), ported from ggml.
//!
//! Super-block layouts live here; [`crate::quant::dequant`] dispatches rows.

use crate::quant::dequant::read_f16;

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
        for &byte in q {
            out[o] = d1 * (byte & 0xF) as f32 - m1;
            o += 1;
        }
        for &byte in q {
            out[o] = d2 * (byte >> 4) as f32 - m2;
            o += 1;
        }
        is += 2;
    }
    out
}

/// One Q5_K super-block (176 bytes) → 256 floats.
///
/// Layout: `d`, `dmin`, 12 scale bytes, 32 high-bit bytes, 128 low-nibble
/// bytes. The 32 `qh` bytes are reused across all four 64-value groups with
/// a shifting 2-bit mask (`u1`/`u2`), unlike Q4_K's advancing `qs` only.
pub fn dequant_q5_k_block(block: &[u8]) -> [f32; 256] {
    let d = read_f16(&block[0..2]);
    let min = read_f16(&block[2..4]);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut out = [0.0f32; 256];
    let mut o = 0;
    let (mut is, mut u1, mut u2) = (0, 1u8, 2u8);
    for chunk in 0..4 {
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let (sc, m) = scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, min * m as f32);
        let (sc, m) = scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, min * m as f32);
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            out[o] = d1 * ((q[l] & 0xF) as f32 + hi) - m1;
            o += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            out[o] = d2 * ((q[l] >> 4) as f32 + hi) - m2;
            o += 1;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    out
}

/// One Q6_K super-block (210 bytes) → 256 floats.
///
/// Layout: 128 low-nibble bytes, 64 2-bit-high bytes, 16 `i8` scales, `d`.
/// Values are 6-bit signed around zero (`q - 32`); there is no min term.
pub fn dequant_q6_k_block(block: &[u8]) -> [f32; 256] {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = read_f16(&block[208..210]);
    let mut out = [0.0f32; 256];
    for half in 0..2 {
        let (ql, qh, sc) = (&ql[half * 64..], &qh[half * 32..], &scales[half * 8..]);
        let y = &mut out[half * 128..];
        for l in 0..32 {
            let is = l / 16;
            // 6-bit patterns are < 64, so `as i8` is exact (matches C cast).
            let q = [
                (((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 as f32) - 32.0,
                (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 as f32) - 32.0,
                (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as f32) - 32.0,
                (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as f32) - 32.0,
            ];
            for (k, v) in q.iter().enumerate() {
                y[l + k * 32] = d * sc[is + k * 2] as i8 as f32 * v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn q5_k_high_bit_reuse_across_groups() {
        // d = 1.0, dmin = 0.0, all scales/mins 1, low nibbles 1.
        let mut block = vec![0x00, 0x3C, 0x00, 0x00];
        block.extend([1u8; 12]);
        block.extend([0u8; 32]); // qh: no high bits
        block.extend([0x11u8; 128]);
        let out = dequant_q5_k_block(&block);
        assert!(out.iter().all(|&v| v == 1.0));
        // Same block with all high bits set: every group gains 16.
        let mut block_hi = block.clone();
        block_hi[16..48].fill(0xFF);
        let out_hi = dequant_q5_k_block(&block_hi);
        assert!(out_hi.iter().all(|&v| v == 17.0));
        // Single bit: qh[0] bit0 feeds group 0 low nibbles only.
        let mut block_one = block.clone();
        block_one[16] = 0x01;
        let out_one = dequant_q5_k_block(&block_one);
        assert_eq!(out_one[0], 17.0);
        assert_eq!(out_one[1], 1.0);
        assert_eq!(out_one[32], 1.0); // u2 mask (bit1) not set
    }

    #[test]
    fn q6_k_signed_values_and_scales() {
        // All-zero quants, all scales 1, d = 1.0 → every value -32.
        let mut block = vec![0u8; 208];
        block.extend([0x00, 0x3C]); // d = 1.0
        block[192..208].fill(1);
        let out = dequant_q6_k_block(&block);
        assert!(out.iter().all(|&v| v == -32.0));
        // ql[0] = 0x11, qh[0] = 0xFF: q1 = (1 | 48) - 32 = 17.
        let mut block_q = block.clone();
        block_q[0] = 0x11;
        block_q[128] = 0xFF;
        block_q[192..200].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let out_q = dequant_q6_k_block(&block_q);
        assert_eq!(out_q[0], 17.0); // sc[0] = 1
        assert_eq!(out_q[32], 48.0); // q2 = (0|48)-32 = 16, sc[2] = 3
        assert_eq!(out_q[64], 85.0); // q3 = 17, sc[4] = 5
        assert_eq!(out_q[96], 112.0); // q4 = 16, sc[6] = 7
        // l = 16 selects the odd scale lane (is = 1).
        let mut block_lane = block.clone();
        block_lane[192..200].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        block_lane[16] = 0x11;
        let out_lane = dequant_q6_k_block(&block_lane);
        assert_eq!(out_lane[16], -62.0); // (1 - 32) * sc[1]=2
    }
}