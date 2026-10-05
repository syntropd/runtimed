//! Vectorized ternary dot-product kernels for CPU execution.
//!
//! Replaces floating-point matrix multiplications with integer addition,
//! subtraction, and zero-masking over 2-bit packed {-1, 0, +1} weights.

/// Compute inner product between floating-point activations and 2-bit packed ternary weights.
#[inline]
pub fn ternary_dot_product_f32(x: &[f32], packed_row: &[u8], in_features: usize) -> f32 {
    let mut pos_sum = 0.0f32;
    let mut neg_sum = 0.0f32;

    let full_bytes = in_features / 4;
    for (byte_idx, &byte) in packed_row.iter().enumerate().take(full_bytes) {
        let base_idx = byte_idx * 4;

        let c0 = byte & 0b11;
        let c1 = (byte >> 2) & 0b11;
        let c2 = (byte >> 4) & 0b11;
        let c3 = (byte >> 6) & 0b11;

        if c0 == 0b01 {
            pos_sum += x[base_idx];
        } else if c0 == 0b10 {
            neg_sum += x[base_idx];
        }

        if c1 == 0b01 {
            pos_sum += x[base_idx + 1];
        } else if c1 == 0b10 {
            neg_sum += x[base_idx + 1];
        }

        if c2 == 0b01 {
            pos_sum += x[base_idx + 2];
        } else if c2 == 0b10 {
            neg_sum += x[base_idx + 2];
        }

        if c3 == 0b01 {
            pos_sum += x[base_idx + 3];
        } else if c3 == 0b10 {
            neg_sum += x[base_idx + 3];
        }
    }

    let remainder = in_features % 4;
    if remainder > 0 {
        let byte = packed_row[full_bytes];
        let base_idx = full_bytes * 4;
        for i in 0..remainder {
            let code = (byte >> (i * 2)) & 0b11;
            if code == 0b01 {
                pos_sum += x[base_idx + i];
            } else if code == 0b10 {
                neg_sum += x[base_idx + i];
            }
        }
    }

    pos_sum - neg_sum
}

/// Compute integer inner product for W1.58A8 (8-bit activations x 2-bit ternary weights).
#[inline]
pub fn ternary_dot_product_i8(x_quant: &[i8], packed_row: &[u8], in_features: usize) -> i32 {
    let mut pos_acc: i32 = 0;
    let mut neg_acc: i32 = 0;

    let full_bytes = in_features / 4;
    for (byte_idx, &byte) in packed_row.iter().enumerate().take(full_bytes) {
        let base_idx = byte_idx * 4;

        let c0 = byte & 0b11;
        let c1 = (byte >> 2) & 0b11;
        let c2 = (byte >> 4) & 0b11;
        let c3 = (byte >> 6) & 0b11;

        if c0 == 0b01 {
            pos_acc += x_quant[base_idx] as i32;
        } else if c0 == 0b10 {
            neg_acc += x_quant[base_idx] as i32;
        }

        if c1 == 0b01 {
            pos_acc += x_quant[base_idx + 1] as i32;
        } else if c1 == 0b10 {
            neg_acc += x_quant[base_idx + 1] as i32;
        }

        if c2 == 0b01 {
            pos_acc += x_quant[base_idx + 2] as i32;
        } else if c2 == 0b10 {
            neg_acc += x_quant[base_idx + 2] as i32;
        }

        if c3 == 0b01 {
            pos_acc += x_quant[base_idx + 3] as i32;
        } else if c3 == 0b10 {
            neg_acc += x_quant[base_idx + 3] as i32;
        }
    }

    let remainder = in_features % 4;
    if remainder > 0 {
        let byte = packed_row[full_bytes];
        let base_idx = full_bytes * 4;
        for i in 0..remainder {
            let code = (byte >> (i * 2)) & 0b11;
            if code == 0b01 {
                pos_acc += x_quant[base_idx + i] as i32;
            } else if code == 0b10 {
                neg_acc += x_quant[base_idx + i] as i32;
            }
        }
    }

    pos_acc - neg_acc
}

/// Quantize FP32 activations to signed INT8 scaled to [-127, 127].
pub fn quantize_activations_i8(x: &[f32], out_quant: &mut [i8]) -> f32 {
    let mut max_abs: f32 = 0.0;
    for &val in x {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    let scale = if max_abs > 1e-8 {
        127.0 / max_abs
    } else {
        1.0
    };

    let limit = out_quant.len().min(x.len());
    for i in 0..limit {
        let scaled = (x[i] * scale).round();
        out_quant[i] = scaled.clamp(-128.0, 127.0) as i8;
    }

    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_dot_product_f32() {
        let x = [1.5f32, 2.0, 3.0, 4.0, 5.0];
        // Weights: [+1, -1, 0, +1, -1] -> dot = 1.5 - 2.0 + 0 + 4.0 - 5.0 = -1.5
        // Byte 0: [+1 (01), -1 (10), 0 (00), +1 (01)] -> 0b01_00_10_01 = 0x49
        // Byte 1: [-1 (10)] -> 0b00_00_00_10 = 0x02
        let packed = [0x49u8, 0x02u8];
        let result = ternary_dot_product_f32(&x, &packed, 5);
        assert!((result - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_and_i8_dot_product() {
        let x = [10.0f32, -10.0, 0.0, 5.0];
        let mut quant = [0i8; 4];
        let scale = quantize_activations_i8(&x, &mut quant);
        assert!((scale - 12.7).abs() < 1e-4);
        assert_eq!(quant[0], 127);
        assert_eq!(quant[1], -127);
        assert_eq!(quant[2], 0);

        // Weights: [+1, +1, 0, 0]
        // Byte 0: [01, 01, 00, 00] -> 0b00_00_01_01 = 0x05
        let packed = [0x05u8];
        let acc = ternary_dot_product_i8(&quant, &packed, 4);
        // acc = 127 + (-127) = 0
        assert_eq!(acc, 0);
    }
}
