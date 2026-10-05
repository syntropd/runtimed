//! Packed 2-bit ternary weight container for BitNet b1.58.
//!
//! Stores ternary weights {-1, 0, +1} packed at 4 weights per byte, with
//! per-channel floating-point scale factors.

use crate::error::{ModelError, Result};

/// Container for packed ternary weights and per-channel scales.
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryWeight {
    pub(crate) packed: Vec<u8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

impl TernaryWeight {
    /// Create a new ternary weight matrix from pre-packed bytes and scales.
    pub fn new(
        packed: Vec<u8>,
        scales: Vec<f32>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let expected_row_bytes = in_features.div_ceil(4);
        let expected_total = expected_row_bytes * out_features;
        if packed.len() != expected_total {
            return Err(ModelError::Config(format!(
                "Ternary packed length mismatch: expected {expected_total} bytes, got {}",
                packed.len()
            )));
        }
        if scales.len() != out_features {
            return Err(ModelError::Config(format!(
                "Ternary scale length mismatch: expected {out_features}, got {}",
                scales.len()
            )));
        }
        Ok(Self {
            packed,
            scales,
            in_features,
            out_features,
        })
    }

    /// Construct packed ternary weights from unquantized FP32 weights.
    pub fn from_unquantized(
        weights: &[f32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if weights.len() != in_features * out_features {
            return Err(ModelError::Config(format!(
                "Weight size mismatch: expected {}, got {}",
                in_features * out_features,
                weights.len()
            )));
        }

        let row_bytes = in_features.div_ceil(4);
        let mut packed = vec![0u8; row_bytes * out_features];
        let mut scales = Vec::with_capacity(out_features);

        for row in 0..out_features {
            let row_start = row * in_features;
            let row_slice = &weights[row_start..row_start + in_features];

            let sum_abs: f32 = row_slice.iter().map(|w| w.abs()).sum();
            let gamma = (sum_abs / (in_features as f32)).max(1e-8);
            scales.push(gamma);

            let out_row_start = row * row_bytes;
            for (idx, &w) in row_slice.iter().enumerate() {
                let scaled = (w / gamma).round();
                let ternary = if scaled >= 1.0 {
                    1i8
                } else if scaled <= -1.0 {
                    -1i8
                } else {
                    0i8
                };

                let byte_idx = out_row_start + (idx / 4);
                let bit_shift = (idx % 4) * 2;
                let code = match ternary {
                    1 => 0b01u8,
                    -1 => 0b10u8,
                    _ => 0b00u8,
                };
                packed[byte_idx] |= code << bit_shift;
            }
        }

        Ok(Self {
            packed,
            scales,
            in_features,
            out_features,
        })
    }

    #[inline]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.in_features.div_ceil(4)
    }

    #[inline]
    pub fn row_slice(&self, row: usize) -> &[u8] {
        let rb = self.row_bytes();
        let start = row * rb;
        &self.packed[start..start + rb]
    }

    #[inline]
    pub fn scale(&self, row: usize) -> f32 {
        self.scales[row]
    }

    /// Unpack a single row into an array of ternary signed values {-1, 0, +1}.
    pub fn unpack_row(&self, row: usize, out: &mut [i8]) {
        let slice = self.row_slice(row);
        let limit = out.len().min(self.in_features);
        for i in 0..limit {
            let byte = slice[i / 4];
            let code = (byte >> ((i % 4) * 2)) & 0b11;
            out[i] = match code {
                0b01 => 1,
                0b10 => -1,
                _ => 0,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_packing_and_unpacking() {
        let raw = vec![1.2, -0.9, 0.05, 1.1, -1.5, 0.0, 0.0, 0.8];
        let tw = TernaryWeight::from_unquantized(&raw, 4, 2).expect("packing");
        assert_eq!(tw.in_features(), 4);
        assert_eq!(tw.out_features(), 2);
        assert_eq!(tw.row_bytes(), 1);

        let mut row0 = [0i8; 4];
        tw.unpack_row(0, &mut row0);
        assert_eq!(row0, [1, -1, 0, 1]);

        let mut row1 = [0i8; 4];
        tw.unpack_row(1, &mut row1);
        assert_eq!(row1, [-1, 0, 0, 1]);
    }
}
