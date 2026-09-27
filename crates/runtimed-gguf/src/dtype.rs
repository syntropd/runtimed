//! GGML dtype ids and block geometry, per the GGUF spec.

/// Dtype ids from the GGUF spec. Only a subset decodes (see [`GgmlDtype::block`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    BF16 = 30,
    /// Any other id parses (headers stay total) but never decodes.
    Unknown(u32),
}

impl GgmlDtype {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    /// (elements per block, bytes per block) for decodable types.
    pub fn block(self) -> Option<(usize, usize)> {
        let b = match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            _ => return None,
        };
        Some(b)
    }

    pub fn decodable(self) -> bool {
        self.block().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_ids_match_spec() {
        assert_eq!(GgmlDtype::from_id(0), GgmlDtype::F32);
        assert_eq!(GgmlDtype::from_id(8), GgmlDtype::Q8_0);
        assert_eq!(GgmlDtype::from_id(12), GgmlDtype::Q4K);
        assert_eq!(GgmlDtype::from_id(30), GgmlDtype::BF16);
        assert_eq!(GgmlDtype::from_id(99), GgmlDtype::Unknown(99));
        assert!(!GgmlDtype::Unknown(99).decodable());
    }

    #[test]
    fn block_geometry() {
        assert_eq!(GgmlDtype::Q8_0.block(), Some((32, 34)));
        assert_eq!(GgmlDtype::Q4K.block(), Some((256, 144)));
        assert_eq!(GgmlDtype::F32.block(), Some((1, 4)));
        assert_eq!(GgmlDtype::Q6K.block(), None);
    }
}
