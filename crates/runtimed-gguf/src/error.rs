use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("bad magic: not a GGUF file")]
    BadMagic,

    #[error("unsupported GGUF version: {0} (want 3)")]
    Version(u32),

    #[error("truncated file at offset {0}")]
    Truncated(usize),

    #[error("unknown metadata type id: {0}")]
    MetaType(u32),

    #[error("tensor '{0}': byte range out of bounds")]
    Range(String),

    #[error("tensor '{0}': element count overflows")]
    Overflow(String),

    #[error("decode of {0:?} not implemented")]
    Unsupported(super::dtype::GgmlDtype),

    #[error("hash mismatch for {0}: want {1}, got {2}")]
    Hash(String, String, String),

    #[error("registry: {0}")]
    Registry(String),

    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;
