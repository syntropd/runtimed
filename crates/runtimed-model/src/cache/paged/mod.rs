//! Quantized paged KV cache (FP8 / INT8 / FP16) with two-tier storage.

pub mod block;
pub mod quantize;
pub mod table;

pub use block::{CacheBlock, CachePrecision, QuantizedCacheBlock, StorageTier, BLOCK_SIZE};
pub use quantize::{dequantize_fp8, quantize_block_kv, quantize_fp8};
pub use table::PagedKvCache;
