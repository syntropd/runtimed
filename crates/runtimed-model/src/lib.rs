//! Owned LLM forward pass on the candle tensor substrate.
//!
//! The math here is ours: arch configs parsed from GGUF metadata, weights
//! decoded by `runtimed-gguf`, every op (norm, RoPE, attention, MLP) written
//! out explicitly. Candle provides matmul/softmax/transpose only.

pub mod arch;
pub mod audio;
pub mod cache;
pub mod config;
pub mod decode;
pub mod error;
pub mod lora;
pub mod ops;
pub mod pipeline;
pub mod sampler;
pub mod tokenizer;
pub mod tp;
pub mod vision;
pub mod visual_gen;
pub mod weights;

pub use audio::{AcousticNet, BufferSink, KokoroConfig, KokoroEngine, PcmSink, PwCatSink};
pub use visual_gen::{
    create_sealed_memfd, TurboUnet, VisualComputeLease, VisualGenConfig, VisualGenSampler,
};

pub use cache::{
    sink_causal_mask, CacheBlock, PagedKvCache, SinkWindowCache, SinkWindowConfig, SpillManager,
    StorageTier, BLOCK_SIZE,
};
pub use config::{Activation, Arch, ArchConfig, LayerConfig};
pub use decode::generate::generate;
pub use decode::{HeterogeneousDraftSession, Session};
pub use error::{ModelError, Result};
pub use lora::LoraAdapter;
pub use pipeline::{forward_pipeline, PipelineStage};
pub use sampler::{
    get_or_create_shared_vocab_trie, sample_with_grammar, FsmGrammar, FsmState, GrammarTransition,
    JsonFsm, LogitMask, RegexFsm, VarlinkFsm, VocabTrie,
};
pub use tokenizer::EngineTokenizer;
pub use tp::{ring_all_reduce, ColumnParallelLinear, RingReducer, RowParallelLinear};
pub use vision::{PreparedImage, VisionTower};
pub use weights::Weights;

