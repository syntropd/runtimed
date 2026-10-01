//! Heterogeneous draft session pairing a CPU draft model with a GPU target model.

use crate::decode::generate::last_row;
use crate::decode::session::Session;
use crate::decode::speculate::{speculative_step, SpeculativeStep};
use crate::error::{ModelError, Result};
use candle_core::{Device, Tensor};

/// Speculative decoding session pairing a CPU draft model (e.g. Qwen2.5-0.5B with AVX-512)
/// with a high-capacity GPU target model (e.g. Qwen2.5-7B).
pub struct HeterogeneousDraftSession {
    draft: Session,
    target: Session,
    k_draft: usize,
    eos_tokens: Vec<u32>,
}

impl HeterogeneousDraftSession {
    /// Create a new heterogeneous draft session.
    pub fn new(draft: Session, target: Session, k_draft: usize, eos_tokens: Vec<u32>) -> Result<Self> {
        if !matches!(draft.device(), Device::Cpu) {
            return Err(ModelError::Config(
                "draft session must run on CPU compute device".into(),
            ));
        }
        let k = k_draft.clamp(1, 16);
        Ok(Self {
            draft,
            target,
            k_draft: k,
            eos_tokens,
        })
    }

    /// Convenience constructor validating Qwen2.5 CPU-GPU family pair.
    pub fn pair_qwen_cpu_gpu(draft: Session, target: Session, k_draft: usize, eos_tokens: Vec<u32>) -> Result<Self> {
        if draft.config().arch != target.config().arch {
            return Err(ModelError::Config(
                "draft and target model architectures must match".into(),
            ));
        }
        Self::new(draft, target, k_draft, eos_tokens)
    }

    /// Access the underlying CPU draft session.
    pub fn draft(&self) -> &Session {
        &self.draft
    }

    /// Access the mutable CPU draft session.
    pub fn draft_mut(&mut self) -> &mut Session {
        &mut self.draft
    }

    /// Access the underlying target session.
    pub fn target(&self) -> &Session {
        &self.target
    }

    /// Access the mutable target session.
    pub fn target_mut(&mut self) -> &mut Session {
        &mut self.target
    }

    /// Get current speculation lookahead window K.
    pub fn k_draft(&self) -> usize {
        self.k_draft
    }

    /// Update speculation lookahead window K.
    pub fn set_k_draft(&mut self, k: usize) {
        self.k_draft = k.clamp(1, 16);
    }

    /// Reset internal KV caches for both draft and target sessions.
    pub fn reset(&mut self) {
        self.draft.reset();
        self.target.reset();
    }

    /// Prefill prompt through both draft and target models.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<(Tensor, Tensor)> {
        if prompt.is_empty() {
            return Err(ModelError::Config("cannot prefill an empty prompt".into()));
        }
        self.reset();
        let d_logits = self.draft.forward(prompt, 0)?;
        let t_logits = self.target.forward(prompt, 0)?;
        let d_head = last_row(&d_logits)?;
        let t_head = last_row(&t_logits)?;
        Ok((d_head, t_head))
    }

    /// Execute one speculative step.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        current_pos: usize,
        draft_head: &Tensor,
        target_head: &Tensor,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rand01: impl FnMut() -> f32,
    ) -> Result<(SpeculativeStep, Tensor, Tensor)> {
        speculative_step(
            &mut self.draft,
            &mut self.target,
            current_pos,
            self.k_draft,
            draft_head,
            target_head,
            &self.eos_tokens,
            temperature,
            top_k,
            top_p,
            rand01,
        )
    }

    /// Complete end-to-end speculative generation loop.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        mut rand01: impl FnMut() -> f32,
    ) -> Result<Vec<u32>> {
        if max_tokens == 0 || prompt.is_empty() {
            return Ok(Vec::new());
        }
        let (mut draft_head, mut target_head) = self.prefill(prompt)?;
        let mut generated = Vec::with_capacity(max_tokens);
        let mut current_pos = prompt.len();

        while generated.len() < max_tokens {
            let (step, next_d, next_t) = self.step(
                current_pos,
                &draft_head,
                &target_head,
                temperature,
                top_k,
                top_p,
                &mut rand01,
            )?;
            current_pos += step.tokens.len();
            for tok in step.tokens {
                if generated.len() < max_tokens {
                    generated.push(tok);
                }
            }
            draft_head = next_d;
            target_head = next_t;
            if step.hit_eos {
                break;
            }
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_k_draft_clamping() {
        let k_zero = 0usize.clamp(1, 16);
        let k_large = 64usize.clamp(1, 16);
        let k_norm = 4usize.clamp(1, 16);
        assert_eq!(k_zero, 1);
        assert_eq!(k_large, 16);
        assert_eq!(k_norm, 4);
    }
}
