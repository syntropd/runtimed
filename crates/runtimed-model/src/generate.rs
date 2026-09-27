//! Autoregressive generation over any text model.
//!
//! The caller tokenizes; this loop prefills the prompt once, then decodes
//! one token at a time through the KV cache until an end token appears.

use crate::error::Result;
use candle_core::Tensor;

/// Anything that maps token ids to `[1, seq, vocab]` logits.
pub trait TextModel {
    fn forward(&mut self, ids: &[u32], q0: usize) -> Result<Tensor>;
    fn reset(&mut self);
}

impl TextModel for crate::session::Session {
    fn forward(&mut self, ids: &[u32], q0: usize) -> Result<Tensor> {
        crate::session::Session::forward(self, ids, q0)
    }
    fn reset(&mut self) {
        crate::session::Session::reset(self)
    }
}

/// Last row `[vocab]` of a `[1, seq, vocab]` logit tensor.
pub fn last_row(logits: &Tensor) -> Result<Tensor> {
    let t = logits.dim(1)?;
    Ok(logits.narrow(1, t - 1, 1)?.squeeze(1)?.squeeze(0)?)
}

/// Generate up to `max_new` ids after `prompt` (prompt excluded from output).
/// Stops early on any id in `eos`. `next` maps a `[vocab]` row to one id.
pub fn generate<M: TextModel>(
    model: &mut M,
    prompt: &[u32],
    eos: &[u32],
    max_new: usize,
    mut next: impl FnMut(&Tensor) -> Result<u32>,
) -> Result<Vec<u32>> {
    model.reset();
    let logits = model.forward(prompt, 0)?;
    decode_loop(model, &logits, prompt.len(), eos, max_new, &mut next)
}

/// Multimodal generate: the prefill scatters `soft` tokens over the
/// `IMG_TOKEN` placeholders of `prompt`; decoding then runs text-only.
pub fn generate_mm(
    model: &mut crate::session::Session,
    prompt: &[u32],
    soft: &Tensor,
    pad_id: u32,
    eos: &[u32],
    max_new: usize,
    mut next: impl FnMut(&Tensor) -> Result<u32>,
) -> Result<Vec<u32>> {
    model.reset();
    let logits = model.forward_mm(prompt, soft, pad_id)?;
    decode_loop(model, &logits, prompt.len(), eos, max_new, &mut next)
}

fn decode_loop<M: TextModel>(
    model: &mut M,
    first_logits: &Tensor,
    prompt_len: usize,
    eos: &[u32],
    max_new: usize,
    next: &mut impl FnMut(&Tensor) -> Result<u32>,
) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut id = next(&last_row(first_logits)?)?;
    let mut pos = prompt_len;
    loop {
        out.push(id);
        if out.len() >= max_new || eos.contains(&id) {
            break;
        }
        let logits = model.forward(std::slice::from_ref(&id), pos)?;
        pos += 1;
        id = next(&last_row(&logits)?)?;
    }
    Ok(out)
}
