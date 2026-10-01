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

impl TextModel for crate::decode::session::Session {
    fn forward(&mut self, ids: &[u32], q0: usize) -> Result<Tensor> {
        crate::decode::session::Session::forward(self, ids, q0)
    }
    fn reset(&mut self) {
        crate::decode::session::Session::reset(self)
    }
}

/// Last row `[vocab]` of a `[1, seq, vocab]` logit tensor.
pub fn last_row(logits: &Tensor) -> Result<Tensor> {
    let t = logits.dim(1)?;
    if t == 0 {
        return Err(crate::error::ModelError::Config(
            "empty logits sequence has no last row".into(),
        ));
    }
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
    model: &mut crate::decode::session::Session,
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Canned-logits model: every forward returns a peaked row for `id`.
    struct Stub {
        vocab: usize,
        id: u32,
        forwards: usize,
    }

    impl TextModel for Stub {
        fn forward(&mut self, ids: &[u32], _q0: usize) -> Result<Tensor> {
            self.forwards += 1;
            let mut row = vec![0.0f32; self.vocab];
            row[self.id as usize] = 9.0;
            let seq = ids.len().max(1);
            let mut flat = Vec::with_capacity(seq * self.vocab);
            for _ in 0..seq {
                flat.extend_from_slice(&row);
            }
            Ok(Tensor::from_vec(flat, (1, seq, self.vocab), &Device::Cpu)?)
        }

        fn reset(&mut self) {}
    }

    fn argmax(logits: &Tensor) -> Result<u32> {
        let v = logits.to_vec1::<f32>()?;
        Ok(v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32)
    }

    #[test]
    fn last_row_selects_final_step() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu).unwrap();
        let row = last_row(&t).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(row, vec![3.0, 4.0]);
    }

    #[test]
    fn last_row_empty_sequence_returns_error() {
        let t = Tensor::zeros((1, 0, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(last_row(&t).is_err());
    }

    #[test]
    fn stops_on_eos_within_budget() {
        let mut m = Stub { vocab: 8, id: 5, forwards: 0 };
        let out = generate(&mut m, &[1, 2], &[5], 16, argmax).unwrap();
        assert_eq!(out, vec![5]);
        assert_eq!(m.forwards, 1);
    }

    #[test]
    fn fills_budget_without_eos() {
        let mut m = Stub { vocab: 8, id: 3, forwards: 0 };
        let out = generate(&mut m, &[1, 2], &[5], 4, argmax).unwrap();
        assert_eq!(out, vec![3, 3, 3, 3]);
        assert_eq!(m.forwards, 4);
    }
}
