//! Gemma4 chat templates (the GGUF embeds none; the reference server
//! ships a native one, revealed via its apply-template endpoint).
//!
//! Layout: BOS, a system turn holding the thinking marker, a user turn
//! (image chunk first when present), and an open model turn:
//!
//! ```text
//! <|turn>system\n<|think|>\n<turn|>\n
//! <|turn>user\n[boi N*image eoi]{text}<turn|>\n
//! <|turn>model\n
//! ```
//!
//! Special ids resolve by piece text so they track the file's vocab.

use crate::error::{ModelError, Result};
use runtimed_gguf::GgufBpe;

fn special(tok: &GgufBpe, piece: &str) -> Result<u32> {
    tok.piece_id(piece)
        .ok_or_else(|| ModelError::Config(format!("chat template needs {piece}")))
}

/// Plain-text user prompt ids (BOS + system + user + open model turn).
pub fn text_prompt(tok: &GgufBpe, user: &str) -> Result<Vec<u32>> {
    let bos = tok.bos_id().ok_or_else(|| ModelError::Config("chat template needs BOS".into()))?;
    let (turn, turn_end, think) = (special(tok, "<|turn>")?, special(tok, "<turn|>")?, special(tok, "<|think|>")?);
    let mut ids = vec![bos];
    ids.push(turn);
    ids.extend(tok.encode("system\n", false));
    ids.push(think);
    ids.extend(tok.encode("\n", false));
    ids.push(turn_end);
    ids.extend(tok.encode("\n", false));
    ids.push(turn);
    ids.extend(tok.encode("user\n", false));
    ids.extend(tok.encode(user, false));
    ids.push(turn_end);
    ids.extend(tok.encode("\n", false));
    ids.push(turn);
    ids.extend(tok.encode("model\n", false));
    Ok(ids)
}

/// Multimodal user prompt ids: as [`text_prompt`], with `n_soft`
/// `IMG_TOKEN` placeholders bracketed by boi/eoi ahead of the text.
pub fn mm_prompt(tok: &GgufBpe, user: &str, n_soft: usize) -> Result<Vec<u32>> {
    let bos = tok.bos_id().ok_or_else(|| ModelError::Config("chat template needs BOS".into()))?;
    let (turn, turn_end, think) = (special(tok, "<|turn>")?, special(tok, "<turn|>")?, special(tok, "<|think|>")?);
    let mut ids = vec![bos];
    ids.push(turn);
    ids.extend(tok.encode("system\n", false));
    ids.push(think);
    ids.extend(tok.encode("\n", false));
    ids.push(turn_end);
    ids.extend(tok.encode("\n", false));
    ids.push(turn);
    ids.extend(tok.encode("user\n", false));
    ids.push(crate::vision::IMG_BEG);
    ids.extend(std::iter::repeat(crate::vision::IMG_TOKEN).take(n_soft));
    ids.push(crate::vision::IMG_END);
    ids.extend(tok.encode(user, false));
    ids.push(turn_end);
    ids.extend(tok.encode("\n", false));
    ids.push(turn);
    ids.extend(tok.encode("model\n", false));
    Ok(ids)
}
