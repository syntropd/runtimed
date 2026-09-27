//! BPE encode/decode algorithms for [`crate::bpe::GgufBpe`].
//!
//! Loading and vocabulary access stay in [`crate::bpe`]; this module
//! holds the merge loop, special-aware splitting, and decoding.

use crate::bpe::{GgufBpe, TokType};

impl GgufBpe {
    /// Encode one newline-delimited word via leftmost-lowest-rank merges.
    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        // Newline runs try a direct whole-word lookup first.
        if !word.is_empty() && word.bytes().all(|b| b == b'\n') {
            if let Some(&id) = self.id_of.get(word) {
                out.push(id);
                return;
            }
        }
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if parts.is_empty() {
            return;
        }
        loop {
            // Scan adjacent LIVE pairs (emptied slots drop out, like the
            // reference's prev/next chain): the lowest rank merges first,
            // leftmost wins ties.
            let live: Vec<usize> = parts
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.is_empty())
                .map(|(i, _)| i)
                .collect();
            let mut best: Option<(usize, usize, usize)> = None; // (rank, left, right)
            for pair in live.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                if let Some(&rank) = self.ranks.get(&(parts[a].clone(), parts[b].clone())) {
                    if best.map(|(r, _, _)| rank < r).unwrap_or(true) {
                        best = Some((rank, a, b));
                    }
                }
            }
            match best {
                Some((_, a, b)) => {
                    let right = std::mem::take(&mut parts[b]);
                    parts[a].push_str(&right);
                }
                None => break,
            }
        }
        for part in parts.iter().filter(|p| !p.is_empty()) {
            match self.id_of.get(part) {
                Some(&id) => out.push(id),
                None => {
                    // Byte fallback: `<0xXX>` uppercase hex per UTF-8 byte.
                    for b in part.bytes() {
                        let name = format!("<0x{b:02X}>");
                        if let Some(&id) = self.id_of.get(&name) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(bos) = self.bos {
                out.push(bos);
            }
        }
        self.encode_plain(text, &mut out);
        out
    }

    /// Encode with special-token parsing: special strings in the text
    /// become their single ids (longest match wins); other spans encode
    /// as plain BPE. Mirrors the reference's `parse_special` path.
    pub fn encode_special_aware(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(bos) = self.bos {
                out.push(bos);
            }
        }
        let mut plain = String::new();
        let mut i = 0;
        while i < text.len() {
            let rest = &text[i..];
            let hit = self.special_texts.iter().find(|(s, _)| rest.starts_with(s.as_str()));
            match hit {
                Some((s, id)) => {
                    self.encode_plain(&plain, &mut out);
                    plain.clear();
                    out.push(*id);
                    i += s.len();
                }
                None => {
                    let c = rest.chars().next().unwrap();
                    plain.push(c);
                    i += c.len_utf8();
                }
            }
        }
        self.encode_plain(&plain, &mut out);
        out
    }

    /// Plain BPE over text with no special handling.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        // Escape spaces, then split into `[^\n]+|[\n]+` runs.
        let escaped = text.replace(' ', "\u{2581}");
        let mut start = 0;
        let bytes = escaped.as_bytes();
        while start < bytes.len() {
            let is_nl = bytes[start] == b'\n';
            let mut end = start + 1;
            while end < bytes.len() && (bytes[end] == b'\n') == is_nl {
                end += 1;
            }
            self.encode_word(&escaped[start..end], out);
            start = end;
        }
    }

    /// Decode ids to bytes: byte tokens emit raw bytes, normal tokens are
    /// unescaped (`▁` back to space), specials emit literally unless
    /// skipped, other controls are suppressed.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let text = match self.tokens.get(id as usize) {
                Some(t) => t,
                None => continue,
            };
            let ty = self.types[id as usize];
            if skip_special && (self.specials.contains(&id) || ty == TokType::Control) {
                continue;
            }
            match ty {
                TokType::Byte => {
                    if let Some(b) = parse_byte_token(text) {
                        bytes.push(b);
                    }
                }
                TokType::Normal => {
                    bytes.extend(text.replace('\u{2581}', " ").as_bytes());
                }
                _ if self.specials.contains(&id) || ty == TokType::UserDefined => {
                    bytes.extend(text.as_bytes());
                }
                _ => {} // suppressed controls/unknowns
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// `<0xAB>` (any case) to its byte value.
pub(crate) fn parse_byte_token(text: &str) -> Option<u8> {
    let b = text.as_bytes();
    if b.len() == 6 && b[0] == b'<' && b[1] == b'0' && (b[2] == b'x' || b[2] == b'X') && b[5] == b'>'
    {
        u8::from_str_radix(&text[3..5], 16).ok()
    } else {
        None
    }
}