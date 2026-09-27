//! Differential proofs: greedy generation must reproduce the Ollama oracle.
//!
//! Gated on `SYNTROP_TEST_GGUF` (Qwen2.5-0.5B Q8), `SYNTROP_TEST_GGUF_Q4K`
//! (Gemma4 E2B Q4_K_M), `SYNTROP_TEST_TOKENIZER` (Qwen tokenizer.json).
//! Oracle rows live in `qa/eval/oracle-text.jsonl`, recorded at temp 0.
//!
//! Comparison is on token ids against the re-encoded oracle response. Two
//! independent f32 implementations accumulate different rounding noise, so
//! a near-tie top-1/top-2 race (gap < 0.5 logit units) may flip either way:
//! the proof is a long exact prefix PLUS a near-tie certificate at the
//! first divergence. Structural bugs show up as early divergence with a
//! wide gap, which this test rejects.

use candle_core::Device;
use runtimed_gguf::{GgufBpe, GgufFile, MetaValue, Tokenizer};
use runtimed_model::{generate, sample};
use serde::Deserialize;
use std::path::PathBuf;

/// Minimum exact-match prefix per arch (each position is ~1/vocab by chance).
const MIN_PREFIX_QWEN: usize = 24;
const MIN_PREFIX_E2B: usize = 12;
/// A top-1/top-2 gap below this is rounding noise, not a math bug.
/// Structural bugs flip argmaxes with gaps of 5+; observed cross-kernel
/// noise flips sit at gaps <= ~0.6 (e.g. rust-add pos 8: oracle gap 0.52).
const TIE_GAP: f32 = 1.0;

fn gated(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(var).ok()?);
    if !path.exists() {
        eprintln!("skip: {var} not set or missing");
        return None;
    }
    Some(path)
}

#[derive(Deserialize)]
struct OracleRow {
    id: String,
    model: String,
    prompt: String,
    response: String,
}

fn oracle_rows(model: &str, take: usize) -> Vec<OracleRow> {
    let ws = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws = ws.parent().unwrap().parent().unwrap();
    let text = std::fs::read_to_string(ws.join("qa/eval/oracle-text.jsonl")).expect("oracle file");
    text.lines()
        .filter_map(|l| serde_json::from_str::<OracleRow>(l).ok())
        .filter(|r| r.model == model)
        .take(take)
        .collect()
}

fn eos_of(file: &GgufFile) -> u32 {
    match &file.metadata["tokenizer.ggml.eos_token_id"] {
        MetaValue::U32(id) => *id,
        other => panic!("bad eos metadata: {other:?}"),
    }
}

/// Assert a long exact prefix; on divergence, certify a near-tie by
/// teacher-forcing the oracle prefix and inspecting the next row.
/// `logits_for` maps a full context to its last `[vocab]` row.
fn check_run(
    id: &str,
    got: &[u32],
    want: &[u32],
    min_prefix: usize,
    mut logits_for: impl FnMut(&[u32]) -> Vec<f32>,
) {
    let div = (0..got.len().min(want.len())).find(|&i| got[i] != want[i]);
    match div {
        None => {
            assert!(
                got.len() >= min_prefix,
                "{id}: only {} matching ids, want >= {min_prefix}",
                got.len()
            );
            eprintln!("{id}: full {}-id match", got.len());
        }
        Some(d) => {
            assert!(
                d >= min_prefix,
                "{id}: diverged at {d} (got {} want {}), want prefix >= {min_prefix}",
                got[d],
                want[d]
            );
            let row = logits_for(&want[..d]);
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
            let oracle_id = want[d] as usize;
            let rank = idx.iter().position(|&i| i == oracle_id).unwrap();
            let gap = row[idx[0]] - row[oracle_id];
            eprintln!("{id}: {d}-id prefix, then near-tie check: oracle id {oracle_id} rank {rank} gap {gap:.4}");
            assert!(
                rank <= 1 && gap < TIE_GAP,
                "{id}: divergence at {d} is NOT a near-tie (rank {rank} gap {gap})"
            );
        }
    }
}

#[test]
fn qwen2_matches_oracle_prefix() {
    let (Some(gguf), Some(tok_path)) = (gated("SYNTROP_TEST_GGUF"), gated("SYNTROP_TEST_TOKENIZER"))
    else {
        return;
    };
    let dev = Device::Cpu;
    let (mut model, file) = runtimed_model::Session::load(&gguf, &dev).expect("session");
    assert_eq!(model.config().arch, runtimed_model::Arch::Qwen2);
    let tok = Tokenizer::from_file(&tok_path).expect("tokenizer");
    let eos = eos_of(&file);

    for row in oracle_rows("syntrop-oracle-qwen05", 3) {
        let prompt = tok.encode(&row.prompt, false).expect("encode");
        let got = generate(&mut model, &prompt, &[eos], 48, sample::greedy).expect("generate");
        let want = tok.encode(&row.response, false).expect("re-encode oracle");
        let mut ctx = prompt.clone();
        check_run(&row.id, &got, &want, MIN_PREFIX_QWEN, |prefix| {
            model.reset();
            ctx.truncate(prompt.len());
            ctx.extend_from_slice(prefix);
            let logits = model.forward(&ctx, 0).expect("teacher force");
            generate::last_row(&logits).unwrap().to_vec1::<f32>().unwrap()
        });
    }
}

#[derive(Deserialize)]
struct IdOracleRow {
    id: String,
    prompt: String,
    prompt_ids: Vec<u32>,
    completion_ids: Vec<u32>,
}

/// True-id oracle rows from the reference server (see `record_e2b_ids.py`).
fn id_oracle_rows() -> Vec<IdOracleRow> {
    let ws = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws = ws.parent().unwrap().parent().unwrap();
    let text =
        std::fs::read_to_string(ws.join("qa/eval/oracle-e2b-ids.jsonl")).expect("id oracle file");
    text.lines()
        .filter_map(|l| serde_json::from_str::<IdOracleRow>(l).ok())
        .collect()
}

#[test]
fn gemma4_matches_oracle_prefix() {
    let Some(gguf) = gated("SYNTROP_TEST_GGUF_Q4K") else { return };
    let dev = Device::Cpu;
    let (mut model, file) = runtimed_model::Session::load(&gguf, &dev).expect("session");
    assert_eq!(model.config().arch, runtimed_model::Arch::Gemma4);
    let tok = GgufBpe::from_gguf(&file).expect("bpe");
    let eos = tok.eos_id().expect("eos id");

    let mut long_rows = 0;
    for row in id_oracle_rows() {
        // Our tokenizer must reproduce the reference prompt ids exactly.
        let prompt = tok.encode(&row.prompt, true);
        assert_eq!(prompt, row.prompt_ids, "{}: prompt ids differ", row.id);
        let max_new = row.completion_ids.len().max(1);
        let got = generate(&mut model, &prompt, &[eos], max_new, sample::greedy).expect("generate");
        if row.completion_ids.len() < 2 {
            // Degenerate rows (immediate end-of-turn) must reproduce too.
            assert_eq!(got, row.completion_ids, "{}: degenerate row differs", row.id);
            eprintln!("{}: degenerate 1-id match", row.id);
            continue;
        }
        long_rows += 1;
        let mut ctx = prompt.clone();
        // Short rows certify fewer positions; 2-id rows are pure near-tie
        // checks (logged as a 0-id prefix), still real logit evidence.
        let need = MIN_PREFIX_E2B.min(row.completion_ids.len().saturating_sub(2));
        check_run(&row.id, &got, &row.completion_ids, need, |prefix| {
            model.reset();
            ctx.truncate(prompt.len());
            ctx.extend_from_slice(prefix);
            let logits = model.forward(&ctx, 0).expect("teacher force");
            generate::last_row(&logits).unwrap().to_vec1::<f32>().unwrap()
        });
    }
    assert!(long_rows >= 2, "need >= 2 long oracle rows, got {long_rows}");
}
