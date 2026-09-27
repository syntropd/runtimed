//! Debug probe: top-5 logits per position for one oracle prompt.
use candle_core::Device;
use runtimed_gguf::{MetaValue, Tokenizer};
use runtimed_model::{generate, sample};

fn main() {
    let id = std::env::args().nth(1).expect("usage: diverge <prompt-id> [n]");
    let n: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(40);
    let dev = Device::Cpu;
    let (mut model, file) =
        runtimed_model::Session::load("/var/lib/models/gguf/qwen2.5-0.5b-instruct-q8_0.gguf".as_ref(), &dev).unwrap();
    let tok = Tokenizer::from_file("/var/lib/models/gguf/qwen2.5-0.5b-tokenizer.json".as_ref()).unwrap();
    let eos = match &file.metadata["tokenizer.ggml.eos_token_id"] {
        MetaValue::U32(x) => *x,
        _ => panic!("eos"),
    };
    let ws = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let text = std::fs::read_to_string(ws.join("qa/eval/oracle-text.jsonl")).unwrap();
    let row: serde_json::Value = text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|r| r["id"] == id && r["model"] == "syntrop-oracle-qwen05")
        .expect("row");
    let prompt = tok.encode(row["prompt"].as_str().unwrap(), false).unwrap();
    let want_text = row["response"].as_str().unwrap().to_string();
    let want_ids = tok.encode(&want_text, false).unwrap();
    let got = generate(&mut model, &prompt, &[eos], n, sample::greedy).unwrap();
    println!("our text: {:?}", tok.decode(&got, true).unwrap());
    println!("oracle  : {:?}", &want_text[..want_text.len().min(200)]);
    for i in 0..n.min(got.len()).min(want_ids.len()) {
        let mark = if got[i] == want_ids[i] { " " } else { "*" };
        println!("{mark} pos {i}: got {} want {}", got[i], want_ids[i]);
        if got[i] != want_ids[i] {
            break;
        }
    }
    // Top-5 at the divergence point, replayed with teacher forcing.
    let div = (0..n).find(|&i| got.get(i) != want_ids.get(i)).unwrap_or(n);
    println!("first divergence at {div}");
    model.reset();
    let mut ctx = prompt.clone();
    ctx.extend_from_slice(&want_ids[..div]);
    let logits = model.forward(&ctx, 0).unwrap();
    let row = runtimed_model::generate::last_row(&logits).unwrap().to_vec1::<f32>().unwrap();
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
    println!("top-5 at div: {:?}", idx[..5].iter().map(|&i| (i, row[i])).collect::<Vec<_>>());
    println!("gap top1-top2: {}", row[idx[0]] - row[idx[1]]);
}
use std::path::PathBuf;
