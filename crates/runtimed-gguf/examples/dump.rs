//! Dump GGUF metadata + tensor inventory. Usage: cargo run -p syntrop-runtimed-gguf --example dump -- <file.gguf>
use std::collections::BTreeMap;
use runtimed_gguf::GgufFile;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <file.gguf>");
    let f = GgufFile::open(path.as_ref()).expect("open");
    println!("== metadata ({}) ==", f.metadata.len());
    let mut keys: Vec<_> = f.metadata.keys().collect();
    keys.sort();
    for k in keys {
        if k.starts_with("tokenizer.") && !k.starts_with("tokenizer.ggml.") {
            continue;
        }
        let v = &f.metadata[k];
        let s = format!("{v:?}");
        let s = if s.len() > 160 { format!("{}…", &s[..160]) } else { s };
        println!("{k} = {s}");
    }
    println!("== tensors ({}) ==", f.tensors.len());
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for t in &f.tensors {
        *hist.entry(format!("{:?}", t.dtype)).or_default() += 1;
    }
    for (d, n) in &hist {
        println!("{d}: {n}");
    }
    println!("== layer-0 tensor names ==");
    for t in f.tensors.iter().filter(|t| t.name.contains(".0.")).take(30) {
        println!("{} {:?} {:?}", t.name, t.dims, t.dtype);
    }
    println!("== non-blk tensors ==");
    for t in f.tensors.iter().filter(|t| !t.name.starts_with("blk.")) {
        println!("{} {:?} {:?}", t.name, t.dims, t.dtype);
    }
}
