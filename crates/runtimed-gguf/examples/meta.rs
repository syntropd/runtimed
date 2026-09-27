//! Dump GGUF metadata keys and tensor summary: aids debugging, no deps.
use runtimed_gguf::GgufFile;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: meta <file.gguf>");
    let file = GgufFile::open(path.as_ref()).expect("must parse");
    let mut keys: Vec<&String> = file.metadata.keys().collect();
    keys.sort();
    println!("metadata keys ({}):", keys.len());
    for k in &keys {
        let v = &file.metadata[*k];
        let s = format!("{v:?}");
        println!("  {k} = {}", s.chars().take(100).collect::<String>());
    }
    let mut by_dtype: BTreeMap<String, usize> = BTreeMap::new();
    for t in &file.tensors {
        *by_dtype.entry(format!("{:?}", t.dtype)).or_default() += 1;
    }
    println!("tensors: {} {:?}", file.tensors.len(), by_dtype);
    println!("first: {}", file.tensors.first().map(|t| t.name.as_str()).unwrap_or("?"));
}
