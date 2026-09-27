//! Decode every tensor to finite f32. Usage: decode_all -- <file.gguf>
use runtimed_gguf::GgufFile;
fn main() {
    let path = std::env::args().nth(1).expect("usage: shapes <file.gguf>");
    let f = GgufFile::open(path.as_ref()).expect("open");
    let mut n = 0u64;
    let mut sum = 0.0f64;
    for t in &f.tensors {
        let v = f.tensor_f32(t).unwrap_or_else(|e| panic!("{}: {e:?}", t.name));
        assert_eq!(v.len(), t.n_elements, "len {}", t.name);
        for &x in &v {
            assert!(x.is_finite(), "non-finite in {}", t.name);
            sum += x as f64;
        }
        n += v.len() as u64;
    }
    println!("decoded {} tensors, {} elems, sum {:.6}", f.tensors.len(), n, sum);
}
