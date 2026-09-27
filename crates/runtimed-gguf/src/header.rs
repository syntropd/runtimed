//! GGUF header parsing over a memory-mapped file.
//!
//! Layout: magic `GGUF`, version u32, tensor count u64, metadata count u64,
//! then metadata entries, tensor infos, then tensor data starting at the
//! `general.alignment` boundary (default 32). Little-endian throughout.

use crate::dtype::GgmlDtype;
use crate::error::{GgufError, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlDtype,
    /// Absolute file offset of this tensor's bytes.
    pub offset: u64,
    pub n_elements: usize,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated(self.pos))?;
        if end > self.bytes.len() {
            return Err(GgufError::Truncated(self.pos));
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| GgufError::Truncated(self.pos))
    }

    fn value(&mut self, ty: u32) -> Result<MetaValue> {
        let v = match ty {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.u8()? as i8),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.u16()? as i16),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::Str(self.string()?),
            9 => {
                let elem = self.u32()?;
                let len = self.u64()? as usize;
                let mut items = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    items.push(self.value(elem)?);
                }
                MetaValue::Arr(items)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(GgufError::MetaType(other)),
        };
        Ok(v)
    }
}

/// A parsed GGUF file backed by an mmap. All tensor access is bounds-checked.
pub struct GgufFile {
    mmap: Mmap,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: weight files are verified by hash before use and never
        // mutated while mapped; all tensor access is bounds-checked.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let mut cur = Cursor { bytes: &mmap, pos: 0 };
        if cur.take(4)? != b"GGUF" {
            return Err(GgufError::BadMagic);
        }
        let version = cur.u32()?;
        if version != 3 {
            return Err(GgufError::Version(version));
        }
        let n_tensors = cur.u64()? as usize;
        let n_meta = cur.u64()? as usize;
        let mut metadata = HashMap::with_capacity(n_meta.min(4096));
        for _ in 0..n_meta {
            let key = cur.string()?;
            let ty = cur.u32()?;
            metadata.insert(key, cur.value(ty)?);
        }
        let mut raw_infos = Vec::with_capacity(n_tensors.min(100_000));
        for _ in 0..n_tensors {
            let name = cur.string()?;
            let n_dims = cur.u32()? as usize;
            if n_dims > 8 {
                return Err(GgufError::Truncated(cur.pos));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.u64()?);
            }
            let dtype = GgmlDtype::from_id(cur.u32()?);
            let offset = cur.u64()?;
            raw_infos.push((name, dims, dtype, offset));
        }
        let align = match metadata.get("general.alignment") {
            Some(MetaValue::U32(a)) => *a as u64,
            _ => 32,
        }
        .max(1);
        let data_start = (cur.pos as u64 + align - 1) / align * align;
        let file_len = mmap.len() as u64;
        let mut tensors = Vec::with_capacity(raw_infos.len());
        for (name, dims, dtype, rel) in raw_infos {
            let n_elements: usize = dims.iter().try_fold(1usize, |a, &d| {
                a.checked_mul(d as usize)
                    .ok_or_else(|| GgufError::Overflow(name.clone()))
            })?;
            let offset = data_start
                .checked_add(rel)
                .ok_or_else(|| GgufError::Overflow(name.clone()))?;
            if offset > file_len {
                return Err(GgufError::Range(name));
            }
            tensors.push(TensorInfo { name, dims, dtype, offset, n_elements });
        }
        Ok(Self { mmap, metadata, tensors })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes of a tensor, bounds-checked against the mapping.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let Some((elems, block_bytes)) = info.dtype.block() else {
            return Err(GgufError::Unsupported(info.dtype));
        };
        let blocks = info.n_elements.div_ceil(elems);
        let len = blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| GgufError::Overflow(info.name.clone()))?;
        let start = info.offset as usize;
        let end = start.checked_add(len).ok_or_else(|| GgufError::Range(info.name.clone()))?;
        if end > self.mmap.len() {
            return Err(GgufError::Range(info.name.clone()));
        }
        Ok(&self.mmap[start..end])
    }

    /// Decode a whole tensor to `f32`.
    pub fn tensor_f32(&self, info: &TensorInfo) -> Result<Vec<f32>> {
        let bytes = self.tensor_bytes(info)?;
        crate::dequant::dequant_tensor(info.dtype, bytes, info.n_elements)
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetaValue::Str(s)) => Some(s),
            _ => None,
        }
    }
}
