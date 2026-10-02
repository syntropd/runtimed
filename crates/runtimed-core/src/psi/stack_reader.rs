//! Zero-allocation non-blocking stack-buffered PSI readers for runtimed.

use rustix::fs::{open, Mode, OFlags};
use rustix::io::read;
use std::path::Path;

/// PSI metrics parsed directly on the stack.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StackPsiValues {
    pub some_avg10: f32,
    pub full_avg10: f32,
}

/// Zero-allocation non-blocking PSI reader.
pub struct StackPsiReader;

impl StackPsiReader {
    /// Read PSI metrics using a fixed stack buffer of `[u8; 64]`.
    pub fn read_path(path: &Path) -> Option<StackPsiValues> {
        let fd = open(
            path,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .ok()?;

        let mut buf = [0u8; 64];
        let mut values = StackPsiValues::default();
        let mut parser = StreamPsiParser::default();

        loop {
            let n = match read(&fd, &mut buf) {
                Ok(0) => break,
                Ok(bytes) => bytes,
                Err(rustix::io::Errno::INTR) => continue,
                Err(_) => break,
            };
            parser.feed(&buf[..n], &mut values);
        }
        parser.finish(&mut values);
        Some(values)
    }

    /// Parse an arbitrary byte slice on the stack.
    pub fn parse_slice(bytes: &[u8]) -> StackPsiValues {
        let mut values = StackPsiValues::default();
        let mut parser = StreamPsiParser::default();
        parser.feed(bytes, &mut values);
        parser.finish(&mut values);
        values
    }
}

struct StreamPsiParser {
    line_start: bool,
    in_some: bool,
    in_full: bool,
    prefix_len: usize,
    prefix_buf: [u8; 5],
    avg10_idx: usize,
    seeking_avg10: bool,
    parsing_num: bool,
    int_part: u32,
    frac_part: u32,
    frac_div: f32,
    in_frac: bool,
    has_digits: bool,
}

impl Default for StreamPsiParser {
    fn default() -> Self {
        Self {
            line_start: true,
            in_some: false,
            in_full: false,
            prefix_len: 0,
            prefix_buf: [0; 5],
            avg10_idx: 0,
            seeking_avg10: false,
            parsing_num: false,
            int_part: 0,
            frac_part: 0,
            frac_div: 1.0,
            in_frac: false,
            has_digits: false,
        }
    }
}

impl StreamPsiParser {
    fn feed(&mut self, chunk: &[u8], values: &mut StackPsiValues) {
        const TARGET: &[u8; 6] = b"avg10=";
        for &b in chunk {
            if b == b'\n' {
                self.commit(values);
                self.reset();
                continue;
            }

            if self.parsing_num {
                if b.is_ascii_digit() {
                    self.has_digits = true;
                    let d = (b - b'0') as u32;
                    if self.in_frac {
                        self.frac_part = self.frac_part.saturating_mul(10).saturating_add(d);
                        self.frac_div *= 10.0;
                    } else {
                        self.int_part = self.int_part.saturating_mul(10).saturating_add(d);
                    }
                } else if b == b'.' && !self.in_frac {
                    self.in_frac = true;
                } else {
                    self.commit(values);
                    self.parsing_num = false;
                    self.seeking_avg10 = false;
                }
                continue;
            }

            if self.line_start {
                if self.prefix_len < 5 {
                    self.prefix_buf[self.prefix_len] = b;
                    self.prefix_len += 1;
                    if self.prefix_len == 5 {
                        if &self.prefix_buf == b"some " {
                            self.in_some = true;
                            self.seeking_avg10 = true;
                            self.line_start = false;
                        } else if &self.prefix_buf == b"full " {
                            self.in_full = true;
                            self.seeking_avg10 = true;
                            self.line_start = false;
                        }
                    }
                } else if b == b' ' {
                    self.line_start = false;
                }
                continue;
            }

            if self.seeking_avg10 {
                if b == TARGET[self.avg10_idx] {
                    self.avg10_idx += 1;
                    if self.avg10_idx == TARGET.len() {
                        self.seeking_avg10 = false;
                        self.parsing_num = true;
                        self.int_part = 0;
                        self.frac_part = 0;
                        self.frac_div = 1.0;
                        self.in_frac = false;
                        self.has_digits = false;
                        self.avg10_idx = 0;
                    }
                } else if b == TARGET[0] {
                    self.avg10_idx = 1;
                } else {
                    self.avg10_idx = 0;
                }
            }
        }
    }

    fn commit(&mut self, values: &mut StackPsiValues) {
        if self.parsing_num && self.has_digits {
            let val = (self.int_part as f32) + (self.frac_part as f32 / self.frac_div);
            if self.in_some {
                values.some_avg10 = val;
            } else if self.in_full {
                values.full_avg10 = val;
            }
        }
    }

    fn finish(&mut self, values: &mut StackPsiValues) {
        self.commit(values);
    }

    fn reset(&mut self) {
        self.line_start = true;
        self.in_some = false;
        self.in_full = false;
        self.prefix_len = 0;
        self.avg10_idx = 0;
        self.seeking_avg10 = false;
        self.parsing_num = false;
        self.int_part = 0;
        self.frac_part = 0;
        self.frac_div = 1.0;
        self.in_frac = false;
        self.has_digits = false;
    }
}
