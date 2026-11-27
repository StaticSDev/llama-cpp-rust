//! Port of `src/llama-io.h` and `src/llama-io.cpp`.
//!
//! The C++ interfaces are virtual; Rust uses traits. Tensor-aware methods stay
//! out of the traits until a tensor type exists in the port.

/// `llama_io_write_i`.
pub trait IoWrite {
    fn write(&mut self, src: &[u8]);
    fn n_bytes(&self) -> usize;

    fn write_string(&mut self, str_: &str) {
        let str_size = str_.len() as u32;
        self.write(&str_size.to_le_bytes());
        self.write(str_.as_bytes());
    }
}

/// `llama_io_read_i`.
pub trait IoRead {
    fn read(&mut self, dst: &mut [u8]);
    fn n_bytes(&self) -> usize;

    /// Drop tensor data read but not applied, e.g. when a restore fails.
    fn discard(&mut self) {}

    fn read_string(&mut self) -> String {
        let mut size_buf = [0u8; 4];
        self.read(&mut size_buf);
        let str_size = u32::from_le_bytes(size_buf) as usize;
        let mut buf = vec![0u8; str_size];
        self.read(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cursor {
        data: Vec<u8>,
        pos: usize,
    }

    impl IoWrite for Cursor {
        fn write(&mut self, src: &[u8]) {
            self.data.extend_from_slice(src);
        }
        fn n_bytes(&self) -> usize {
            self.data.len()
        }
    }

    impl IoRead for Cursor {
        fn read(&mut self, dst: &mut [u8]) {
            let end = (self.pos + dst.len()).min(self.data.len());
            let n = end - self.pos;
            dst[..n].copy_from_slice(&self.data[self.pos..end]);
            self.pos = end;
        }
        fn n_bytes(&self) -> usize {
            self.pos
        }
    }

    #[test]
    fn string_round_trip() {
        let mut cur = Cursor { data: Vec::new(), pos: 0 };
        cur.write_string("hello");
        cur.pos = 0;
        assert_eq!(cur.read_string(), "hello");
    }
}
