//! Port of `src/llama-mmap.h` and `src/llama-mmap.cpp`.
//!
//! Three C++ types are translated: `llama_file`, `llama_mmap` and `llama_mlock`.
//! The C++ versions use OS primitives (`open`/`mmap`/`mlock`, or their Win32
//! equivalents). This port uses only `std::fs` and safe `std` APIs.
//!
//! Shortcut: `std` cannot request `O_DIRECT`, map a file into memory, or lock
//! pages. Direct I/O is kept as a recorded flag with alignment 1. `LlamaMmap`
//! owns a byte buffer and tracks the mapped ranges instead of a raw pointer.
//! `LlamaMlock` records the locked size instead of calling `mlock`.
//! Page size is assumed to be 4096; `std` has no portable page-size query.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

use crate::core::{Error, Result};

/// Staging buffer size for direct I/O reads (`LLAMA_DIRECT_IO_BUFFER_SIZE`).
pub const LLAMA_DIRECT_IO_BUFFER_SIZE: usize = 64 * 1024 * 1024;

/// Assumed page size. C++ reads this with `sysconf(_SC_PAGESIZE)`.
pub const PAGE_SIZE: usize = 4096;

/// `SEEK_SET` from `<cstdio>`.
pub const SEEK_SET: i32 = 0;
/// `SEEK_CUR` from `<cstdio>`.
pub const SEEK_CUR: i32 = 1;
/// `SEEK_END` from `<cstdio>`.
pub const SEEK_END: i32 = 2;

fn io_err(context: &str, err: std::io::Error) -> Error {
    Error::new(format!("{}: {}", context, err))
}

// ---------------------------------------------------------------------------
// llama_file
// ---------------------------------------------------------------------------

/// Port of `llama_file`. Owns a `std::fs::File` and a snapshot of its size.
pub struct LlamaFile {
    file: File,
    size: usize,
    alignment: usize,
    use_direct_io: bool,
    fname: String,
}

/// `llama_files`.
pub type LlamaFiles = Vec<LlamaFile>;

impl LlamaFile {
    /// `llama_file(fname, mode, use_direct_io)`.
    ///
    /// `mode` follows `fopen`: `r`, `rb`, `r+`, `w`, `wb`, `w+`, `a`, `ab`,
    /// `a+` and their `+b` variants. Like the C++ constructor, the file is
    /// positioned at offset 0 on return and `size` is the end-of-file offset.
    pub fn open(fname: &str, mode: &str, use_direct_io: bool) -> Result<Self> {
        let mut opts = OpenOptions::new();
        match mode {
            "r" | "rb" => {
                opts.read(true);
            }
            "r+" | "rb+" | "r+b" => {
                opts.read(true).write(true);
            }
            "w" | "wb" => {
                opts.write(true).create(true).truncate(true);
            }
            "w+" | "wb+" | "w+b" => {
                opts.read(true).write(true).create(true).truncate(true);
            }
            "a" | "ab" => {
                opts.append(true).create(true);
            }
            "a+" | "ab+" | "a+b" => {
                opts.read(true).append(true).create(true);
            }
            _ => return Err(Error::new(format!("unsupported file mode '{}'", mode))),
        }

        let mut file = opts
            .open(fname)
            .map_err(|e| io_err(&format!("failed to open '{}'", fname), e))?;
        let size = Self::measure(&mut file, fname)?;
        Ok(Self {
            file,
            size,
            alignment: 1,
            use_direct_io,
            fname: fname.to_string(),
        })
    }

    /// `llama_file(FILE * file)`. The C++ version moves the position to the
    /// end and back to 0; this does the same.
    pub fn from_file(mut file: File) -> Result<Self> {
        let size = Self::measure(&mut file, "(file*)")?;
        Ok(Self {
            file,
            size,
            alignment: 1,
            use_direct_io: false,
            fname: "(file*)".to_string(),
        })
    }

    fn measure(file: &mut File, fname: &str) -> Result<usize> {
        let end = file
            .seek(SeekFrom::End(0))
            .map_err(|e| io_err(&format!("seek error on '{}'", fname), e))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| io_err(&format!("seek error on '{}'", fname), e))?;
        Ok(end as usize)
    }

    /// `tell()`. Uses a cloned handle, which shares the file position, so the
    /// signature stays `&self` like the C++ `const` method.
    pub fn tell(&self) -> Result<usize> {
        let mut handle = self
            .file
            .try_clone()
            .map_err(|e| io_err("tell error (clone)", e))?;
        let pos = handle
            .stream_position()
            .map_err(|e| io_err(&format!("tell error on '{}'", self.fname), e))?;
        Ok(pos as usize)
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.size
    }

    /// `file_id()`. Raw fd on Unix; -1 elsewhere since there is no portable
    /// equivalent. `LlamaMmap` does not need it: it reads through `LlamaFile`.
    pub fn file_id(&self) -> i64 {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            self.file.as_raw_fd() as i64
        }
        #[cfg(not(unix))]
        {
            let _ = &self.file;
            -1
        }
    }

    /// `seek(offset, whence)`. `whence` is one of `SEEK_SET`, `SEEK_CUR`,
    /// `SEEK_END`.
    pub fn seek(&mut self, offset: usize, whence: i32) -> Result<()> {
        let pos = match whence {
            SEEK_SET => SeekFrom::Start(offset as u64),
            SEEK_CUR => SeekFrom::Current(offset as i64),
            SEEK_END => SeekFrom::End(offset as i64),
            _ => return Err(Error::new(format!("invalid whence {}", whence))),
        };
        self.file
            .seek(pos)
            .map_err(|e| io_err(&format!("seek error on '{}'", self.fname), e))?;
        Ok(())
    }

    fn read_buffered(&mut self, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let offset = self.tell()?;
        if offset > self.size || dst.len() > self.size - offset {
            return Err(Error::new("unexpectedly reached end of file"));
        }
        self.file
            .read_exact(dst)
            .map_err(|e| io_err(&format!("read error on '{}'", self.fname), e))?;
        Ok(())
    }

    /// `read_raw_unsafe()`. With no direct I/O this is the plain buffered read.
    pub fn read_raw_unsafe(&mut self, dst: &mut [u8]) -> Result<()> {
        self.read_buffered(dst)
    }

    /// `read_raw()`. Dispatches to the aligned path only when direct I/O is
    /// active; with the `std::fs` shortcut it always uses the buffered read.
    pub fn read_raw(&mut self, dst: &mut [u8]) -> Result<()> {
        if self.has_direct_io() {
            self.read_aligned_chunk(dst)
        } else {
            self.read_buffered(dst)
        }
    }

    /// `read_aligned_chunk()`. Staged through a bounded buffer, matching the
    /// C++ intent of not holding a large tensor twice. `alignment` is 1 in
    /// this port, so the staging loop degenerates to a plain read.
    pub fn read_aligned_chunk(&mut self, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let offset = self.tell()?;
        if offset > self.size || dst.len() > self.size - offset {
            return Err(Error::new("unexpectedly reached end of file"));
        }

        let alignment = self.read_alignment();
        let aligned_offset = offset & !(alignment - 1);
        let offset_from_alignment = offset - aligned_offset;
        let bytes_to_read = (offset_from_alignment + dst.len() + alignment - 1) & !(alignment - 1);
        let buffer_size = bytes_to_read.min(LLAMA_DIRECT_IO_BUFFER_SIZE);
        let mut buffer = vec![0u8; buffer_size];

        self.seek(aligned_offset, SEEK_SET)?;
        let mut skip = offset_from_alignment;
        let mut copied = 0usize;
        let mut done = 0usize;
        while done < bytes_to_read {
            let n = buffer_size.min(bytes_to_read - done);
            self.file
                .read_exact(&mut buffer[..n])
                .map_err(|e| io_err(&format!("read error on '{}'", self.fname), e))?;
            let chunk_skip = skip.min(n);
            let count = (n - chunk_skip).min(dst.len() - copied);
            dst[copied..copied + count].copy_from_slice(&buffer[chunk_skip..chunk_skip + count]);
            copied += count;
            skip -= chunk_skip;
            done += n;
        }
        self.seek(offset + dst.len(), SEEK_SET)?;
        Ok(())
    }

    /// `read_u32()`. Native byte order, as in the C++ raw copy.
    pub fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_raw(&mut buf)?;
        Ok(u32::from_ne_bytes(buf))
    }

    /// `write_raw()`.
    pub fn write_raw(&mut self, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.file
            .write_all(src)
            .map_err(|e| io_err(&format!("write error on '{}'", self.fname), e))
    }

    /// `write_u32()`. Native byte order.
    pub fn write_u32(&mut self, val: u32) -> Result<()> {
        self.write_raw(&val.to_ne_bytes())
    }

    /// Read a `uint32_t` length followed by that many bytes.
    /// Mirrors `llama_io_read_i::read_string`. Invalid UTF-8 is replaced,
    /// matching `String::from_utf8_lossy` elsewhere in the port.
    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        let mut buf = vec![0u8; len];
        self.read_raw(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Write a `uint32_t` length followed by the string bytes.
    /// Mirrors `llama_io_write_i::write_string`.
    pub fn write_string(&mut self, s: &str) -> Result<()> {
        self.write_u32(s.len() as u32)?;
        self.write_raw(s.as_bytes())
    }

    /// `read_alignment()`. Always 1: `std::fs` cannot query or use the
    /// device's direct-I/O alignment.
    pub fn read_alignment(&self) -> usize {
        self.alignment
    }

    /// `has_direct_io()`. Always false with the `std::fs` shortcut; the
    /// requested flag is kept for API parity.
    pub fn has_direct_io(&self) -> bool {
        self.use_direct_io && self.alignment > 1
    }
}

// ---------------------------------------------------------------------------
// llama_mmap
// ---------------------------------------------------------------------------

/// `llama_mmap::ranges`: `[first, last)` byte ranges within a file.
pub type Ranges = Vec<(usize, usize)>;

/// Merge `ranges` and return their complement within `[0, limit)`.
/// Direct port of `ranges_complement`.
pub fn ranges_complement(ranges: &[(usize, usize)], limit: usize) -> Ranges {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable();

    let mut res = Ranges::new();
    let mut pos = 0usize;
    for &(first, last) in &sorted {
        let beg = first.min(limit);
        let end = last.min(limit);
        if beg > pos {
            res.push((pos, beg));
        }
        pos = pos.max(end);
    }
    if pos < limit {
        res.push((pos, limit));
    }
    res
}

/// `llama_mmaps`.
pub type LlamaMmaps = Vec<LlamaMmap>;

/// Port of `llama_mmap`.
///
/// Shortcut: no OS mapping is created. The whole file is read into `data`,
/// and `mapped_fragments` records which byte ranges are still considered
/// mapped. `unmap_fragment` updates that bookkeeping; it does not release the
/// backing bytes.
pub struct LlamaMmap {
    data: Vec<u8>,
    mapped_fragments: Ranges,
    prefetch_ranges: Ranges,
    size: usize,
}

/// `llama_mmap::SUPPORTED`. The owned-buffer representation always works.
pub const LLAMA_MMAP_SUPPORTED: bool = true;

/// Round `[first, last)` out to whole pages, as `align_range` does.
fn align_range(first: &mut usize, last: &mut usize, page_size: usize) {
    let offset_in_page = *first & (page_size - 1);
    let offset_to_page = if offset_in_page == 0 {
        0
    } else {
        page_size - offset_in_page
    };
    *first += offset_to_page;
    *last &= !(page_size - 1);
    if *last <= *first {
        *last = *first;
    }
}

impl LlamaMmap {
    /// `llama_mmap(file, prefetch, numa, lazy_ranges)`.
    ///
    /// `prefetch == usize::MAX` matches the C++ default `(size_t) -1`. With
    /// `numa` set, prefetch is disabled, as in the original.
    pub fn new(
        file: &mut LlamaFile,
        prefetch: usize,
        numa: bool,
        lazy_ranges: &[(usize, usize)],
    ) -> Result<Self> {
        let size = file.size();
        file.seek(0, SEEK_SET)?;
        let mut data = vec![0u8; size];
        file.read_raw(&mut data)?;

        // With no real mmap, all bytes are already resident. Record the same
        // ranges the C++ code would pass to POSIX_MADV_WILLNEED.
        let effective_prefetch = if numa { 0 } else { prefetch };
        let prefetch_ranges = if effective_prefetch > 0 {
            ranges_complement(lazy_ranges, size.min(effective_prefetch))
        } else {
            Ranges::new()
        };

        Ok(Self {
            data,
            mapped_fragments: vec![(0, size)],
            prefetch_ranges,
            size,
        })
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.size
    }

    /// `addr()`. Returns the owned bytes as a slice instead of a raw pointer.
    pub fn addr(&self) -> &[u8] {
        &self.data
    }

    /// Alias for `addr()` with a Rust name.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Ranges the C++ code would prefetch with `POSIX_MADV_WILLNEED`.
    pub fn prefetch_ranges(&self) -> &[(usize, usize)] {
        &self.prefetch_ranges
    }

    /// Ranges still marked as mapped.
    pub fn mapped_fragments(&self) -> &[(usize, usize)] {
        &self.mapped_fragments
    }

    /// `unmap_fragment(first, last)`. Aligns the range to a page boundary and
    /// removes it from `mapped_fragments`, using the same split logic as C++.
    /// The owned bytes are kept, so this is bookkeeping only.
    pub fn unmap_fragment(&mut self, first: usize, last: usize) {
        let (mut first, mut last) = (first, last);
        align_range(&mut first, &mut last, PAGE_SIZE);
        if last <= first {
            return;
        }

        let mut next = Ranges::new();
        for &(frag_first, frag_last) in &self.mapped_fragments {
            if frag_first < first && frag_last > last {
                next.push((frag_first, first));
                next.push((last, frag_last));
            } else if frag_first < first && frag_last > first {
                next.push((frag_first, first));
            } else if frag_first < last && frag_last > last {
                next.push((last, frag_last));
            } else if frag_first >= first && frag_last <= last {
                // fully covered: drop
            } else {
                next.push((frag_first, frag_last));
            }
        }
        self.mapped_fragments = next;
    }
}

// ---------------------------------------------------------------------------
// llama_mlock
// ---------------------------------------------------------------------------

/// `llama_mlocks`.
pub type LlamaMlocks = Vec<LlamaMlock>;

/// Port of `llama_mlock`.
///
/// Shortcut: no pages are actually locked. `grow_to` rounds the target up to
/// the lock granularity and records it, leaving the rest of the state machine
/// identical to the C++ version.
pub struct LlamaMlock {
    addr: usize,
    size: usize,
    failed_already: bool,
}

/// `llama_mlock::SUPPORTED`. The recorded-state representation always works.
pub const LLAMA_MLOCK_SUPPORTED: bool = true;

impl LlamaMlock {
    /// `llama_mlock()`.
    pub fn new() -> Self {
        Self {
            addr: 0,
            size: 0,
            failed_already: false,
        }
    }

    /// `lock_granularity()`. C++ uses the page size on Unix/Windows and
    /// 65536 on unsupported systems.
    pub fn lock_granularity() -> usize {
        PAGE_SIZE
    }

    /// `init(ptr)`. The address is a usize so no raw pointer or `unsafe` is
    /// needed. C++ asserts the lock is fresh; this does the same.
    pub fn init(&mut self, addr: usize) {
        assert_eq!(self.addr, 0, "llama_mlock::init called twice");
        assert_eq!(self.size, 0, "llama_mlock::init called twice");
        self.addr = addr;
    }

    /// `grow_to(target_size)`. Rounds up to the granularity and records the
    /// locked size. A no-op after a failed lock, as in C++.
    pub fn grow_to(&mut self, target_size: usize) {
        if self.addr == 0 || self.failed_already {
            return;
        }
        let granularity = Self::lock_granularity();
        let target = (target_size + granularity - 1) & !(granularity - 1);
        if target > self.size {
            self.size = target;
        }
    }

    /// Number of bytes currently recorded as locked.
    pub fn locked_size(&self) -> usize {
        self.size
    }

    /// Whether a previous grow was abandoned.
    pub fn failed(&self) -> bool {
        self.failed_already
    }
}

impl Default for LlamaMlock {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// llama_prefetch
// ---------------------------------------------------------------------------

/// `llama_memory_range`. `addr` is a usize instead of `const void *`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRange {
    pub addr: usize,
    pub size: usize,
}

/// `llama_memory_ranges`.
pub type MemoryRanges = Vec<MemoryRange>;

/// `llama_prefetch(mr)`. Merges overlapping page-aligned ranges in place.
/// The OS prefetch (`madvise`/`PrefetchVirtualMemory`) is skipped: `std` has
/// no safe API for it, and the owned-buffer mapping is already resident.
pub fn llama_prefetch(mr: &mut MemoryRanges) {
    if mr.is_empty() {
        return;
    }

    let page = PAGE_SIZE;
    let input = std::mem::take(mr);
    let mut merged = input.clone();
    let mut begin = 0usize;
    let mut end = 0usize;
    let mut n = 0usize;

    let mut sorted = input;
    sorted.sort_unstable_by_key(|r| r.addr);
    for range in &sorted {
        if range.addr == 0 || range.size == 0 {
            continue;
        }
        let pointer = range.addr;
        let first = pointer / page * page;
        let last = pointer
            .saturating_add(range.size)
            .saturating_add(page - 1)
            / page
            * page;
        if end != 0 && first > end {
            merged[n] = MemoryRange {
                addr: begin,
                size: end - begin,
            };
            n += 1;
            end = 0;
        }
        if end == 0 {
            begin = first;
        }
        end = end.max(last);
    }
    if end != 0 {
        merged[n] = MemoryRange {
            addr: begin,
            size: end - begin,
        };
        n += 1;
    }
    merged.truncate(n);
    *mr = merged;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("llama_mmap_test_{}_{}", std::process::id(), name));
        path
    }

    #[test]
    fn file_u32_and_string_round_trip() {
        let path = temp_path("roundtrip");
        {
            let mut file = LlamaFile::open(path.to_str().unwrap(), "wb", false).unwrap();
            file.write_u32(0x1234_5678).unwrap();
            file.write_string("hello").unwrap();
        }
        {
            let mut file = LlamaFile::open(path.to_str().unwrap(), "rb", false).unwrap();
            assert_eq!(file.read_u32().unwrap(), 0x1234_5678);
            assert_eq!(file.read_string().unwrap(), "hello");
            assert_eq!(file.tell().unwrap(), 4 + 4 + 5);
            assert_eq!(file.size(), 4 + 4 + 5);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn seek_and_read_raw() {
        let path = temp_path("seek");
        std::fs::write(&path, [1u8, 2, 3, 4, 5]).unwrap();
        let mut file = LlamaFile::open(path.to_str().unwrap(), "rb", false).unwrap();
        let mut buf = [0u8; 2];
        file.read_raw(&mut buf).unwrap();
        assert_eq!(buf, [1, 2]);
        file.seek((-1i64) as usize, SEEK_CUR).unwrap();
        file.read_raw(&mut buf).unwrap();
        assert_eq!(buf, [2, 3]);
        file.seek(0, SEEK_END).unwrap();
        assert_eq!(file.tell().unwrap(), 5);
        assert!(file.read_raw(&mut buf).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ranges_complement_finds_gaps() {
        let got = ranges_complement(&[(10, 20), (30, 40)], 50);
        assert_eq!(got, vec![(0, 10), (20, 30), (40, 50)]);
    }

    #[test]
    fn mmap_reads_file_and_tracks_unmap() {
        let path = temp_path("mmap");
        std::fs::write(&path, vec![7u8; 8192]).unwrap();
        let mut file = LlamaFile::open(path.to_str().unwrap(), "rb", false).unwrap();
        let mut map = LlamaMmap::new(&mut file, usize::MAX, false, &[]).unwrap();
        assert_eq!(map.size(), 8192);
        assert_eq!(map.addr().len(), 8192);
        assert_eq!(map.addr()[0], 7);
        map.unmap_fragment(4096, 8192);
        assert_eq!(map.mapped_fragments(), &[(0, 4096)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mmap_prefetch_skips_lazy_ranges() {
        let path = temp_path("prefetch");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let mut file = LlamaFile::open(path.to_str().unwrap(), "rb", false).unwrap();
        let map = LlamaMmap::new(&mut file, 8192, false, &[(0, 4096)]).unwrap();
        assert_eq!(map.prefetch_ranges(), &[(4096, 8192)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mlock_grows_in_page_steps() {
        let mut lock = LlamaMlock::new();
        lock.init(0x1000);
        lock.grow_to(1);
        assert_eq!(lock.locked_size(), PAGE_SIZE);
        lock.grow_to(PAGE_SIZE + 1);
        assert_eq!(lock.locked_size(), 2 * PAGE_SIZE);
    }

    #[test]
    fn prefetch_merges_overlapping_ranges() {
        let mut ranges = vec![
            MemoryRange { addr: 100, size: 100 },
            MemoryRange { addr: 4000, size: 200 },
        ];
        llama_prefetch(&mut ranges);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].addr, 0);
        assert_eq!(ranges[0].size, 2 * PAGE_SIZE);
    }
}
