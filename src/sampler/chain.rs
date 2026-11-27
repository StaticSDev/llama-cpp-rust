use crate::core::{Error, Result, Token};
use std::time::Instant;

use super::candidate::{
    cmp_logit, ChainParams, TokenData, TokenDataArray, LLAMA_DEFAULT_SEED,
};

pub trait Sampler {
    fn name(&self) -> &str;
    fn apply(&mut self, cur: &mut TokenDataArray);
    fn accept(&mut self, _token: Token) {}
    fn reset(&mut self) {}
    fn clone_sampler(&self) -> Box<dyn Sampler>;

    /// `seed_cur` for samplers that own an RNG; used by `llama_sampler_get_seed`.
    fn seed_cur(&self) -> Option<u32> {
        None
    }

    /// `llama_sampler_backend_begin` state reset.
    fn backend_begin(&mut self) {}
}

pub struct RingBuffer<T: Clone> {
    capacity: usize,
    sz: usize,
    first: usize,
    pos: usize,
    data: Vec<Option<T>>,
}

impl<T: Clone> Clone for RingBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            sz: self.sz,
            first: self.first,
            pos: self.pos,
            data: self.data.clone(),
        }
    }
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            sz: 0,
            first: 0,
            pos: 0,
            data: (0..capacity).map(|_| None).collect(),
        }
    }

    pub fn front(&self) -> T {
        assert!(self.sz != 0, "ring buffer is empty");
        self.data[self.first].clone().expect("ring buffer slot is empty")
    }

    pub fn back(&self) -> T {
        assert!(self.sz != 0, "ring buffer is empty");
        self.data[self.pos].clone().expect("ring buffer slot is empty")
    }

    pub fn push_back(&mut self, value: T) {
        assert!(self.capacity != 0, "ring buffer: capacity is zero");
        if self.sz == self.capacity {
            // advance the start when buffer is full
            self.first = (self.first + 1) % self.capacity;
        } else {
            self.sz += 1;
        }
        self.data[self.pos] = Some(value);
        self.pos = (self.pos + 1) % self.capacity;
    }

    pub fn pop_front(&mut self) -> T {
        assert!(self.sz != 0, "ring buffer is empty");
        let value = self.data[self.first].clone().expect("ring buffer slot is empty");
        self.first = (self.first + 1) % self.capacity;
        self.sz -= 1;
        value
    }

    /// Reverse index: `0` is the most recently pushed element.
    pub fn rat(&self, i: usize) -> T {
        assert!(i < self.sz, "ring buffer: index out of bounds");
        let idx = (self.first + self.sz - i - 1) % self.capacity;
        self.data[idx].clone().expect("ring buffer slot is empty")
    }

    pub fn to_vector(&self) -> Vec<T> {
        (0..self.sz)
            .map(|i| {
                self.data[(self.first + i) % self.capacity]
                    .clone()
                    .expect("ring buffer slot is empty")
            })
            .collect()
    }

    pub fn clear(&mut self) {
        // here only reset the status of the buffer
        self.sz = 0;
        self.first = 0;
        self.pos = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.sz == 0
    }

    pub fn len(&self) -> usize {
        self.sz
    }
}

/// Writes result in res, does not mutate cur. Bucket-based partial sort.
pub(crate) fn partial_sort(cur: &TokenDataArray, npartial_in: usize) -> Vec<TokenData> {
    const NBUCKETS: usize = 128;
    const BUCKET_LOW: f32 = -10.0;
    const BUCKET_HIGH: f32 = 10.0;
    const BUCKET_SCALE: f32 = NBUCKETS as f32 / (BUCKET_HIGH - BUCKET_LOW);
    const BUCKET_INTER: f32 = -BUCKET_LOW * BUCKET_SCALE;

    let npartial = npartial_in.min(cur.data.len());

    let mut bucket_idx = Vec::with_capacity(cur.data.len());
    let mut histo = vec![0usize; NBUCKETS];

    for td in &cur.data {
        let mut ib = (BUCKET_SCALE * td.logit + BUCKET_INTER) as i32;
        ib = ib.max(0).min((NBUCKETS - 1) as i32);
        bucket_idx.push(ib as usize);
        histo[ib as usize] += 1;
    }

    let mut nhave = 0usize;
    let mut ib = NBUCKETS as i32 - 1;
    while ib >= 0 {
        nhave += histo[ib as usize];
        if nhave >= npartial {
            break;
        }
        ib -= 1;
    }
    let ibu = ib as usize;

    let mut res = vec![TokenData::default(); nhave];
    let mut bucket_ptrs = Vec::with_capacity(NBUCKETS - ibu);
    let mut ptr = 0usize;
    for j in (ibu..NBUCKETS).rev() {
        bucket_ptrs.push(ptr);
        ptr += histo[j];
    }
    for (i, td) in cur.data.iter().enumerate() {
        let j = bucket_idx[i];
        if j >= ibu {
            let slot = &mut bucket_ptrs[NBUCKETS - 1 - j];
            res[*slot] = *td;
            *slot += 1;
        }
    }

    let mut ptr = 0usize;
    let mut ndone = 0usize;
    for j in (ibu + 1..NBUCKETS).rev() {
        res[ptr..ptr + histo[j]].sort_unstable_by(cmp_logit);
        ptr += histo[j];
        ndone += histo[j];
    }
    let kth = npartial - ndone;
    if kth > 0 {
        res[ptr..ptr + kth].select_nth_unstable_by(kth - 1, cmp_logit);
        res[ptr..ptr + kth].sort_unstable_by(cmp_logit);
    }

    res
}

/// Reduces cur_p to npartial, keeping only the top npartial elements.
pub(crate) fn partial_sort_inplace(cur: &mut TokenDataArray, npartial: usize) {
    if npartial <= 128 {
        let n = npartial.min(cur.data.len());
        if n > 0 {
            cur.data.select_nth_unstable_by(n - 1, cmp_logit);
            cur.data[..n].sort_unstable_by(cmp_logit);
        }
        cur.data.truncate(n);
        cur.sorted = true;
        return;
    }

    let tmp = partial_sort(cur, npartial);
    let n = npartial.min(cur.data.len());
    for i in 0..n {
        cur.data[i] = tmp[i];
    }
    cur.data.truncate(n);
    cur.sorted = true;
}

struct ChainEntry {
    is_backend: bool,
    sampler: Box<dyn Sampler>,
}

/// `llama_sampler_chain`.
pub struct Chain {
    params: ChainParams,
    is_init: bool,
    #[allow(dead_code)]
    n_nodes: u32,
    samplers: Vec<ChainEntry>,
    cur: Vec<TokenData>,
    t_sample_us: i64,
    n_sample: i32,
}

impl Chain {
    pub fn new(params: ChainParams) -> Self {
        Self {
            params,
            is_init: false,
            n_nodes: 0,
            samplers: Vec::new(),
            cur: Vec::new(),
            t_sample_us: 0,
            n_sample: 0,
        }
    }

    pub fn add(&mut self, sampler: Box<dyn Sampler>) {
        self.samplers.push(ChainEntry {
            is_backend: false,
            sampler,
        });
    }

    pub fn get(&self, i: i32) -> Option<&dyn Sampler> {
        if i == -1 {
            return Some(self);
        }
        if i < 0 || i as usize >= self.samplers.len() {
            return None;
        }
        Some(self.samplers[i as usize].sampler.as_ref())
    }

    pub fn remove(&mut self, i: i32) -> Option<Box<dyn Sampler>> {
        if i < 0 || i as usize >= self.samplers.len() {
            return None;
        }
        Some(self.samplers.remove(i as usize).sampler)
    }

    pub fn len(&self) -> i32 {
        self.samplers.len() as i32
    }

    pub fn is_empty(&self) -> bool {
        self.samplers.is_empty()
    }

    pub fn clone_chain(&self) -> Self {
        // note: is_init, n_nodes and is_backend belong to the current graph
        let mut result = Self::new(self.params);
        for entry in &self.samplers {
            result.samplers.push(ChainEntry {
                is_backend: false,
                sampler: entry.sampler.clone_sampler(),
            });
        }
        result
    }

    pub fn copy_state(&mut self, src: &Chain) {
        assert!(
            self.samplers.len() == src.samplers.len(),
            "chain state: sampler count mismatch"
        );
        for (dst, smpl) in self.samplers.iter_mut().zip(src.samplers.iter()) {
            copy_sampler(smpl.sampler.as_ref(), &mut dst.sampler).expect("chain state: copy failed");
        }
        self.params = src.params;
        self.cur = src.cur.clone();
        self.t_sample_us = src.t_sample_us;
        self.n_sample = src.n_sample;
    }

    /// Fill the pre-allocated buffer with all vocabulary logits and sample.
    pub fn sample(&mut self, logits: &[f32]) -> Result<Token> {
        self.cur.clear();
        for (id, logit) in logits.iter().enumerate() {
            self.cur.push(TokenData {
                id: id as Token,
                logit: *logit,
                p: 0.0,
            });
        }
        self.sample_cur()
    }

    /// Sample from an already-built candidate list.
    pub fn sample_with_data(&mut self, candidates: Vec<TokenData>) -> Result<Token> {
        self.cur = candidates;
        self.sample_cur()
    }

    fn sample_cur(&mut self) -> Result<Token> {
        let mut arr = TokenDataArray {
            data: std::mem::take(&mut self.cur),
            selected: -1,
            sorted: false,
        };
        self.apply(&mut arr);
        let selected = arr.selected;
        self.cur = arr.data;

        if selected < 0 || selected as usize >= self.cur.len() {
            return Err(Error::new("llama_sampler_sample: invalid selected index"));
        }
        let token = self.cur[selected as usize].id;
        self.accept(token);
        Ok(token)
    }
}

pub fn init_chain(params: ChainParams) -> Chain {
    Chain::new(params)
}

pub fn chain_add(chain: &mut Chain, sampler: Box<dyn Sampler>) {
    chain.add(sampler);
}

pub fn chain_get<'a>(chain: &'a Chain, i: i32) -> Option<&'a dyn Sampler> {
    chain.get(i)
}

pub fn chain_remove(chain: &mut Chain, i: i32) -> Option<Box<dyn Sampler>> {
    chain.remove(i)
}

pub fn chain_n(chain: &Chain) -> i32 {
    chain.len()
}

impl Sampler for Chain {
    fn name(&self) -> &str {
        "chain"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        let started = if self.params.no_perf {
            None
        } else {
            Some(Instant::now())
        };

        let mut is_backend = self.is_init;
        for entry in &mut self.samplers {
            if is_backend && entry.is_backend {
                continue;
            }
            is_backend = false;
            entry.sampler.apply(cur);
        }

        if let Some(started) = started {
            self.t_sample_us += started.elapsed().as_micros() as i64;
        }
    }

    fn accept(&mut self, token: Token) {
        let started = if self.params.no_perf {
            None
        } else {
            Some(Instant::now())
        };

        for entry in &mut self.samplers {
            entry.sampler.accept(token);
        }
        self.n_sample += 1;

        if let Some(started) = started {
            self.t_sample_us += started.elapsed().as_micros() as i64;
        }
    }

    fn reset(&mut self) {
        for entry in &mut self.samplers {
            entry.sampler.reset();
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone_chain())
    }

    fn seed_cur(&self) -> Option<u32> {
        for entry in self.samplers.iter().rev() {
            if let Some(seed) = entry.sampler.seed_cur() {
                if seed != LLAMA_DEFAULT_SEED {
                    return Some(seed);
                }
            }
        }
        None
    }

    fn backend_begin(&mut self) {
        for entry in &mut self.samplers {
            if !entry.is_backend {
                break;
            }
            entry.sampler.backend_begin();
        }
    }
}

/// `llama_sampler_sample` for a non-chain sampler with an explicit list.
pub fn sample(smpl: &mut dyn Sampler, cur: Vec<TokenData>) -> Result<Token> {
    let mut arr = TokenDataArray::new(cur);
    smpl.apply(&mut arr);
    if arr.selected < 0 || arr.selected as usize >= arr.data.len() {
        return Err(Error::new("llama_sampler_sample: invalid selected index"));
    }
    let token = arr.data[arr.selected as usize].id;
    smpl.accept(token);
    Ok(token)
}

pub fn get_seed(smpl: &dyn Sampler) -> u32 {
    smpl.seed_cur().unwrap_or(LLAMA_DEFAULT_SEED)
}

pub fn copy_sampler(src: &dyn Sampler, dst: &mut Box<dyn Sampler>) -> Result<()> {
    if src.name() != dst.name() {
        return Err(Error::new(
            "llama_sampler_copy: cannot copy between different sampler types",
        ));
    }
    *dst = src.clone_sampler();
    Ok(())
}

pub fn backend_begin(smpl: &mut dyn Sampler) {
    smpl.backend_begin();
}

// ---------------------------------------------------------------------------
// perf
// ---------------------------------------------------------------------------

/// `llama_perf_sampler_data`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerfSamplerData {
    pub t_sample_ms: f64,
    pub n_sample: i32,
}

pub fn perf_sampler(chain: &Chain) -> PerfSamplerData {
    PerfSamplerData {
        t_sample_ms: 1e-3 * chain.t_sample_us as f64,
        n_sample: chain.n_sample.max(0),
    }
}

pub fn perf_sampler_reset(chain: &mut Chain) {
    chain.t_sample_us = 0;
    chain.n_sample = 0;
}
