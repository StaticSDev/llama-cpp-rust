//! Runtime-generated IQ codebooks and neighbour searches.
//!
//! `shortcut: codebooks are initialized lazily on first use instead of
//! requiring an explicit `ggml_quantize_init()` call; `*_free_impl` clears
//! the cached entry.`

use std::sync::{Arc, Mutex};

use crate::core::{Error, Result};
use crate::ggml::quants::tables::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Iq2Kind {
    Iq2Xxs,
    Iq2Xs,
    Iq1S,
    Iq1M,
    Iq2S,
}

impl Iq2Kind {
    pub fn index(self) -> usize {
        match self {
            Iq2Kind::Iq2Xxs => 0,
            Iq2Kind::Iq2Xs => 1,
            Iq2Kind::Iq1S | Iq2Kind::Iq1M => 2,
            Iq2Kind::Iq2S => 3,
        }
    }

    pub fn grid_size(self) -> usize {
        match self {
            Iq2Kind::Iq2Xxs => 256,
            Iq2Kind::Iq2Xs => 512,
            Iq2Kind::Iq1S | Iq2Kind::Iq1M => NGRID_IQ1S,
            Iq2Kind::Iq2S => 1024,
        }
    }

    pub fn nwant(self) -> usize {
        match self {
            Iq2Kind::Iq1S | Iq2Kind::Iq1M => 3,
            Iq2Kind::Iq2S => 1,
            _ => 2,
        }
    }

    fn kgrid(self) -> &'static [u16] {
        match self {
            Iq2Kind::Iq2Xxs => &KGRID_2BIT_256,
            Iq2Kind::Iq2Xs => &KGRID_2BIT_512,
            Iq2Kind::Iq1S | Iq2Kind::Iq1M => &KGRID_1BIT_2048,
            Iq2Kind::Iq2S => &KGRID_2BIT_1024,
        }
    }
}

const NGRID_IQ1S: usize = 2048;
const IQ2_KMAP_SIZE: usize = 43692;
const IQ3_KMAP_SIZE: usize = 4096;

#[derive(Clone, Debug)]
pub struct Iq2Data {
    pub grid: Vec<u64>,
    pub map: Vec<i32>,
    pub neighbours: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct Iq3Data {
    pub grid: Vec<u64>,
    pub map: Vec<i32>,
    pub neighbours: Vec<u16>,
}

static IQ2_DATA: Mutex<[Option<Arc<Iq2Data>>; 4]> = Mutex::new([None, None, None, None]);
static IQ3_DATA: Mutex<[Option<Arc<Iq3Data>>; 2]> = Mutex::new([None, None]);

fn build_codebook(
    kgrid: &[u16],
    grid_size: usize,
    kmap_size: usize,
    dims: usize,
    bits: u32,
    nwant: usize,
) -> (Vec<u64>, Vec<i32>, Vec<u16>) {
    let mut grid = vec![0u64; grid_size];
    for k in 0..grid_size {
        let mut bytes = [0u8; 8];
        for i in 0..dims {
            let l = (kgrid[k] >> (bits * i as u32)) & ((1u16 << bits) - 1);
            bytes[i] = (2 * l + 1) as u8;
        }
        grid[k] = u64::from_le_bytes(bytes);
    }
    let mut map = vec![-1i32; kmap_size];
    for i in 0..grid_size {
        let bytes = grid[i].to_le_bytes();
        let mut index = 0u16;
        for k in 0..dims {
            let q = ((bytes[k] as i16 - 1) / 2) as u16;
            index |= q << (bits * k as u32);
        }
        map[index as usize] = i as i32;
    }
    let mut n_per_i = vec![0usize; kmap_size];
    let mut num_neighbors = 0usize;
    let mut num_not_in_map = 0usize;
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        num_not_in_map += 1;
        let mut pos = [0i32; 8];
        for k in 0..dims {
            let l = (i >> (bits * k as u32)) & ((1 << bits) - 1);
            pos[k] = 2 * l as i32 + 1;
        }
        let mut dist2: Vec<(i32, usize)> = (0..grid_size)
            .map(|j| {
                let g = grid[j].to_le_bytes();
                let d: i32 = (0..dims)
                    .map(|k| {
                        let diff = g[k] as i32 - pos[k];
                        diff * diff
                    })
                    .sum();
                (d, j)
            })
            .collect();
        dist2.sort_unstable();
        let mut n = 0usize;
        let mut d2 = dist2[0].0;
        let mut nhave = 1usize;
        for (dv, _) in dist2.iter() {
            if *dv > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = *dv;
                nhave += 1;
            }
            n += 1;
        }
        n_per_i[i] = n;
        num_neighbors += n;
    }
    let mut neighbours = vec![0u16; num_neighbors + num_not_in_map];
    let mut offsets = vec![-1i32; kmap_size];
    let mut counter = 0usize;
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        offsets[i] = counter as i32;
        counter += 1 + n_per_i[i];
    }
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        let mut pos = [0i32; 8];
        for k in 0..dims {
            let l = (i >> (bits * k as u32)) & ((1 << bits) - 1);
            pos[k] = 2 * l as i32 + 1;
        }
        let mut dist2: Vec<(i32, usize)> = (0..grid_size)
            .map(|j| {
                let g = grid[j].to_le_bytes();
                let d: i32 = (0..dims)
                    .map(|k| {
                        let diff = g[k] as i32 - pos[k];
                        diff * diff
                    })
                    .sum();
                (d, j)
            })
            .collect();
        dist2.sort_unstable();
        let local_counter = offsets[i] as usize;
        map[i] = -(local_counter as i32 + 1);
        let mut d2 = dist2[0].0;
        neighbours[local_counter] = 0;
        let mut nhave = 1usize;
        let mut n = 0usize;
        for (dv, j) in dist2.iter() {
            if *dv > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = *dv;
                nhave += 1;
            }
            neighbours[local_counter + 1 + n] = *j as u16;
            n += 1;
        }
        neighbours[local_counter] = n as u16;
    }
    (grid, map, neighbours)
}

pub fn iq2xs_init_impl(kind: Iq2Kind) {
    let (grid, map, neighbours) = build_codebook(kind.kgrid(), kind.grid_size(), IQ2_KMAP_SIZE, 8, 2, kind.nwant());
    let data = Arc::new(Iq2Data { grid, map, neighbours });
    if let Ok(mut all) = IQ2_DATA.lock() {
        all[kind.index()] = Some(data);
    }
}

pub fn iq2xs_free_impl(kind: Iq2Kind) {
    if let Ok(mut all) = IQ2_DATA.lock() {
        all[kind.index()] = None;
    }
}

pub fn iq2_data(kind: Iq2Kind) -> Result<Arc<Iq2Data>> {
    if let Some(data) = IQ2_DATA
        .lock()
        .ok()
        .and_then(|all| all[kind.index()].clone())
    {
        return Ok(data);
    }
    iq2xs_init_impl(kind);
    IQ2_DATA
        .lock()
        .ok()
        .and_then(|all| all[kind.index()].clone())
        .ok_or_else(|| Error::new("iq2 codebook initialization failed"))
}

pub fn iq3xs_init_impl(grid_size: usize) {
    let (kgrid, nwant) = match grid_size {
        256 => (&KGRID_256[..], 2),
        512 => (&KGRID_512[..], 3),
        _ => return,
    };
    let (grid, map, neighbours) = build_codebook(kgrid, grid_size, IQ3_KMAP_SIZE, 4, 3, nwant);
    let data = Arc::new(Iq3Data { grid, map, neighbours });
    if let Ok(mut all) = IQ3_DATA.lock() {
        all[if grid_size == 256 { 0 } else { 1 }] = Some(data);
    }
}

pub fn iq3xs_free_impl(grid_size: usize) {
    if let Ok(mut all) = IQ3_DATA.lock() {
        all[if grid_size == 256 { 0 } else { 1 }] = None;
    }
}

pub fn iq3_data(grid_size: usize) -> Result<Arc<Iq3Data>> {
    let idx = if grid_size == 256 { 0 } else { 1 };
    if let Some(data) = IQ3_DATA.lock().ok().and_then(|all| all[idx].clone()) {
        return Ok(data);
    }
    iq3xs_init_impl(grid_size);
    IQ3_DATA
        .lock()
        .ok()
        .and_then(|all| all[idx].clone())
        .ok_or_else(|| Error::new("iq3 codebook initialization failed"))
}

#[inline]
pub fn levels8(word: u64) -> [i8; 8] {
    word.to_le_bytes().map(|b| b as i8)
}

#[inline]
pub fn levels4(word: u64) -> [i8; 4] {
    let b = word.to_le_bytes();
    [b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8]
}

/// Neighbour list for a pattern not on the grid: first element is the count.
pub fn neighbours_of(data: &Iq2Data, pattern: usize) -> &[u16] {
    let m = data.map[pattern];
    debug_assert!(m < 0);
    let offset = (-m - 1) as usize;
    let count = data.neighbours[offset] as usize;
    &data.neighbours[offset..offset + 1 + count]
}

pub fn neighbours_of_iq3(data: &Iq3Data, pattern: usize) -> &[u16] {
    let m = data.map[pattern];
    debug_assert!(m < 0);
    let offset = (-m - 1) as usize;
    let count = data.neighbours[offset] as usize;
    &data.neighbours[offset..offset + 1 + count]
}

pub fn iq2_find_best_neighbour(
    data: &Iq2Data,
    neighbours: &[u16],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> usize {
    let mut best_d2 = f32::MAX;
    let mut grid_index = 0usize;
    let mut found = false;
    for j in 1..=neighbours[0] as usize {
        let levels = levels8(data.grid[neighbours[j] as usize]);
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let diff = scale * levels[i] as f32 - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = neighbours[j] as usize;
            found = true;
        }
    }
    debug_assert!(found);
    let levels = levels8(data.grid[grid_index]);
    for i in 0..8 {
        l[i] = (levels[i] - 1) / 2;
    }
    grid_index
}

pub const IQ2_NEIGHBOUR_CACHE_MAX: usize = 512;

pub struct Iq2NeighbourCache {
    pub pattern: i32,
    pub last_best: usize,
    pub sum_w: f64,
    pub sum_wx: f64,
    pub sum_wxx: f64,
    pub sum_w1: f64,
    pub sum_w1x: f64,
    pub sum_w1xx: f64,
    pub lin_term: [[f64; 8]; 8],
    pub quad_term: [[f64; 8]; 8],
    pub lin_coef: [f64; IQ2_NEIGHBOUR_CACHE_MAX],
    pub quad_coef: [f64; IQ2_NEIGHBOUR_CACHE_MAX],
}

impl Iq2NeighbourCache {
    pub fn new() -> Self {
        Self {
            pattern: -1,
            last_best: 0,
            sum_w: 0.0,
            sum_wx: 0.0,
            sum_wxx: 0.0,
            sum_w1: 0.0,
            sum_w1x: 0.0,
            sum_w1xx: 0.0,
            lin_term: [[0.0; 8]; 8],
            quad_term: [[0.0; 8]; 8],
            lin_coef: [0.0; IQ2_NEIGHBOUR_CACHE_MAX],
            quad_coef: [0.0; IQ2_NEIGHBOUR_CACHE_MAX],
        }
    }

    pub fn init(&mut self, xval: &[f32], weight: &[f32]) {
        self.pattern = -1;
        self.sum_w = 0.0;
        self.sum_wx = 0.0;
        self.sum_wxx = 0.0;
        self.sum_w1 = 0.0;
        self.sum_w1x = 0.0;
        self.sum_w1xx = 0.0;
        for i in 0..8 {
            let w = weight[i] as f64;
            let x = xval[i] as f64;
            let abs_w = w.abs();
            let abs_x = x.abs();
            self.sum_w += abs_w;
            self.sum_wx += abs_w * abs_x;
            self.sum_wxx += abs_w * abs_x * abs_x;
            self.sum_w1 += 1.0 + abs_w;
            self.sum_w1x += (1.0 + abs_w) * abs_x;
            self.sum_w1xx += (1.0 + abs_w) * abs_x * abs_x;
            for q in 0..8 {
                self.lin_term[i][q] = 2.0 * w * x * q as f64;
                self.quad_term[i][q] = w * q as f64 * q as f64;
            }
        }
    }
}

fn iq2_count_at_most(values: &[f64; IQ2_NEIGHBOUR_CACHE_MAX], n: usize, limit: f64) -> usize {
    let mut count = 0;
    for j in 0..n {
        count += (values[j] <= limit) as usize;
    }
    count
}

pub fn iq2_find_best_neighbour_cached(
    cache: &mut Iq2NeighbourCache,
    pattern: usize,
    neighbours: &[u16],
    data: &Iq2Data,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> usize {
    let num_neighbors = neighbours[0] as usize;
    let scale_d = scale as f64;
    let max_level = 7.0 * scale_d;
    let size = max_level * max_level * cache.sum_w
        + 2.0 * max_level * cache.sum_wx
        + cache.sum_wxx;
    let size_subnormal = (1.0 + max_level) * (1.0 + max_level) * cache.sum_w1
        + 2.0 * (1.0 + max_level) * cache.sum_w1x
        + cache.sum_w1xx;
    let tolerance = (2.0f64.powi(-18) * size + 2.0f64.powi(-100) * size_subnormal) / scale_d;
    if num_neighbors == 0
        || num_neighbors > IQ2_NEIGHBOUR_CACHE_MAX
        || !(scale_d > 0.0
            && size < 2.0f64.powi(100)
            && size_subnormal < 2.0f64.powi(100)
            && tolerance > 0.0)
    {
        return iq2_find_best_neighbour(data, neighbours, xval, weight, scale, l);
    }
    if cache.pattern != pattern as i32 {
        for j in 0..num_neighbors {
            let levels = levels8(data.grid[neighbours[j + 1] as usize]);
            let mut lin = 0.0f64;
            let mut quad = 0.0f64;
            for i in 0..8 {
                lin += cache.lin_term[i][levels[i] as usize];
                quad += cache.quad_term[i][levels[i] as usize];
            }
            cache.lin_coef[j] = lin;
            cache.quad_coef[j] = quad;
        }
        cache.pattern = pattern as i32;
        cache.last_best = 0;
    }
    let mut score = [0.0f64; IQ2_NEIGHBOUR_CACHE_MAX];
    for j in 0..num_neighbors {
        score[j] = scale_d * cache.quad_coef[j] - cache.lin_coef[j];
    }
    let mut best = cache.last_best;
    if iq2_count_at_most(&score, num_neighbors, score[best] + tolerance) != 1 {
        for j in 0..num_neighbors {
            if score[j] < score[best] {
                best = j;
            }
        }
        if iq2_count_at_most(&score, num_neighbors, score[best] + tolerance) != 1 {
            return iq2_find_best_neighbour(data, neighbours, xval, weight, scale, l);
        }
        cache.last_best = best;
    }
    let grid_index = neighbours[best + 1] as usize;
    let levels = levels8(data.grid[grid_index]);
    for i in 0..8 {
        l[i] = (levels[i] - 1) / 2;
    }
    grid_index
}

pub fn iq3_find_best_neighbour(
    data: &Iq3Data,
    neighbours: &[u16],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> usize {
    let mut best_d2 = f32::MAX;
    let mut grid_index = 0usize;
    let mut found = false;
    for j in 1..=neighbours[0] as usize {
        let levels = levels4(data.grid[neighbours[j] as usize]);
        let mut d2 = 0.0f32;
        for i in 0..4 {
            let diff = scale * levels[i] as f32 - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = neighbours[j] as usize;
            found = true;
        }
    }
    debug_assert!(found);
    let levels = levels4(data.grid[grid_index]);
    for i in 0..4 {
        l[i] = (levels[i] - 1) / 2;
    }
    grid_index
}

#[allow(dead_code)]
pub fn iq1_find_best_neighbour(
    data: &Iq2Data,
    neighbours: &[u16],
    xval: &[f32],
    weight: &[f32],
    scale: &mut f32,
    l: &mut [i8],
) -> usize {
    let mut best_score = -f32::MAX;
    let mut grid_index = -1i32;
    for j in 1..=neighbours[0] as usize {
        let levels = levels8(data.grid[neighbours[j] as usize]);
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for i in 0..8 {
            let q = (levels[i] - 3) as f32 * 0.5;
            sumqx += weight[i] * q * xval[i];
            sumq2 += weight[i] * q * q;
        }
        if sumqx > 0.0 && sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
            *scale = sumqx / sumq2;
            best_score = *scale * sumqx;
            grid_index = neighbours[j] as i32;
        }
    }
    if grid_index < 0 {
        for i in 0..data.grid.len() {
            let levels = levels8(data.grid[i]);
            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for j in 0..8 {
                let q = (levels[j] - 3) as f32 * 0.5;
                sumqx += weight[j] * q * xval[j];
                sumq2 += weight[j] * q * q;
            }
            if sumqx > 0.0 && sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                *scale = sumqx / sumq2;
                best_score = *scale * sumqx;
                grid_index = i as i32;
            }
        }
    }
    debug_assert!(grid_index >= 0);
    *scale *= 1.05;
    let levels = levels8(data.grid[grid_index as usize]);
    for i in 0..8 {
        l[i] = (levels[i] - 1) / 2;
    }
    grid_index as usize
}

pub fn iq1_find_best_neighbour2(
    data: &Iq2Data,
    neighbours: &[u16],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    xg: &[f32; 3],
    l: &mut [i8],
) -> usize {
    let mut best_score = f32::MAX;
    let mut grid_index = -1i32;
    for j in 1..=neighbours[0] as usize {
        let levels = levels8(data.grid[neighbours[j] as usize]);
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = xg[((levels[i] - 1) / 2) as usize];
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_score {
            best_score = d2;
            grid_index = neighbours[j] as i32;
        }
    }
    if grid_index < 0 {
        for i in 0..data.grid.len() {
            let levels = levels8(data.grid[i]);
            let mut d2 = 0.0f32;
            for j in 0..8 {
                let q = xg[((levels[j] - 1) / 2) as usize];
                let diff = scale * q - xval[j];
                d2 += weight[j] * diff * diff;
            }
            if d2 < best_score {
                best_score = d2;
                grid_index = i as i32;
            }
        }
    }
    debug_assert!(grid_index >= 0);
    let levels = levels8(data.grid[grid_index as usize]);
    for i in 0..8 {
        l[i] = (levels[i] - 1) / 2;
    }
    grid_index as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq2_codebook_maps_all_grid_points() {
        let data = iq2_data(Iq2Kind::Iq2Xxs).unwrap();
        assert_eq!(data.grid.len(), 256);
        assert_eq!(data.map.len(), IQ2_KMAP_SIZE);
        let mut on_grid = 0;
        for m in data.map.iter() {
            if *m >= 0 {
                on_grid += 1;
            }
        }
        assert_eq!(on_grid, 256);
    }

    #[test]
    fn iq3_codebook_maps_all_grid_points() {
        let data = iq3_data(256).unwrap();
        assert_eq!(data.grid.len(), 256);
        assert_eq!(data.map.len(), IQ3_KMAP_SIZE);
    }
}
