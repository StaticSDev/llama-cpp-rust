//! Port of `ggml-opt.cpp` and `ggml-opt.h`.
//!
//! The C++ optimizer builds on ggml tensors, contexts and a backend scheduler.
//! This module keeps those primitives in a small pure-Rust form so the
//! optimizer algorithms and constants stay close to the reference. Tensors own
//! their byte buffers; `Rc<RefCell<..>>` models the C++ pointers shared
//! between a graph and its duplicates.

use crate::core::{Error, Result};
use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

const GGML_MAX_DIMS: usize = 4;
const GGML_MAX_SRC: usize = 10;
const GGML_DEFAULT_GRAPH_SIZE: usize = 2048;
const TENSOR_OVERHEAD: usize = 256;

const FLAG_INPUT: u32 = 1;
const FLAG_OUTPUT: u32 = 2;
const FLAG_PARAM: u32 = 4;
const FLAG_LOSS: u32 = 8;

fn ensure(cond: bool, msg: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(Error::new(msg))
    }
}

// ====== Minimal ggml primitives ======

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    F32,
    F16,
    BF16,
    I32,
    I64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
}

impl Type {
    fn block_size(self) -> usize {
        match self {
            Type::Q4_0 | Type::Q4_1 | Type::Q5_0 | Type::Q5_1 | Type::Q8_0 | Type::Q8_1 => 32,
            _ => 1,
        }
    }

    fn type_size(self) -> usize {
        match self {
            Type::F32 | Type::I32 => 4,
            Type::F16 | Type::BF16 => 2,
            Type::I64 => 8,
            Type::Q4_0 => 18,
            Type::Q4_1 => 20,
            Type::Q5_0 => 22,
            Type::Q5_1 => 24,
            Type::Q8_0 => 34,
            Type::Q8_1 => 40,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GradKind {
    Broadcast,
    Identity,
    Negate,
    Scale(f32),
    Sqr,
    CrossEntropy,
}

impl Op {
    fn is_none(&self) -> bool {
        matches!(self, Op::None)
    }
}

#[derive(Clone, Debug)]
enum Op {
    None,
    Sum,
    Scale(f32),
    Sub,
    Sqr,
    CrossEntropyLoss,
    Argmax,
    CountEqual,
    AdamW,
    Sgd,
    Accum { target: Tensor, kind: GradKind },
}

pub type Tensor = Rc<RefCell<TensorData>>;

#[derive(Debug)]
pub struct TensorData {
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub ty: Type,
    pub data: Rc<RefCell<Vec<u8>>>,
    pub flags: u32,
    pub name: String,
    op: Op,
    src: [Option<Tensor>; GGML_MAX_SRC],
    view_src: Option<Tensor>,
    view_offs: usize,
}

pub fn tensor_id(t: &Tensor) -> usize {
    Rc::as_ptr(t) as usize
}

fn nelements_ne(ne: &[i64; GGML_MAX_DIMS]) -> usize {
    let mut n = 1usize;
    for &d in ne {
        n *= d.max(1) as usize;
    }
    n
}

fn nelements(t: &Tensor) -> usize {
    nelements_ne(&t.borrow().ne)
}

fn type_nbytes(ty: Type, n: usize) -> usize {
    let bs = ty.block_size();
    if bs == 1 {
        n * ty.type_size()
    } else {
        (n / bs) * ty.type_size()
    }
}

fn nbytes(t: &Tensor) -> usize {
    let d = t.borrow();
    type_nbytes(d.ty, nelements_ne(&d.ne))
}

fn is_scalar(t: &Tensor) -> bool {
    nelements(t) == 1
}

pub struct Context {
    mem_size: usize,
    tensors: Vec<Tensor>,
}

impl Context {
    pub fn new(mem_size: usize) -> Self {
        Self {
            mem_size,
            tensors: Vec::new(),
        }
    }

    pub fn new_tensor(&mut self, ty: Type, ne: [i64; GGML_MAX_DIMS]) -> Tensor {
        let bs = ty.block_size();
        let ts = ty.type_size();
        let mut nb = [0usize; GGML_MAX_DIMS];
        for d in 0..GGML_MAX_DIMS {
            nb[d] = if d == 0 {
                ts
            } else {
                nb[d - 1] * (ne[d - 1].max(1) as usize / bs)
            };
        }
        let n = type_nbytes(ty, nelements_ne(&ne));
        let t = Rc::new(RefCell::new(TensorData {
            ne,
            nb,
            ty,
            data: Rc::new(RefCell::new(vec![0u8; n])),
            flags: 0,
            name: String::new(),
            op: Op::None,
            src: std::array::from_fn(|_| None),
            view_src: None,
            view_offs: 0,
        }));
        self.tensors.push(t.clone());
        let _ = self.mem_size;
        t
    }

    // ggml_dup_tensor copies shape/type/strides only; op, flags and data stay
    // unset. ggml-opt's map_tensor fills those in after duplicating.
    pub fn dup_tensor(&mut self, other: &Tensor) -> Tensor {
        let d = other.borrow();
        let n = type_nbytes(d.ty, nelements_ne(&d.ne));
        let t = Rc::new(RefCell::new(TensorData {
            ne: d.ne,
            nb: d.nb,
            ty: d.ty,
            data: Rc::new(RefCell::new(vec![0u8; n])),
            flags: 0,
            name: String::new(),
            op: Op::None,
            src: std::array::from_fn(|_| None),
            view_src: None,
            view_offs: 0,
        }));
        drop(d);
        self.tensors.push(t.clone());
        t
    }

    pub fn new_tensor_1d(&mut self, ty: Type, ne0: i64) -> Tensor {
        self.new_tensor(ty, [ne0, 1, 1, 1])
    }

    pub fn new_tensor_2d(&mut self, ty: Type, ne0: i64, ne1: i64) -> Tensor {
        self.new_tensor(ty, [ne0, ne1, 1, 1])
    }
}

fn f32_data(t: &Tensor) -> Vec<f32> {
    let d = t.borrow();
    let buf = d.data.borrow();
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn write_f32_data(t: &Tensor, vals: &[f32]) {
    let d = t.borrow();
    let mut buf = d.data.borrow_mut();
    for (i, v) in vals.iter().enumerate() {
        let b = v.to_le_bytes();
        buf[i * 4..i * 4 + 4].copy_from_slice(&b);
    }
}

fn set_f32(t: &Tensor, v: f32) {
    let n = nelements(t);
    write_f32_data(t, &vec![v; n]);
}

#[allow(dead_code)] // kept for fidelity with ggml_get_data_f32 helpers
fn first_f32(t: &Tensor) -> f32 {
    f32_data(t).first().copied().unwrap_or(0.0)
}

fn i32_data(t: &Tensor) -> Vec<i32> {
    let d = t.borrow();
    let buf = d.data.borrow();
    buf.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i64_data(t: &Tensor) -> Vec<i64> {
    let d = t.borrow();
    let buf = d.data.borrow();
    buf.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn write_i32_data(t: &Tensor, vals: &[i32]) {
    let d = t.borrow();
    let mut buf = d.data.borrow_mut();
    for (i, v) in vals.iter().enumerate() {
        buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_i64_data(t: &Tensor, vals: &[i64]) {
    let d = t.borrow();
    let mut buf = d.data.borrow_mut();
    for (i, v) in vals.iter().enumerate() {
        buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
}

#[allow(dead_code)] // kept for fidelity with ggml_backend_tensor_set
fn tensor_set(t: &Tensor, src: &[u8], offset: usize) {
    let d = t.borrow();
    let mut buf = d.data.borrow_mut();
    buf[offset..offset + src.len()].copy_from_slice(src);
}

#[allow(dead_code)] // kept for fidelity with ggml_backend_tensor_get
fn tensor_get(t: &Tensor, offset: usize, size: usize) -> Vec<u8> {
    let d = t.borrow();
    let buf = d.data.borrow();
    buf[offset..offset + size].to_vec()
}

fn set_input(t: &Tensor) {
    t.borrow_mut().flags |= FLAG_INPUT;
}

fn set_output(t: &Tensor) {
    t.borrow_mut().flags |= FLAG_OUTPUT;
}

fn set_loss(t: &Tensor) {
    t.borrow_mut().flags |= FLAG_LOSS;
}

fn set_name(t: &Tensor, name: &str) {
    t.borrow_mut().name = name.to_string();
}

fn format_name(t: &Tensor, name: String) {
    t.borrow_mut().name = name;
}

fn src0(t: &Tensor) -> [Option<Tensor>; GGML_MAX_SRC] {
    let mut s: [Option<Tensor>; GGML_MAX_SRC] = std::array::from_fn(|_| None);
    s[0] = Some(t.clone());
    s
}

fn src1(a: &Tensor, b: &Tensor) -> [Option<Tensor>; GGML_MAX_SRC] {
    let mut s = src0(a);
    s[1] = Some(b.clone());
    s
}

fn op_tensor(
    ctx: &mut Context,
    ty: Type,
    ne: [i64; GGML_MAX_DIMS],
    op: Op,
    src: [Option<Tensor>; GGML_MAX_SRC],
) -> Tensor {
    let t = ctx.new_tensor(ty, ne);
    let mut d = t.borrow_mut();
    d.op = op;
    d.src = src;
    drop(d);
    t
}

fn sum(ctx: &mut Context, a: &Tensor) -> Tensor {
    op_tensor(ctx, Type::F32, [1, 1, 1, 1], Op::Sum, src0(a))
}

fn scale_op(ctx: &mut Context, a: &Tensor, s: f32) -> Tensor {
    let ne = a.borrow().ne;
    op_tensor(ctx, a.borrow().ty, ne, Op::Scale(s), src0(a))
}

fn sub(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Tensor {
    let ne = a.borrow().ne;
    op_tensor(ctx, Type::F32, ne, Op::Sub, src1(a, b))
}

fn sqr(ctx: &mut Context, a: &Tensor) -> Tensor {
    let ne = a.borrow().ne;
    op_tensor(ctx, Type::F32, ne, Op::Sqr, src0(a))
}

fn cross_entropy_loss(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Tensor {
    op_tensor(
        ctx,
        Type::F32,
        [1, 1, 1, 1],
        Op::CrossEntropyLoss,
        src1(a, b),
    )
}

fn argmax(ctx: &mut Context, a: &Tensor) -> Tensor {
    let ne = a.borrow().ne;
    op_tensor(
        ctx,
        Type::I32,
        [1, ne[1], ne[2], ne[3]],
        Op::Argmax,
        src0(a),
    )
}

fn count_equal(ctx: &mut Context, a: &Tensor, b: &Tensor) -> Tensor {
    op_tensor(ctx, Type::I64, [1, 1, 1, 1], Op::CountEqual, src1(a, b))
}

fn opt_step_adamw(
    ctx: &mut Context,
    a: &Tensor,
    grad: &Tensor,
    m: &Tensor,
    v: &Tensor,
    params: &Tensor,
) -> Tensor {
    let ne = a.borrow().ne;
    let mut s: [Option<Tensor>; GGML_MAX_SRC] = std::array::from_fn(|_| None);
    s[0] = Some(a.clone());
    s[1] = Some(grad.clone());
    s[2] = Some(m.clone());
    s[3] = Some(v.clone());
    s[4] = Some(params.clone());
    op_tensor(ctx, Type::F32, ne, Op::AdamW, s)
}

fn opt_step_sgd(ctx: &mut Context, a: &Tensor, grad: &Tensor, params: &Tensor) -> Tensor {
    let ne = a.borrow().ne;
    let mut s: [Option<Tensor>; GGML_MAX_SRC] = std::array::from_fn(|_| None);
    s[0] = Some(a.clone());
    s[1] = Some(grad.clone());
    s[2] = Some(params.clone());
    op_tensor(ctx, Type::F32, ne, Op::Sgd, s)
}

fn nrows(t: &Tensor) -> usize {
    let ne = t.borrow().ne;
    (ne[1].max(1) as usize) * (ne[2].max(1) as usize) * (ne[3].max(1) as usize)
}

fn compute_node(node: &Tensor) {
    let d = node.borrow();
    let src: Vec<Tensor> = d.src.iter().flatten().cloned().collect();
    match &d.op {
        Op::None => {}
        Op::Sum => {
            let s: f64 = f32_data(&src[0]).iter().map(|&x| x as f64).sum();
            write_f32_data(node, &[s as f32]);
        }
        Op::Scale(s) => {
            let x = f32_data(&src[0]);
            let y: Vec<f32> = x.iter().map(|v| v * s).collect();
            write_f32_data(node, &y);
        }
        Op::Sub => {
            let a = f32_data(&src[0]);
            let b = f32_data(&src[1]);
            let y: Vec<f32> = a
                .iter()
                .enumerate()
                .map(|(i, v)| v - if b.len() == 1 { b[0] } else { b[i] })
                .collect();
            write_f32_data(node, &y);
        }
        Op::Sqr => {
            let x = f32_data(&src[0]);
            let y: Vec<f32> = x.iter().map(|v| v * v).collect();
            write_f32_data(node, &y);
        }
        Op::CrossEntropyLoss => {
            let logits = f32_data(&src[0]);
            let labels = f32_data(&src[1]);
            let nc = node_ne0(&src[0]);
            let nr = nrows(&src[0]);
            let mut loss = 0.0f64;
            for row in 0..nr {
                let off = row * nc;
                let max = logits[off..off + nc]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f64 = logits[off..off + nc]
                    .iter()
                    .map(|&x| ((x - max) as f64).exp())
                    .sum();
                let log_sum = sum_exp.ln();
                for i in 0..nc {
                    let log_softmax = logits[off + i] as f64 - max as f64 - log_sum;
                    loss -= labels[off + i] as f64 * log_softmax;
                }
            }
            write_f32_data(node, &[(loss / nr as f64) as f32]);
        }
        Op::Argmax => {
            let x = f32_data(&src[0]);
            let nc = node_ne0(&src[0]);
            let nr = nrows(&src[0]);
            let mut y = vec![0i32; nr];
            for row in 0..nr {
                let off = row * nc;
                let mut best = off;
                for i in off + 1..off + nc {
                    if x[i] > x[best] {
                        best = i;
                    }
                }
                y[row] = (best - off) as i32;
            }
            write_i32_data(node, &y);
        }
        Op::CountEqual => {
            let a = i32_data(&src[0]);
            let b = i32_data(&src[1]);
            let n = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
            write_i64_data(node, &[n as i64]);
        }
        Op::AdamW => {
            let p = f32_data(&src[4]);
            let alpha = p[0];
            let beta1 = p[1];
            let beta2 = p[2];
            let eps = p[3];
            let wd = p[4];
            let beta1h = p[5];
            let beta2h = p[6];
            let keep = 1.0f32 - alpha * wd;
            let mut w = f32_data(&src[0]);
            let g = f32_data(&src[1]);
            let mut m = f32_data(&src[2]);
            let mut v = f32_data(&src[3]);
            for i in 0..w.len() {
                m[i] = m[i] * beta1 + g[i] * (1.0 - beta1);
                v[i] = v[i] * beta2 + g[i] * g[i] * (1.0 - beta2);
                let mh = m[i] * beta1h;
                let vh = (v[i] * beta2h).sqrt() + eps;
                w[i] = w[i] * keep - alpha * mh / vh;
            }
            write_f32_data(&src[2], &m);
            write_f32_data(&src[3], &v);
            write_f32_data(&src[0], &w);
        }
        Op::Sgd => {
            let p = f32_data(&src[2]);
            let alpha = p[0];
            let keep = 1.0f32 - alpha * p[1];
            let mut w = f32_data(&src[0]);
            let g = f32_data(&src[1]);
            for i in 0..w.len() {
                w[i] = w[i] * keep - alpha * g[i];
            }
            write_f32_data(&src[0], &w);
        }
        Op::Accum { target, kind } => {
            let grad = f32_data(&src[0]);
            let mut acc = f32_data(target);
            match kind {
                GradKind::Broadcast => {
                    let g = grad.first().copied().unwrap_or(0.0);
                    for x in acc.iter_mut() {
                        *x += g;
                    }
                }
                GradKind::Identity | GradKind::Negate | GradKind::Scale(_) | GradKind::Sqr => {
                    let x = if *kind == GradKind::Sqr {
                        f32_data(&src[1])
                    } else {
                        Vec::new()
                    };
                    for i in 0..acc.len() {
                        let g = if grad.len() == 1 { grad[0] } else { grad[i] };
                        let delta = match kind {
                            GradKind::Identity => g,
                            GradKind::Negate => -g,
                            GradKind::Scale(s) => g * s,
                            GradKind::Sqr => 2.0 * x[i] * g,
                            GradKind::Broadcast | GradKind::CrossEntropy => 0.0,
                        };
                        acc[i] += delta;
                    }
                }
                GradKind::CrossEntropy => {
                    let logits = f32_data(&src[1]);
                    let labels = f32_data(&src[2]);
                    let nc = node_ne0(&src[1]);
                    let nr = nrows(&src[1]);
                    let d_by_nr = grad.first().copied().unwrap_or(0.0) / nr as f32;
                    for row in 0..nr {
                        let off = row * nc;
                        let max = logits[off..off + nc]
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        let sum: f64 = logits[off..off + nc]
                            .iter()
                            .map(|&x| ((x - max) as f64).exp())
                            .sum();
                        for i in 0..nc {
                            let sm = ((logits[off + i] - max) as f64).exp() / sum;
                            acc[off + i] += ((sm as f32) - labels[off + i]) * d_by_nr;
                        }
                    }
                }
            }
            write_f32_data(target, &acc);
        }
    }
}

fn node_ne0(t: &Tensor) -> usize {
    t.borrow().ne[0].max(1) as usize
}

#[derive(Clone)]
pub struct CGraph {
    pub size: usize,
    pub leafs: Vec<Tensor>,
    pub nodes: Vec<Tensor>,
    pub grads: HashMap<usize, Tensor>,
    pub grad_accs: HashMap<usize, Tensor>,
    visited: HashSet<usize>,
}

fn new_graph(size: usize) -> CGraph {
    CGraph {
        size,
        leafs: Vec::new(),
        nodes: Vec::new(),
        grads: HashMap::new(),
        grad_accs: HashMap::new(),
        visited: HashSet::new(),
    }
}

fn graph_overhead_custom(size: usize, grads: bool) -> usize {
    std::mem::size_of::<CGraph>()
        + (if grads { 2 } else { 1 }) * size * std::mem::size_of::<usize>()
}

fn build_forward_expand(graph: &mut CGraph, t: &Tensor) {
    let id = tensor_id(t);
    if graph.visited.contains(&id) {
        return;
    }
    graph.visited.insert(id);
    let is_leaf = {
        let d = t.borrow();
        d.op.is_none() && d.flags & FLAG_PARAM == 0 && d.view_src.is_none()
    };
    if is_leaf {
        graph.leafs.push(t.clone());
    } else {
        let srcs: Vec<Tensor> = t.borrow().src.iter().flatten().cloned().collect();
        for s in &srcs {
            build_forward_expand(graph, s);
        }
        graph.nodes.push(t.clone());
    }
}

fn map_tensor(map: &mut HashMap<usize, Tensor>, ctx: &mut Context, tensor: &Tensor) -> Tensor {
    let id = tensor_id(tensor);
    if let Some(t) = map.get(&id) {
        return t.clone();
    }
    let new_tensor = ctx.dup_tensor(tensor);
    // ggml-opt's map_tensor keeps the original data pointer.
    new_tensor.borrow_mut().data = tensor.borrow().data.clone();
    map.insert(id, new_tensor.clone());

    let (srcs, view_src, op, flags, name, view_offs) = {
        let d = tensor.borrow();
        (
            d.src.clone(),
            d.view_src.clone(),
            d.op.clone(),
            d.flags,
            d.name.clone(),
            d.view_offs,
        )
    };
    let new_view_src = view_src.map(|t| map_tensor(map, ctx, &t));
    let new_srcs: [Option<Tensor>; GGML_MAX_SRC] =
        std::array::from_fn(|i| srcs[i].as_ref().map(|t| map_tensor(map, ctx, t)));
    let new_op = match op {
        Op::Accum { target, kind } => Op::Accum {
            target: map_tensor(map, ctx, &target),
            kind,
        },
        other => other,
    };
    {
        let mut d = new_tensor.borrow_mut();
        d.flags = flags;
        d.name = name;
        d.view_offs = view_offs;
        d.view_src = new_view_src;
        d.src = new_srcs;
        d.op = new_op;
    }
    new_tensor
}

fn graph_dup(ctx: &mut Context, src: &CGraph, force_grads: bool) -> CGraph {
    let mut map: HashMap<usize, Tensor> = HashMap::new();
    let mut dst = new_graph(src.size);
    for t in src.leafs.iter().chain(src.nodes.iter()) {
        let mapped = map_tensor(&mut map, ctx, t);
        build_forward_expand(&mut dst, &mapped);
    }
    if force_grads {
        let extra: Vec<Tensor> = src
            .grad_accs
            .values()
            .chain(src.grads.values())
            .cloned()
            .collect();
        for t in &extra {
            let mapped = map_tensor(&mut map, ctx, t);
            build_forward_expand(&mut dst, &mapped);
        }
    }
    for (id, g) in &src.grads {
        let dst_id = map.get(id).map(tensor_id);
        if let Some(dst_id) = dst_id {
            let mapped = map_tensor(&mut map, ctx, g);
            dst.grads.insert(dst_id, mapped);
        }
    }
    for (id, g) in &src.grad_accs {
        let dst_id = map.get(id).map(tensor_id);
        if let Some(dst_id) = dst_id {
            let mapped = map_tensor(&mut map, ctx, g);
            dst.grad_accs.insert(dst_id, mapped);
        }
    }
    dst
}

fn dup_graph(ctx: &mut Context, src: &CGraph) -> CGraph {
    let mut map: HashMap<usize, Tensor> = HashMap::new();
    let mut dst = new_graph(src.size);
    for t in &src.leafs {
        let mapped = map_tensor(&mut map, ctx, t);
        build_forward_expand(&mut dst, &mapped);
    }
    assert_eq!(dst.leafs.len(), src.leafs.len());
    for t in &src.nodes {
        let mapped = map_tensor(&mut map, ctx, t);
        build_forward_expand(&mut dst, &mapped);
    }
    assert_eq!(dst.nodes.len(), src.nodes.len());
    for node in &src.nodes {
        let id = tensor_id(node);
        let dst_id = map.get(&id).map(tensor_id).expect("mapped node");
        if let Some(g) = src.grads.get(&id) {
            let mapped = map_tensor(&mut map, ctx, g);
            dst.grads.insert(dst_id, mapped);
        }
        if let Some(g) = src.grad_accs.get(&id) {
            let mapped = map_tensor(&mut map, ctx, g);
            dst.grad_accs.insert(dst_id, mapped);
        }
    }
    dst
}

fn graph_get_grad(graph: &CGraph, node: &Tensor) -> Option<Tensor> {
    graph.grads.get(&tensor_id(node)).cloned()
}

fn graph_get_grad_acc(graph: &CGraph, node: &Tensor) -> Option<Tensor> {
    graph.grad_accs.get(&tensor_id(node)).cloned()
}

fn graph_reset(graph: &CGraph) {
    for ga in graph.grad_accs.values() {
        set_f32(ga, 0.0);
    }
    for node in &graph.nodes {
        if node.borrow().flags & FLAG_LOSS != 0 {
            if let Some(g) = graph_get_grad(graph, node) {
                set_f32(&g, 1.0);
            }
        }
    }
}

fn build_backward_expand(
    ctx: &mut Context,
    graph: &mut CGraph,
    grad_accs: &[Option<Tensor>],
) -> Result<()> {
    let n = graph.nodes.len();
    for i in 0..n {
        let node = graph.nodes[i].clone();
        if let Some(ga) = grad_accs.get(i).and_then(|x| x.clone()) {
            let one = node.borrow().flags & FLAG_LOSS != 0;
            set_f32(&ga, if one { 1.0 } else { 0.0 });
            build_forward_expand(graph, &ga);
            graph.grad_accs.insert(tensor_id(&node), ga.clone());
            graph.grads.insert(tensor_id(&node), ga);
        }
    }

    for ii in (0..n).rev() {
        let node = graph.nodes[ii].clone();
        let grad_y = match graph_get_grad(graph, &node) {
            Some(g) => g,
            None => continue,
        };
        let (srcs, op) = {
            let d = node.borrow();
            (d.src.clone(), d.op.clone())
        };
        let mut contribs: Vec<(Tensor, GradKind, Vec<Tensor>)> = Vec::new();
        match op {
            Op::Sum => {
                if let Some(s) = srcs[0].clone() {
                    contribs.push((s, GradKind::Broadcast, Vec::new()));
                }
            }
            Op::Scale(s) => {
                if let Some(a) = srcs[0].clone() {
                    contribs.push((a, GradKind::Scale(s), Vec::new()));
                }
            }
            Op::Sub => {
                if let Some(a) = srcs[0].clone() {
                    contribs.push((a, GradKind::Identity, Vec::new()));
                }
                if let Some(b) = srcs[1].clone() {
                    contribs.push((b, GradKind::Negate, Vec::new()));
                }
            }
            Op::Sqr => {
                if let Some(a) = srcs[0].clone() {
                    // d(x*x) = 2*x*grad; src[1] of the Accum node must be x,
                    // not the Sqr output.
                    contribs.push((a.clone(), GradKind::Sqr, vec![a]));
                }
            }
            Op::CrossEntropyLoss => {
                if let (Some(a), Some(b)) = (srcs[0].clone(), srcs[1].clone()) {
                    // src[1] = logits, src[2] = labels.
                    contribs.push((a.clone(), GradKind::CrossEntropy, vec![a, b]));
                }
            }
            _ => {}
        }

        for (src, kind, extra) in contribs {
            let target = if let Some(g) = graph_get_grad(graph, &src) {
                g
            } else {
                let ne = src.borrow().ne;
                let g = op_tensor(ctx, Type::F32, ne, Op::None, std::array::from_fn(|_| None));
                build_forward_expand(graph, &g);
                graph.grads.insert(tensor_id(&src), g.clone());
                g
            };
            let mut s: [Option<Tensor>; GGML_MAX_SRC] = std::array::from_fn(|_| None);
            s[0] = Some(grad_y.clone());
            s[1] = extra.first().cloned();
            s[2] = extra.get(1).cloned();
            let acc = op_tensor(
                ctx,
                Type::F32,
                [1, 1, 1, 1],
                Op::Accum { target, kind },
                s,
            );
            build_forward_expand(graph, &acc);
        }
    }
    Ok(())
}

fn compute_graph(graph: &CGraph) {
    for node in &graph.nodes {
        compute_node(node);
    }
}

#[derive(Clone, Debug)]
pub struct BackendSched {
    pub n_backends: usize,
    allocated_nodes: usize,
}

impl BackendSched {
    pub fn new(n_backends: usize) -> Self {
        Self {
            n_backends,
            allocated_nodes: 0,
        }
    }

    #[allow(dead_code)] // kept for fidelity with ggml_backend_sched_get_backend
    fn get_backend(&self, i: usize) -> usize {
        i.min(self.n_backends.saturating_sub(1))
    }

    // Drop the allocation of the previous graph.
    fn reset(&mut self) {
        self.allocated_nodes = 0;
    }

    // Tensors own their buffers here; allocation only records the graph size.
    fn alloc_graph(&mut self, graph: &CGraph) {
        self.allocated_nodes = graph.nodes.len();
    }

    fn graph_compute(&mut self, graph: &CGraph) {
        compute_graph(graph);
    }
}

// Tensors own their data, so allocation fills any buffer that is still short.
fn alloc_ctx_tensors(ctx: &mut Context) {
    for t in &ctx.tensors {
        let n = nbytes(t);
        let d = t.borrow();
        let mut buf = d.data.borrow_mut();
        if buf.len() < n {
            buf.resize(n, 0);
        }
    }
}

fn alloc_ctx_tensors_from_buft(ctx: &mut Context) {
    alloc_ctx_tensors(ctx);
}

// ====== Dataset ======

pub struct Dataset {
    pub ctx: Context,
    #[allow(dead_code)] // mirrors the dataset backend buffer flag
    allocated: bool,
    pub data: Tensor,
    pub labels: Option<Tensor>,
    pub ndata: i64,
    pub ndata_shard: i64,
    pub nbs_data: usize,
    pub nbs_labels: usize,
    pub permutation: Vec<i64>,
}

pub fn dataset_init(
    type_data: Type,
    type_label: Type,
    ne_datapoint: i64,
    ne_label: i64,
    ndata: i64,
    ndata_shard: i64,
) -> Result<Dataset> {
    ensure(ne_datapoint > 0, "ne_datapoint must be positive")?;
    ensure(ne_label >= 0, "ne_label must be non-negative")?;
    ensure(ndata > 0, "ndata must be positive")?;
    ensure(ndata_shard > 0, "ndata_shard must be positive")?;

    let mut ctx = Context::new(2 * TENSOR_OVERHEAD);
    let data = ctx.new_tensor_2d(type_data, ne_datapoint, ndata);
    let nbs_data = nbytes(&data) * ndata_shard as usize / ndata as usize;

    let (labels, nbs_labels) = if ne_label > 0 {
        let l = ctx.new_tensor_2d(type_label, ne_label, ndata);
        let n = nbytes(&l) * ndata_shard as usize / ndata as usize;
        (Some(l), n)
    } else {
        (None, 0)
    };
    alloc_ctx_tensors_from_buft(&mut ctx);

    let nshards = (ndata / ndata_shard) as usize;
    Ok(Dataset {
        ctx,
        allocated: true,
        data,
        labels,
        ndata,
        ndata_shard,
        nbs_data,
        nbs_labels,
        permutation: (0..nshards as i64).collect(),
    })
}

pub fn dataset_free(dataset: Dataset) {
    drop(dataset);
}

pub fn dataset_ndata(dataset: &Dataset) -> i64 {
    dataset.ndata
}

pub fn dataset_data(dataset: &Dataset) -> Tensor {
    dataset.data.clone()
}

pub fn dataset_labels(dataset: &Dataset) -> Option<Tensor> {
    dataset.labels.clone()
}

fn shuffle_range(v: &mut [i64], rng: &mut crate::core::Mt19937) {
    for i in (1..v.len()).rev() {
        let j = rng.next_u32() as usize % (i + 1);
        v.swap(i, j);
    }
}

pub fn dataset_shuffle(opt_ctx: &mut OptContext, dataset: &mut Dataset, idata: i64) -> Result<()> {
    ensure(idata <= dataset.ndata, "idata exceeds dataset size")?;
    if idata < 0 {
        shuffle_range(&mut dataset.permutation, &mut opt_ctx.rng);
        return Ok(());
    }
    ensure(idata % dataset.ndata_shard == 0, "idata not shard aligned")?;
    let ishard_max = (idata / dataset.ndata_shard) as usize;
    shuffle_range(&mut dataset.permutation[..ishard_max], &mut opt_ctx.rng);
    Ok(())
}

pub fn dataset_get_batch(
    dataset: &Dataset,
    data_batch: &Tensor,
    labels_batch: Option<&Tensor>,
    ibatch: i64,
) -> Result<()> {
    ensure(data_batch.borrow().ty == dataset.data.borrow().ty, "data type mismatch")?;
    match (labels_batch, &dataset.labels) {
        (Some(b), Some(l)) => ensure(b.borrow().ty == l.borrow().ty, "label type mismatch")?,
        (None, None) => {}
        _ => return Err(Error::new("labels_batch presence mismatch")),
    }
    let nb_data_batch = nbytes(data_batch);
    ensure(nb_data_batch % dataset.nbs_data == 0, "batch size not shard aligned")?;
    let shards_per_batch = nb_data_batch / dataset.nbs_data;
    if let Some(b) = labels_batch {
        ensure(
            nbytes(b) == shards_per_batch * dataset.nbs_labels,
            "label batch size mismatch",
        )?;
    }
    ensure(
        (ibatch + 1) * shards_per_batch as i64 <= dataset.permutation.len() as i64,
        "batch out of range",
    )?;

    for ishard_batch in 0..shards_per_batch {
        let ishard = dataset.permutation[(ibatch as usize) * shards_per_batch + ishard_batch];
        let soff = ishard as usize * dataset.nbs_data;
        let doff = ishard_batch * dataset.nbs_data;
        {
            let src = dataset.data.borrow();
            let dst = data_batch.borrow();
            let sbuf = src.data.borrow();
            let mut dbuf = dst.data.borrow_mut();
            dbuf[doff..doff + dataset.nbs_data].copy_from_slice(&sbuf[soff..soff + dataset.nbs_data]);
        }
        if let (Some(lb), Some(l)) = (labels_batch, &dataset.labels) {
            let soff = ishard as usize * dataset.nbs_labels;
            let doff = ishard_batch * dataset.nbs_labels;
            let src = l.borrow();
            let dst = lb.borrow();
            let sbuf = src.data.borrow();
            let mut dbuf = dst.data.borrow_mut();
            dbuf[doff..doff + dataset.nbs_labels]
                .copy_from_slice(&sbuf[soff..soff + dataset.nbs_labels]);
        }
    }
    Ok(())
}

pub fn dataset_get_batch_host(
    dataset: &Dataset,
    data_batch: &mut [u8],
    nb_data_batch: usize,
    mut labels_batch: Option<&mut [u8]>,
    ibatch: i64,
) -> Result<()> {
    ensure(
        labels_batch.is_none() == dataset.labels.is_none(),
        "labels presence mismatch",
    )?;
    ensure(
        nb_data_batch % dataset.nbs_data == 0,
        "batch size not shard aligned",
    )?;
    let shards_per_batch = nb_data_batch / dataset.nbs_data;
    ensure(
        (ibatch + 1) * shards_per_batch as i64 <= dataset.permutation.len() as i64,
        "batch out of range",
    )?;

    for ishard_batch in 0..shards_per_batch {
        let ishard = dataset.permutation[(ibatch as usize) * shards_per_batch + ishard_batch];
        let soff = ishard as usize * dataset.nbs_data;
        let doff = ishard_batch * dataset.nbs_data;
        {
            let src = dataset.data.borrow();
            let sbuf = src.data.borrow();
            data_batch[doff..doff + dataset.nbs_data]
                .copy_from_slice(&sbuf[soff..soff + dataset.nbs_data]);
        }
        if dataset.labels.is_some() {
            let lb = labels_batch.as_deref_mut().expect("labels batch missing");
            let l = dataset.labels.as_ref().unwrap();
            let soff = ishard as usize * dataset.nbs_labels;
            let doff = ishard_batch * dataset.nbs_labels;
            let src = l.borrow();
            let sbuf = src.data.borrow();
            lb[doff..doff + dataset.nbs_labels]
                .copy_from_slice(&sbuf[soff..soff + dataset.nbs_labels]);
        }
    }
    Ok(())
}

// ====== Model / Context ======

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossType {
    Mean,
    Sum,
    CrossEntropy,
    MeanSquaredError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildType {
    Forward = 10,
    Grad = 20,
    Opt = 30,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerType {
    AdamW,
    Sgd,
}

#[derive(Clone, Copy, Debug)]
pub struct OptimizerParams {
    pub adamw: AdamWParams,
    pub sgd: SgdParams,
}

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub alpha: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SgdParams {
    pub alpha: f32,
    pub wd: f32,
}

pub type GetOptPars = fn(&dyn Any) -> OptimizerParams;
pub type EpochCallback = fn(bool, &OptContext, &Dataset, &OptResult, i64, i64, i64);

pub fn get_default_optimizer_params(_userdata: &dyn Any) -> OptimizerParams {
    OptimizerParams {
        adamw: AdamWParams {
            alpha: 0.001f32,
            beta1: 0.9f32,
            beta2: 0.999f32,
            eps: 1e-8f32,
            wd: 0.0f32,
        },
        sgd: SgdParams {
            alpha: 1e-3f32,
            wd: 0.0f32,
        },
    }
}

pub fn get_constant_optimizer_params(userdata: &dyn Any) -> OptimizerParams {
    *userdata
        .downcast_ref::<OptimizerParams>()
        .expect("userdata must be OptimizerParams")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphKind {
    Forward,
    Grad,
    Opt,
}

pub struct Params {
    pub backend_sched: BackendSched,
    pub ctx_compute: Option<Context>,
    pub inputs: Option<Tensor>,
    pub outputs: Option<Tensor>,
    pub loss_type: LossType,
    pub build_type: BuildType,
    pub opt_period: i32,
    pub get_opt_pars: GetOptPars,
    pub get_opt_pars_ud: Option<Box<dyn Any>>,
    pub optimizer: OptimizerType,
}

pub fn default_params(backend_sched: BackendSched, loss_type: LossType) -> Params {
    Params {
        backend_sched,
        ctx_compute: None,
        inputs: None,
        outputs: None,
        loss_type,
        build_type: BuildType::Opt,
        opt_period: 1,
        get_opt_pars: get_default_optimizer_params,
        get_opt_pars_ud: None,
        optimizer: OptimizerType::AdamW,
    }
}

pub struct OptContext {
    pub backend_sched: BackendSched,
    allocated_graph: Option<GraphKind>,
    allocated_graph_copy: Option<CGraph>,
    ctx_static: Option<Context>,
    pub ctx_cpu: Option<Context>,
    ctx_compute: Option<Context>,
    ctx_copy: Option<Context>,
    buf_static: bool,
    buf_cpu: bool,
    pub rng: crate::core::Mt19937,
    pub loss_type: LossType,
    pub build_type: BuildType,
    build_type_alloc: BuildType,
    pub inputs: Option<Tensor>,
    pub outputs: Option<Tensor>,
    pub labels: Option<Tensor>,
    pub loss: Option<Tensor>,
    pub pred: Option<Tensor>,
    pub ncorrect: Option<Tensor>,
    gf: Option<CGraph>,
    gb_grad: Option<CGraph>,
    gb_opt: Option<CGraph>,
    pub static_graphs: bool,
    eval_ready: bool,
    grad_accs: Vec<Option<Tensor>>,
    grad_m: Vec<Option<Tensor>>,
    grad_v: Vec<Option<Tensor>>,
    pub iter: i64,
    pub opt_period: i32,
    pub opt_i: i32,
    pub loss_per_datapoint: bool,
    get_opt_pars: GetOptPars,
    get_opt_pars_ud: Option<Box<dyn Any>>,
    opt_step_params: Option<Tensor>,
    pub optimizer: OptimizerType,
}

fn build_loss(
    ctx: &mut Context,
    loss_type: LossType,
    outputs: &Tensor,
    opt_period: i32,
) -> (Tensor, Option<Tensor>, bool) {
    match loss_type {
        LossType::Mean => {
            let mut l = sum(ctx, outputs);
            set_name(&l, "loss_sum");
            let scale = 1.0f32 / (opt_period as f32 * nelements(outputs) as f32);
            l = scale_op(ctx, &l, scale);
            set_name(&l, "loss_mean");
            (l, None, true)
        }
        LossType::Sum => {
            let l = sum(ctx, outputs);
            set_name(&l, "loss_sum");
            (l, None, false)
        }
        LossType::CrossEntropy => {
            let labels = ctx.dup_tensor(outputs);
            set_input(&labels);
            set_name(&labels, "labels");
            let mut l = cross_entropy_loss(ctx, outputs, &labels);
            set_name(&l, "loss_cross_entropy");
            if opt_period > 1 {
                l = scale_op(ctx, &l, 1.0f32 / opt_period as f32);
                set_name(&l, "loss_cross_entropy_scaled");
            }
            (l, Some(labels), true)
        }
        LossType::MeanSquaredError => {
            let labels = ctx.dup_tensor(outputs);
            set_input(&labels);
            set_name(&labels, "labels");
            let mut l = sub(ctx, outputs, &labels);
            set_name(&l, "loss_error");
            l = sqr(ctx, &l);
            set_name(&l, "loss_squared_error");
            l = sum(ctx, &l);
            set_name(&l, "loss_sum_squared_error");
            let scale = 1.0f32 / (opt_period as f32 * nelements(outputs) as f32);
            l = scale_op(ctx, &l, scale);
            set_name(&l, "loss_mean_squared_error");
            (l, Some(labels), true)
        }
    }
}

impl OptContext {
    fn build(&mut self) -> Result<()> {
        ensure(self.ctx_compute.is_some(), "no compute context set")?;
        if self.static_graphs {
            let allocated = self
                .inputs
                .as_ref()
                .map(|t| !t.borrow().data.borrow().is_empty())
                .unwrap_or(false);
            ensure(allocated, "static graph inputs must be allocated")?;
        }

        let optimizer = self.optimizer;
        let accumulate = self.build_type as i32 >= BuildType::Grad as i32
            && !(self.static_graphs
                && self.build_type_alloc == BuildType::Opt
                && self.opt_period == 1);
        let need_momenta = self.build_type_alloc == BuildType::Opt && optimizer == OptimizerType::AdamW;

        if let Some(i) = &self.inputs {
            set_input(i);
        }
        if let Some(o) = &self.outputs {
            set_output(o);
        }

        let nodes = self.gf.as_ref().map(|g| g.nodes.clone()).unwrap_or_default();
        let mut n_param = 0;
        for node in &nodes {
            let flags = node.borrow().flags;
            if flags & FLAG_PARAM != 0 {
                n_param += 1;
            }
            ensure(flags & FLAG_LOSS == 0, "extra loss terms not implemented")?;
        }

        if self.ctx_static.is_none() {
            let n_loss = 1;
            let tensors_per_param = (if accumulate { 1 } else { 0 }) + (if need_momenta { 2 } else { 0 });
            let tensors_const = if self.static_graphs { 9 } else { 0 };
            let size_meta = (n_loss + tensors_per_param * n_param + tensors_const) * TENSOR_OVERHEAD;
            self.ctx_static = Some(Context::new(size_meta));
        }
        ensure(
            (self.build_type as i32) <= (self.build_type_alloc as i32),
            "build type exceeds allocated build type",
        )?;

        let cpu = Context::new(TENSOR_OVERHEAD);
        self.ctx_cpu = Some(cpu);
        self.buf_cpu = false;

        let outputs = self.outputs.clone().expect("outputs not set");
        let (loss, labels, per_datapoint) = if self.static_graphs {
            let ctx = self.ctx_static.as_mut().unwrap();
            build_loss(ctx, self.loss_type, &outputs, self.opt_period)
        } else {
            let ctx = self.ctx_compute.as_mut().unwrap();
            build_loss(ctx, self.loss_type, &outputs, self.opt_period)
        };
        set_output(&loss);
        set_loss(&loss);
        if let Some(gf) = self.gf.as_mut() {
            build_forward_expand(gf, &loss);
        }
        self.loss = Some(loss.clone());
        self.loss_per_datapoint = per_datapoint;
        if let Some(l) = labels {
            self.labels = Some(l);
        }

        if self.loss_type == LossType::CrossEntropy {
            let labels = self.labels.clone().unwrap();
            let (pred, ncorrect) = if self.static_graphs {
                let ctx = self.ctx_static.as_mut().unwrap();
                let p = argmax(ctx, &outputs);
                set_name(&p, "pred");
                let lmax = argmax(ctx, &labels);
                let n = count_equal(ctx, &p, &lmax);
                set_name(&n, "ncorrect");
                (p, n)
            } else {
                let ctx = self.ctx_compute.as_mut().unwrap();
                let p = argmax(ctx, &outputs);
                set_name(&p, "pred");
                let lmax = argmax(ctx, &labels);
                let n = count_equal(ctx, &p, &lmax);
                set_name(&n, "ncorrect");
                (p, n)
            };
            set_output(&pred);
            if let Some(gf) = self.gf.as_mut() {
                build_forward_expand(gf, &pred);
            }
            set_output(&ncorrect);
            if let Some(gf) = self.gf.as_mut() {
                build_forward_expand(gf, &ncorrect);
            }
            self.pred = Some(pred);
            self.ncorrect = Some(ncorrect);
        }

        if self.buf_static {
            if self.build_type == BuildType::Forward {
                return Ok(());
            }
        } else if self.build_type_alloc == BuildType::Forward {
            alloc_ctx_tensors(self.ctx_static.as_mut().unwrap());
            self.buf_static = true;
            return Ok(());
        }

        if self.grad_accs.is_empty() {
            let nodes = self.gf.as_ref().unwrap().nodes.clone();
            let n_nodes = nodes.len();
            self.grad_accs = vec![None; n_nodes];
            for (i, node) in nodes.iter().enumerate() {
                let flags = node.borrow().flags;
                if (accumulate && flags & FLAG_PARAM != 0) || flags & FLAG_LOSS != 0 {
                    let ne = node.borrow().ne;
                    let ctx = self.ctx_static.as_mut().unwrap();
                    self.grad_accs[i] = Some(ctx.new_tensor(Type::F32, ne));
                }
            }
            if need_momenta && self.build_type_alloc as i32 >= BuildType::Opt as i32 {
                self.grad_m = vec![None; n_nodes];
                self.grad_v = vec![None; n_nodes];
                for (i, node) in nodes.iter().enumerate() {
                    if node.borrow().flags & FLAG_PARAM != 0 {
                        let ne = node.borrow().ne;
                        let ctx = self.ctx_static.as_mut().unwrap();
                        self.grad_m[i] = Some(ctx.new_tensor(Type::F32, ne));
                        self.grad_v[i] = Some(ctx.new_tensor(Type::F32, ne));
                    }
                }
            }
        }

        let mut gb_grad = {
            let ctx = self.ctx_compute.as_mut().unwrap();
            let gf = self.gf.as_ref().unwrap();
            graph_dup(ctx, gf, true)
        };
        {
            let ctx = self.ctx_compute.as_mut().unwrap();
            build_backward_expand(ctx, &mut gb_grad, &self.grad_accs)?;
        }
        self.gb_grad = Some(gb_grad);

        if self.buf_static {
            if self.build_type == BuildType::Grad {
                return Ok(());
            }
        } else if self.build_type_alloc == BuildType::Grad {
            alloc_ctx_tensors(self.ctx_static.as_mut().unwrap());
            self.buf_static = true;
            if let Some(g) = &self.gb_grad {
                graph_reset(g);
            }
        }

        let n_gf_nodes = self.gf.as_ref().unwrap().nodes.len();
        let mut gb_opt = {
            let ctx = self.ctx_compute.as_mut().unwrap();
            let gb = self.gb_grad.as_ref().unwrap();
            graph_dup(ctx, gb, true)
        };

        let n_pars = if need_momenta { 7 } else { 2 };
        let opt_pars = {
            let ctx = self.ctx_cpu.as_mut().unwrap();
            ctx.new_tensor_1d(Type::F32, n_pars)
        };
        set_input(&opt_pars);
        format_name(&opt_pars, format!("{}_params", optimizer_name(optimizer)));
        self.opt_step_params = Some(opt_pars.clone());

        for i in (0..n_gf_nodes).rev() {
            let node = gb_opt.nodes[i].clone();
            let grad = graph_get_grad(&gb_opt, &node);
            if grad.is_none() || node.borrow().flags & FLAG_PARAM == 0 {
                continue;
            }
            let grad = grad.unwrap();
            let mut m = None;
            let mut v = None;
            if need_momenta {
                m = self.grad_m.get(i).cloned().flatten();
                v = self.grad_v.get(i).cloned().flatten();
                if let Some(mt) = &m {
                    format_name(mt, format!("AdamW m for {}", node.borrow().name));
                }
                if let Some(vt) = &v {
                    format_name(vt, format!("AdamW v for {}", node.borrow().name));
                }
            }
            let step = match optimizer {
                OptimizerType::AdamW => {
                    let mt = m.expect("AdamW momentum m");
                    let vt = v.expect("AdamW momentum v");
                    opt_step_adamw(
                        self.ctx_compute.as_mut().unwrap(),
                        &node,
                        &grad,
                        &mt,
                        &vt,
                        &opt_pars,
                    )
                }
                OptimizerType::Sgd => opt_step_sgd(
                    self.ctx_compute.as_mut().unwrap(),
                    &node,
                    &grad,
                    &opt_pars,
                ),
            };
            format_name(
                &step,
                format!("{} step for {}", optimizer_name(optimizer), node.borrow().name),
            );
            build_forward_expand(&mut gb_opt, &step);
        }
        self.gb_opt = Some(gb_opt);

        if !self.buf_static {
            alloc_ctx_tensors(self.ctx_static.as_mut().unwrap());
            self.buf_static = true;
            if let Some(g) = &self.gb_opt {
                graph_reset(g);
            }
        }
        alloc_ctx_tensors_from_buft(self.ctx_cpu.as_mut().unwrap());
        self.buf_cpu = true;
        Ok(())
    }

    fn graph_for(&self, kind: GraphKind) -> Option<&CGraph> {
        match kind {
            GraphKind::Forward => self.gf.as_ref(),
            GraphKind::Grad => self.gb_grad.as_ref(),
            GraphKind::Opt => self.gb_opt.as_ref(),
        }
    }
}

pub fn opt_init(mut params: Params) -> Result<OptContext> {
    let static_graphs = params.ctx_compute.is_some();
    ensure(params.opt_period >= 1, "opt_period must be at least 1")?;

    let mut result = OptContext {
        backend_sched: params.backend_sched.clone(),
        allocated_graph: None,
        allocated_graph_copy: None,
        ctx_static: None,
        ctx_cpu: None,
        ctx_compute: params.ctx_compute.take(),
        ctx_copy: None,
        buf_static: false,
        buf_cpu: false,
        rng: crate::core::Mt19937::new(5489),
        loss_type: params.loss_type,
        build_type: params.build_type,
        build_type_alloc: params.build_type,
        inputs: params.inputs.take(),
        outputs: params.outputs.take(),
        labels: None,
        loss: None,
        pred: None,
        ncorrect: None,
        gf: None,
        gb_grad: None,
        gb_opt: None,
        static_graphs,
        eval_ready: false,
        grad_accs: Vec::new(),
        grad_m: Vec::new(),
        grad_v: Vec::new(),
        iter: 1,
        opt_period: params.opt_period,
        opt_i: 0,
        loss_per_datapoint: false,
        get_opt_pars: params.get_opt_pars,
        get_opt_pars_ud: params.get_opt_pars_ud.take(),
        opt_step_params: None,
        optimizer: params.optimizer,
    };

    if !static_graphs {
        ensure(
            result.inputs.is_none() && result.outputs.is_none(),
            "inputs/outputs require a compute context",
        )?;
        return Ok(result);
    }
    ensure(
        result.inputs.is_some() && result.outputs.is_some(),
        "static graphs require inputs and outputs",
    )?;

    let outputs = result.outputs.clone().unwrap();
    let mut gf = new_graph(GGML_DEFAULT_GRAPH_SIZE);
    build_forward_expand(&mut gf, &outputs);
    result.gf = Some(gf);
    result.build()?;
    Ok(result)
}

pub fn opt_free(opt_ctx: OptContext) {
    drop(opt_ctx);
}

pub fn opt_reset(opt_ctx: &mut OptContext, optimizer: bool) {
    let graph = if optimizer {
        opt_ctx.gb_opt.as_ref()
    } else {
        opt_ctx.gb_grad.as_ref()
    };
    if let Some(g) = graph {
        graph_reset(g);
    }
    if optimizer {
        opt_ctx.iter = 1;
    }
}

pub fn opt_static_graphs(opt_ctx: &OptContext) -> bool {
    opt_ctx.static_graphs
}

pub fn opt_inputs(opt_ctx: &OptContext) -> Option<Tensor> {
    opt_ctx.inputs.clone()
}

pub fn opt_outputs(opt_ctx: &OptContext) -> Option<Tensor> {
    opt_ctx.outputs.clone()
}

pub fn opt_labels(opt_ctx: &OptContext) -> Option<Tensor> {
    opt_ctx.labels.clone()
}

pub fn opt_loss(opt_ctx: &OptContext) -> Option<Tensor> {
    opt_ctx.loss.clone()
}

pub fn opt_pred(opt_ctx: &OptContext) -> Option<Tensor> {
    opt_ctx.pred.clone()
}

pub fn opt_ncorrect(opt_ctx: &OptContext) -> Option<Tensor> {
    opt_ctx.ncorrect.clone()
}

pub fn opt_grad_acc(opt_ctx: &OptContext, node: &Tensor) -> Option<Tensor> {
    opt_ctx
        .gb_opt
        .as_ref()
        .and_then(|g| graph_get_grad_acc(g, node))
}

pub fn opt_context_optimizer_type(c: &OptContext) -> OptimizerType {
    c.optimizer
}

pub fn optimizer_name(o: OptimizerType) -> &'static str {
    match o {
        OptimizerType::AdamW => "adamw",
        OptimizerType::Sgd => "sgd",
    }
}

// ====== Optimization Result ======

pub struct OptResult {
    pub ndata: i64,
    pub loss: Vec<f32>,
    pub pred: Vec<i32>,
    pub ncorrect: i64,
    opt_period: i64,
    loss_per_datapoint: bool,
}

impl Default for OptResult {
    fn default() -> Self {
        Self {
            ndata: 0,
            loss: Vec::new(),
            pred: Vec::new(),
            ncorrect: 0,
            opt_period: -1,
            loss_per_datapoint: false,
        }
    }
}

pub fn opt_result_init() -> OptResult {
    OptResult::default()
}

pub fn opt_result_free(result: OptResult) {
    drop(result);
}

impl OptResult {
    pub fn reset(&mut self) {
        self.ndata = 0;
        self.loss.clear();
        self.pred.clear();
        self.ncorrect = 0;
    }

    pub fn ndata(&self) -> i64 {
        self.ndata
    }
}

pub fn opt_result_ndata(result: &OptResult) -> i64 {
    result.ndata
}

pub fn opt_result_loss(result: &OptResult, unc: bool) -> (f64, f64) {
    let nbatches = result.loss.len() as i64;
    if nbatches == 0 {
        return (0.0, f64::NAN);
    }
    let mut sum = 0.0f64;
    let mut sum_squared = 0.0f64;
    for &loss in &result.loss {
        let loss_scaled = if result.loss_per_datapoint {
            loss as f64 * result.opt_period as f64
        } else {
            loss as f64
        };
        sum += loss_scaled;
        sum_squared += loss_scaled * loss_scaled;
    }
    let mean = sum / nbatches as f64;
    let value = if result.loss_per_datapoint { mean } else { sum };
    if !unc {
        return (value, 0.0);
    }
    if nbatches < 2 {
        return (value, f64::NAN);
    }
    let var_sum = sum_squared / nbatches as f64 - mean * mean;
    let u = if result.loss_per_datapoint {
        (var_sum / (nbatches - 1) as f64).sqrt()
    } else {
        (var_sum * nbatches as f64 / (nbatches - 1) as f64).sqrt()
    };
    (value, u)
}

pub fn opt_result_pred(result: &OptResult, pred: &mut [i32]) {
    for (i, p) in result.pred.iter().enumerate() {
        pred[i] = *p;
    }
}

pub fn opt_result_accuracy(result: &OptResult, unc: bool) -> (f64, f64) {
    let accuracy = if result.ncorrect >= 0 {
        result.ncorrect as f64 / result.ndata as f64
    } else {
        f64::NAN
    };
    if !unc {
        return (accuracy, 0.0);
    }
    let u = if result.ncorrect >= 0 && result.ndata >= 2 {
        (accuracy * (1.0 - accuracy) / (result.ndata - 1) as f64).sqrt()
    } else {
        f64::NAN
    };
    (accuracy, u)
}

// ====== Computation ======

pub fn prepare_alloc(
    opt_ctx: &mut OptContext,
    ctx_compute: Context,
    gf: CGraph,
    inputs: Tensor,
    outputs: Tensor,
) -> Result<()> {
    ensure(!opt_ctx.static_graphs, "static graphs cannot prepare alloc")?;
    opt_ctx.ctx_compute = Some(ctx_compute);
    opt_ctx.gf = Some(gf);
    opt_ctx.inputs = Some(inputs);
    opt_ctx.outputs = Some(outputs);
    Ok(())
}

impl OptContext {
    pub fn alloc(&mut self, backward: bool) -> Result<()> {
        ensure(!self.eval_ready, "eval must precede the next alloc")?;
        if self.build_type == BuildType::Opt
            && self.opt_period > 1
            && self.opt_i == 0
        {
            if let Some(g) = &self.gb_grad {
                graph_reset(g);
            }
        }
        let opt_i_next = (self.opt_i + 1) % self.opt_period;
        self.build_type = if backward {
            if opt_i_next == 0 {
                BuildType::Opt
            } else {
                BuildType::Grad
            }
        } else {
            BuildType::Forward
        };
        if !self.static_graphs {
            self.build()?;
        }

        let kind = match self.build_type {
            BuildType::Forward => GraphKind::Forward,
            BuildType::Grad => GraphKind::Grad,
            BuildType::Opt => GraphKind::Opt,
        };
        if self.allocated_graph == Some(kind) {
            self.eval_ready = true;
            return Ok(());
        }

        self.backend_sched.reset();
        if self.static_graphs {
            let g = self.graph_for(kind).expect("graph missing").clone();
            let size = g.size;
            self.ctx_copy = Some(Context::new(
                size * TENSOR_OVERHEAD + graph_overhead_custom(size, true),
            ));
            let ctx = self.ctx_copy.as_mut().unwrap();
            self.allocated_graph_copy = Some(dup_graph(ctx, &g));
        } else {
            self.allocated_graph_copy = self.graph_for(kind).cloned();
        }
        if let Some(copy) = &self.allocated_graph_copy {
            self.backend_sched.alloc_graph(copy);
        }
        self.allocated_graph = Some(kind);
        self.eval_ready = true;
        Ok(())
    }

    pub fn eval(&mut self, result: Option<&mut OptResult>) -> Result<()> {
        ensure(self.eval_ready, "alloc must precede eval")?;
        if self.allocated_graph == Some(GraphKind::Opt) {
            let ud: &dyn Any = match &self.get_opt_pars_ud {
                Some(b) => b.as_ref(),
                None => &(),
            };
            let pars = (self.get_opt_pars)(ud);
            match self.optimizer {
                OptimizerType::AdamW => {
                    let a = pars.adamw;
                    ensure(a.alpha > 0.0, "alpha must be positive")?;
                    ensure((0.0..=1.0).contains(&a.beta1), "beta1 out of range")?;
                    ensure((0.0..=1.0).contains(&a.beta2), "beta2 out of range")?;
                    ensure(a.eps >= 0.0, "eps must be non-negative")?;
                    ensure((0.0..=1.0).contains(&a.wd), "wd out of range")?;
                    let beta1h = 1.0f32 / (1.0f32 - a.beta1.powf(self.iter as f32));
                    let beta2h = 1.0f32 / (1.0f32 - a.beta2.powf(self.iter as f32));
                    let p = self.opt_step_params.clone().expect("optimizer params tensor");
                    write_f32_data(&p, &[a.alpha, a.beta1, a.beta2, a.eps, a.wd, beta1h, beta2h]);
                }
                OptimizerType::Sgd => {
                    let s = pars.sgd;
                    ensure(s.alpha > 0.0, "alpha must be positive")?;
                    ensure((0.0..=1.0).contains(&s.wd), "wd out of range")?;
                    let p = self.opt_step_params.clone().expect("optimizer params tensor");
                    write_f32_data(&p, &[s.alpha, s.wd]);
                }
            }
        }

        if let Some(copy) = &self.allocated_graph_copy {
            self.backend_sched.graph_compute(copy);
        }
        self.iter += (self.allocated_graph == Some(GraphKind::Opt)) as i64;
        self.opt_i = (self.opt_i + 1) % self.opt_period;

        if !self.static_graphs {
            self.gf = None;
            self.gb_grad = None;
            self.gb_opt = None;
            self.allocated_graph = None;
            self.allocated_graph_copy = None;
        }
        self.eval_ready = false;

        let result = match result {
            Some(r) => r,
            None => return Ok(()),
        };
        if result.ndata == 0 {
            result.loss_per_datapoint = self.loss_per_datapoint;
            result.opt_period = self.opt_period as i64;
        } else {
            ensure(
                result.loss_per_datapoint == self.loss_per_datapoint && result.opt_period == self.opt_period as i64,
                "result settings mismatch",
            )?;
        }
        let outputs = self.outputs.clone().expect("outputs missing");
        let ndata = outputs.borrow().ne[1];
        ensure(
            result.ndata == ndata * result.loss.len() as i64,
            "varying batch size not supported",
        )?;
        result.ndata += ndata;

        let loss = self.loss.clone().expect("loss missing");
        ensure(is_scalar(&loss) && loss.borrow().ty == Type::F32, "loss must be scalar F32")?;
        let loss_value = f32_data(&loss)[0];
        result.loss.push(loss_value);

        if let Some(pred) = &self.pred {
            ensure(pred.borrow().ty == Type::I32, "pred must be I32")?;
            result.pred.extend(i32_data(pred));
        }
        if self.ncorrect.is_none() || result.ncorrect < 0 {
            result.ncorrect = -1;
            return Ok(());
        }
        let nc = self.ncorrect.clone().unwrap();
        ensure(is_scalar(&nc) && nc.borrow().ty == Type::I64, "ncorrect must be scalar I64")?;
        result.ncorrect += i64_data(&nc)[0];
        Ok(())
    }
}

// ====== High-Level Functions ======

fn time_init() -> Instant {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *START.get_or_init(Instant::now)
}

// Microseconds since time_init; only differences are used by the callbacks.
fn time_us() -> i64 {
    time_init().elapsed().as_micros() as i64
}

pub fn epoch(
    opt_ctx: &mut OptContext,
    dataset: &Dataset,
    mut result_train: Option<&mut OptResult>,
    mut result_eval: Option<&mut OptResult>,
    idata_split: i64,
    callback_train: Option<EpochCallback>,
    callback_eval: Option<EpochCallback>,
) -> Result<()> {
    ensure(opt_ctx.static_graphs, "epoch requires static graphs")?;
    let inputs = opt_ctx.inputs.clone().unwrap();
    let labels = opt_ctx.labels.clone();
    let data = &dataset.data;
    ensure(
        data.borrow().ne[0] == inputs.borrow().ne[0],
        "data/input feature size mismatch",
    )?;
    let ndata = data.borrow().ne[1];
    let ndata_batch = inputs.borrow().ne[1];
    ensure(ndata % ndata_batch == 0, "dataset not divisible by batch")?;
    let nbatches = ndata / ndata_batch;
    let split = if idata_split < 0 { ndata } else { idata_split };
    ensure(split % ndata_batch == 0, "split not batch aligned")?;
    let ibatch_split = split / ndata_batch;

    let mut ibatch = 0i64;
    let mut t_loop_start = time_us();
    while ibatch < ibatch_split {
        opt_ctx.alloc(true)?;
        dataset_get_batch(dataset, &inputs, labels.as_ref(), ibatch)?;
        opt_ctx.eval(result_train.as_deref_mut())?;
        if let (Some(cb), Some(r)) = (callback_train, result_train.as_ref()) {
            cb(true, opt_ctx, dataset, r, ibatch + 1, ibatch_split, t_loop_start);
        }
        ibatch += 1;
    }
    t_loop_start = time_us();
    while ibatch < nbatches {
        opt_ctx.alloc(false)?;
        dataset_get_batch(dataset, &inputs, labels.as_ref(), ibatch)?;
        opt_ctx.eval(result_eval.as_deref_mut())?;
        if let (Some(cb), Some(r)) = (callback_eval, result_eval.as_ref()) {
            cb(
                false,
                opt_ctx,
                dataset,
                r,
                ibatch + 1 - ibatch_split,
                nbatches - ibatch_split,
                t_loop_start,
            );
        }
        ibatch += 1;
    }
    Ok(())
}

pub fn epoch_callback_progress_bar(
    train: bool,
    opt_ctx: &OptContext,
    _dataset: &Dataset,
    result: &OptResult,
    ibatch: i64,
    ibatch_max: i64,
    t_start_us: i64,
) {
    let prefix = if train { "train: " } else { "val:   " };
    eprint!("{}[", prefix);
    let bar_length = 8i64;
    let ibatch8 = 8 * ibatch;
    for j in 0..bar_length {
        let base = ibatch_max * (8 * j + 8) / bar_length;
        if base < ibatch8 {
            eprint!("\u{2588}");
        } else if ibatch_max * (8 * j + 7) / bar_length < ibatch8 {
            eprint!("\u{2589}");
        } else if ibatch_max * (8 * j + 6) / bar_length < ibatch8 {
            eprint!("\u{258A}");
        } else if ibatch_max * (8 * j + 5) / bar_length < ibatch8 {
            eprint!("\u{258B}");
        } else if ibatch_max * (8 * j + 4) / bar_length < ibatch8 {
            eprint!("\u{258C}");
        } else if ibatch_max * (8 * j + 3) / bar_length < ibatch8 {
            eprint!("\u{258D}");
        } else if ibatch_max * (8 * j + 2) / bar_length < ibatch8 {
            eprint!("\u{258E}");
        } else if ibatch_max * (8 * j + 1) / bar_length < ibatch8 {
            eprint!("\u{258F}");
        } else {
            eprint!(" ");
        }
    }

    let batch_size = opt_ctx.inputs.as_ref().map(|t| t.borrow().ne[1]).unwrap_or(0);
    let idata = ibatch * batch_size;
    let idata_max = ibatch_max * batch_size;
    let (loss, loss_unc) = opt_result_loss(result, true);
    let (accuracy, accuracy_unc) = opt_result_accuracy(result, true);

    let t_ibatch_us = time_us() - t_start_us;
    let mut t_ibatch_s = t_ibatch_us / 1_000_000;
    let t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    let t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    let t_eta_us = t_ibatch_us * (ibatch_max - ibatch) / ibatch.max(1);
    let mut t_eta_s = t_eta_us / 1_000_000;
    let t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    let t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    eprint!(
        "] data={:07}/{:07} loss={:.5}\u{b1}{:.5} acc={:.2}\u{b1}{:.2}% t={:02}:{:02}:{:02} ETA={:02}:{:02}:{:02} \r",
        idata,
        idata_max,
        loss,
        loss_unc,
        100.0 * accuracy,
        100.0 * accuracy_unc,
        t_ibatch_h,
        t_ibatch_m,
        t_ibatch_s,
        t_eta_h,
        t_eta_m,
        t_eta_s
    );
    if ibatch == ibatch_max {
        eprint!("\n");
    }
}

pub fn fit(
    backend_sched: &BackendSched,
    ctx_compute: Context,
    inputs: Tensor,
    outputs: Tensor,
    dataset: &mut Dataset,
    loss_type: LossType,
    optimizer: OptimizerType,
    get_opt_pars: GetOptPars,
    nepoch: i64,
    nbatch_logical: i64,
    val_split: f32,
    silent: bool,
) -> Result<()> {
    let _ = time_init();
    let t_start_us = time_us();
    let ndata = dataset.data.borrow().ne[1];
    let nbatch_physical = inputs.borrow().ne[1];
    ensure(ndata % nbatch_logical == 0, "ndata not divisible by logical batch")?;
    ensure(
        nbatch_logical % nbatch_physical == 0,
        "logical batch not divisible by physical batch",
    )?;
    let opt_period = nbatch_logical / nbatch_physical;
    let nbatches_logical = ndata / nbatch_logical;
    ensure((0.0..1.0).contains(&val_split), "val_split out of range")?;
    let ibatch_split = ((1.0f32 - val_split) * nbatches_logical as f32) as i64 * opt_period;
    let idata_split = ibatch_split * nbatch_physical;

    let mut params = default_params(backend_sched.clone(), loss_type);
    params.ctx_compute = Some(ctx_compute);
    params.inputs = Some(inputs);
    params.outputs = Some(outputs);
    params.opt_period = opt_period as i32;
    params.get_opt_pars = get_opt_pars;
    params.optimizer = optimizer;
    let mut opt_ctx = opt_init(params)?;

    if nbatch_logical < ndata {
        dataset_shuffle(&mut opt_ctx, dataset, -1)?;
    }

    let mut result_train = opt_result_init();
    let mut result_val = opt_result_init();
    let callback: Option<EpochCallback> = if silent {
        None
    } else {
        Some(epoch_callback_progress_bar)
    };

    let mut epoch_id = 1i64;
    while epoch_id <= nepoch {
        if nbatch_logical < idata_split {
            dataset_shuffle(&mut opt_ctx, dataset, idata_split)?;
        }
        result_train.reset();
        result_val.reset();
        if !silent {
            eprintln!("ggml_opt_fit: epoch {:04}/{:04}:", epoch_id, nepoch);
        }
        opt_ctx.get_opt_pars_ud = Some(Box::new(epoch_id));
        epoch(
            &mut opt_ctx,
            dataset,
            Some(&mut result_train),
            Some(&mut result_val),
            idata_split,
            callback,
            callback,
        )?;
        if !silent {
            eprintln!();
        }
        epoch_id += 1;
    }

    if !silent {
        let mut t_total_s = (time_us() - t_start_us) / 1_000_000;
        let t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        let t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        eprintln!(
            "ggml_opt_fit: training took {:02}:{:02}:{:02}",
            t_total_h, t_total_m, t_total_s
        );
    }
    opt_free(opt_ctx);
    opt_result_free(result_train);
    opt_result_free(result_val);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_optimizer_params_match_reference() {
        let p = get_default_optimizer_params(&());
        assert_eq!(p.adamw.alpha, 0.001);
        assert_eq!(p.adamw.beta1, 0.9);
        assert_eq!(p.adamw.beta2, 0.999);
        assert_eq!(p.adamw.eps, 1e-8);
        assert_eq!(p.adamw.wd, 0.0);
        assert_eq!(p.sgd.alpha, 1e-3);
        assert_eq!(p.sgd.wd, 0.0);
    }

    #[test]
    fn dataset_shards_copy_to_batch() {
        let ds = dataset_init(Type::F32, Type::F32, 2, 1, 4, 2).unwrap();
        assert_eq!(ds.permutation, vec![0, 1]);
        assert_eq!(ds.nbs_data, 2 * 4 * 2);
        let w = f32_data(&ds.data);
        write_f32_data(&ds.data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(w.len(), 8);
        let mut ctx = Context::new(TENSOR_OVERHEAD);
        let batch = ctx.new_tensor_2d(Type::F32, 2, 2);
        let labels_batch = ctx.new_tensor_2d(Type::F32, 1, 2);
        dataset_get_batch(&ds, &batch, Some(&labels_batch), 0).unwrap();
        assert_eq!(f32_data(&batch), vec![1.0, 2.0, 3.0, 4.0]);
        dataset_get_batch(&ds, &batch, Some(&labels_batch), 1).unwrap();
        assert_eq!(f32_data(&batch), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn one_adamw_step_updates_params() {
        let mut ctx = Context::new(4096);
        let w = ctx.new_tensor_2d(Type::F32, 2, 1);
        w.borrow_mut().flags |= FLAG_PARAM;
        write_f32_data(&w, &[1.0, 2.0]);
        let x = ctx.new_tensor_2d(Type::F32, 2, 1);
        write_f32_data(&x, &[0.5, 0.5]);
        let outputs = sub(&mut ctx, &w, &x);

        let mut params = default_params(BackendSched::new(1), LossType::MeanSquaredError);
        params.ctx_compute = Some(ctx);
        params.inputs = Some(x);
        params.outputs = Some(outputs);
        let mut opt_ctx = opt_init(params).unwrap();

        let labels = opt_ctx.labels.clone().unwrap();
        write_f32_data(&labels, &[0.0, 0.0]);
        opt_ctx.alloc(true).unwrap();
        let mut result = opt_result_init();
        opt_ctx.eval(Some(&mut result)).unwrap();

        assert_eq!(result.loss.len(), 1);
        assert!((result.loss[0] - 1.25).abs() < 1e-5, "unexpected loss");
        let after = f32_data(&w);
        assert!(after[0] < 1.0 && after[1] < 2.0, "params did not move: {:?}", after);
    }

    #[test]
    fn dynamic_graph_path_updates_params() {
        let mut ctx = Context::new(4096);
        let w = ctx.new_tensor_2d(Type::F32, 2, 1);
        w.borrow_mut().flags |= FLAG_PARAM;
        write_f32_data(&w, &[1.0, 2.0]);
        let x = ctx.new_tensor_2d(Type::F32, 2, 1);
        write_f32_data(&x, &[0.5, 0.5]);
        let outputs = sub(&mut ctx, &w, &x);

        let params = default_params(BackendSched::new(1), LossType::Mean);
        let mut opt_ctx = opt_init(params).unwrap();
        let mut gf = new_graph(GGML_DEFAULT_GRAPH_SIZE);
        build_forward_expand(&mut gf, &outputs);
        prepare_alloc(&mut opt_ctx, ctx, gf, x, outputs).unwrap();
        opt_ctx.alloc(true).unwrap();
        opt_ctx.eval(None).unwrap();
        let after = f32_data(&w);
        assert!(after[0] < 1.0 && after[1] < 2.0, "params did not move: {:?}", after);
    }

    #[test]
    fn cross_entropy_builds_predictions() {
        let mut ctx = Context::new(4096);
        let logits = ctx.new_tensor_2d(Type::F32, 3, 2);
        logits.borrow_mut().flags |= FLAG_PARAM;
        write_f32_data(&logits, &[4.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
        let inputs = ctx.new_tensor_2d(Type::F32, 3, 1);
        let mut params = default_params(BackendSched::new(1), LossType::CrossEntropy);
        params.ctx_compute = Some(ctx);
        params.inputs = Some(inputs);
        params.outputs = Some(logits);
        let mut opt_ctx = opt_init(params).unwrap();
        let labels = opt_ctx.labels.clone().unwrap();
        write_f32_data(&labels, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        opt_ctx.alloc(true).unwrap();
        let mut result = opt_result_init();
        opt_ctx.eval(Some(&mut result)).unwrap();
        assert!(opt_ctx.pred.is_some() && opt_ctx.ncorrect.is_some());
        assert_eq!(result.pred, vec![0, 2]);
        assert_eq!(result.ncorrect, 2);
    }

    #[test]
    fn result_loss_statistics() {
        let mut r = opt_result_init();
        r.loss_per_datapoint = true;
        r.opt_period = 2;
        r.loss = vec![1.0, 2.0];
        r.ndata = 4;
        let (loss, unc) = opt_result_loss(&r, true);
        assert_eq!(loss, 3.0);
        assert!((unc - 1.0).abs() < 1e-5);
        r.ncorrect = 3;
        let (acc, acc_unc) = opt_result_accuracy(&r, true);
        assert!((acc - 0.75).abs() < 1e-9);
        assert!(acc_unc.is_finite());
    }
}
