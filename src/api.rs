//! Port of the public API in `include/llama.h`.
//!
//! Raw C pointers become owned `Vec`/`Option` fields and fallible pointer
//! returns become `Result`. Callback payloads, ggml tensors, backends and the
//! model graph are represented by local stand-in types. Inference is a dry
//! stand-in: batches are validated and recorded, and output buffers are sized
//! and filled from the model description. Sampling delegates to `crate::sampler`.

use crate::core::{Error, Result, Token};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::rc::Rc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// `llama_pos`.
pub type LlamaPos = i32;
/// `llama_seq_id`. Negative values match any sequence in the memory API.
pub type LlamaSeqId = i32;
/// `llama_state_seq_flags`.
pub type StateSeqFlags = u32;

pub const LLAMA_DEFAULT_SEED: u32 = u32::MAX;
pub const LLAMA_TOKEN_NULL: Token = -1;
pub const LLAMA_FILE_MAGIC_GGLA: u32 = 0x6767_6c61;
pub const LLAMA_FILE_MAGIC_GGSN: u32 = 0x6767_736e;
pub const LLAMA_FILE_MAGIC_GGSQ: u32 = 0x6767_7371;
pub const LLAMA_SESSION_MAGIC: u32 = LLAMA_FILE_MAGIC_GGSN;
pub const LLAMA_SESSION_VERSION: u32 = 11;
pub const LLAMA_STATE_SEQ_MAGIC: u32 = LLAMA_FILE_MAGIC_GGSQ;
pub const LLAMA_STATE_SEQ_VERSION: u32 = 4;
pub const LLAMA_STATE_SEQ_FLAGS_NONE: StateSeqFlags = 0;
/// Backwards-compatible alias of `PARTIAL_ONLY`.
pub const LLAMA_STATE_SEQ_FLAGS_SWA_ONLY: StateSeqFlags = 1;
pub const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY: StateSeqFlags = 1;
pub const LLAMA_STATE_SEQ_FLAGS_ON_DEVICE: StateSeqFlags = 2;
pub const LLAMA_DRY_MAX_SEQ_BREAKERS: usize = 64;
pub const LLAMA_VERSION: &str = "b6396af0";
pub const LLAMA_MAX_DEVICES: usize = 16;
pub const LLAMA_MAX_TENSOR_BUFT_OVERRIDES: usize = 4096;
pub const LLAMA_QUANT_MAX_BUF_SIZE: usize = 8 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum VocabType {
    None = 0,
    Spm = 1,
    Bpe = 2,
    Wpm = 3,
    Ugm = 4,
    Rwkv = 5,
    Plamo2 = 6,
    Test = 7,
    Plamo3 = 8,
}

impl Default for RopeType {
    fn default() -> Self {
        Self::Norm
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RopeType {
    None = -1,
    Norm = 0,
    Neox = 2,
    Mrope = 8,
    Imrope = 40,
    Vision = 24,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TokenAttrFlag {
    Undefined = 0,
    Unknown = 1 << 0,
    Unused = 1 << 1,
    Normal = 1 << 2,
    Control = 1 << 3,
    UserDefined = 1 << 4,
    Byte = 1 << 5,
    Normalized = 1 << 6,
    Lstrip = 1 << 7,
    Rstrip = 1 << 8,
    SingleWord = 1 << 9,
}

/// Bit mask of `TokenAttrFlag`; C attributes are OR-able enum values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenAttr(pub i32);

impl TokenAttr {
    pub const fn from_flag(flag: TokenAttrFlag) -> Self {
        Self(flag as i32)
    }

    pub fn contains(self, flag: TokenAttrFlag) -> bool {
        self.0 & (flag as i32) != 0
    }

    pub fn with(self, flag: TokenAttrFlag) -> Self {
        Self(self.0 | (flag as i32))
    }
}

impl Default for Ftype {
    fn default() -> Self {
        Self::MostlyQ8_0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Ftype {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2K = 10,
    MostlyQ3KS = 11,
    MostlyQ3KM = 12,
    MostlyQ3KL = 13,
    MostlyQ4KS = 14,
    MostlyQ4KM = 15,
    MostlyQ5KS = 16,
    MostlyQ5KM = 17,
    MostlyQ6K = 18,
    MostlyIq2Xxs = 19,
    MostlyIq2Xs = 20,
    MostlyQ2KS = 21,
    MostlyIq3Xs = 22,
    MostlyIq3Xxs = 23,
    MostlyIq1S = 24,
    MostlyIq4Nl = 25,
    MostlyIq3S = 26,
    MostlyIq3M = 27,
    MostlyIq2S = 28,
    MostlyIq2M = 29,
    MostlyIq4Xs = 30,
    MostlyIq1M = 31,
    MostlyBf16 = 32,
    MostlyTq1_0 = 36,
    MostlyTq2_0 = 37,
    MostlyMxfp4Moe = 38,
    MostlyNvfp4 = 39,
    MostlyQ1_0 = 40,
    MostlyQ2_0 = 41,
    MostlyF8E4M3 = 42,
    Guessed = 1024,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RopeScalingType {
    Unspecified = -1,
    None = 0,
    Linear = 1,
    Yarn = 2,
    Longrope = 3,
}

pub const LLAMA_ROPE_SCALING_TYPE_MAX_VALUE: RopeScalingType = RopeScalingType::Longrope;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AttentionType {
    Unspecified = -1,
    Causal = 0,
    NonCausal = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum FlashAttnType {
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum SplitMode {
    None = 0,
    Layer = 1,
    Row = 2,
    Tensor = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LoadMode {
    Auto = -1,
    None = 0,
    Mmap = 1,
    Mlock = 2,
    MmapMlock = 3,
    DirectIo = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LazyMode {
    Off = 0,
    Auto = 1,
    On = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ContextType {
    Default = 0,
    Mtp = 1,
}

impl Default for DFlashVariant {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DFlashVariant {
    None = -1,
    Dflash = 0,
    Dflash2 = 1,
    Dspark = 2,
    Lilicorr = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ModelKvOverrideType {
    Int = 0,
    Float = 1,
    Bool = 2,
    Str = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ModelMetaKey {
    SamplingSequence = 0,
    SamplingTopK = 1,
    SamplingTopP = 2,
    SamplingMinP = 3,
    SamplingXtcProbability = 4,
    SamplingXtcThreshold = 5,
    SamplingTemp = 6,
    SamplingPenaltyLastN = 7,
    SamplingPenaltyRepeat = 8,
    SamplingMirostat = 9,
    SamplingMirostatTau = 10,
    SamplingMirostatEta = 11,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ProcessType {
    Encode = 0,
    Decode = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    F8E4M3 = 43,
    Count = 44,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum NumaStrategy {
    Disabled = 0,
    Distribute = 1,
    Isolate = 2,
    Numactl = 3,
    Mirror = 4,
    Count = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OptOptimizerType {
    Adamw = 0,
    Sgd = 1,
    Count = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenData {
    pub id: Token,
    pub logit: f32,
    pub p: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TokenDataArray {
    pub data: Vec<TokenData>,
    /// Index into `data`, not a token id.
    pub selected: i64,
    pub sorted: bool,
}

impl TokenDataArray {
    pub fn new(data: Vec<TokenData>) -> Self {
        Self { data, selected: -1, sorted: false }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub n_tokens: i32,
    pub token: Vec<Token>,
    pub embd: Vec<f32>,
    pub pos: Vec<LlamaPos>,
    pub n_seq_id: Vec<i32>,
    pub seq_id: Vec<Vec<LlamaSeqId>>,
    pub logits: Vec<i8>,
}

pub type ProgressCallback = fn(f32) -> bool;
pub type AbortCallback = fn() -> bool;
pub type EvalCallback = fn(&Tensor, bool) -> bool;
pub type ParamFilter = fn(&Tensor) -> bool;

#[derive(Clone, Debug, PartialEq)]
pub enum KvOverrideValue {
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelKvOverride {
    pub tag: ModelKvOverrideType,
    pub key: String,
    pub value: KvOverrideValue,
}

#[derive(Clone, Debug, Default)]
pub struct BackendDevice(pub String);

#[derive(Clone, Debug, Default)]
pub struct Buft(pub String);

#[derive(Clone, Debug)]
pub struct TensorBuftOverride {
    pub pattern: String,
    pub buft: Option<Buft>,
}

#[derive(Clone, Debug)]
pub struct ModelParams {
    pub devices: Vec<BackendDevice>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
    pub n_gpu_layers: i32,
    pub split_mode: SplitMode,
    pub load_mode: LoadMode,
    pub lazy_mode: LazyMode,
    pub main_gpu: i32,
    pub tensor_split: Vec<f32>,
    pub progress_callback: Option<ProgressCallback>,
    pub kv_overrides: Vec<ModelKvOverride>,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            tensor_buft_overrides: Vec::new(),
            n_gpu_layers: -1,
            split_mode: SplitMode::Layer,
            load_mode: LoadMode::Auto,
            lazy_mode: LazyMode::Auto,
            main_gpu: 0,
            tensor_split: Vec::new(),
            progress_callback: None,
            kv_overrides: Vec::new(),
            vocab_only: false,
            check_tensors: false,
            use_extra_bufts: true,
            no_host: false,
            no_alloc: false,
            load_mtp: false,
        }
    }
}

#[derive(Clone)]
pub struct SamplerSeqConfig {
    pub seq_id: LlamaSeqId,
    pub sampler: Sampler,
}

#[derive(Clone)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: ContextType,
    pub rope_scaling_type: RopeScalingType,
    pub pooling_type: PoolingType,
    pub attention_type: AttentionType,
    pub flash_attn_type: FlashAttnType,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub cb_eval: Option<EvalCallback>,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub moe_cache_size: usize,
    pub moe_cache_cpu_share: f32,
    pub abort_callback: Option<AbortCallback>,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: Vec<SamplerSeqConfig>,
    pub ctx_other: Option<Rc<RefCell<Context>>>,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_rs_seq: 0,
            n_outputs_max: 0,
            n_outputs_max_per_seq: 1,
            n_threads: 0,
            n_threads_batch: 0,
            ctx_type: ContextType::Default,
            rope_scaling_type: RopeScalingType::Unspecified,
            pooling_type: PoolingType::Unspecified,
            attention_type: AttentionType::Unspecified,
            flash_attn_type: FlashAttnType::Auto,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: -1.0,
            yarn_beta_fast: -1.0,
            yarn_beta_slow: -1.0,
            yarn_orig_ctx: 0,
            cb_eval: None,
            type_k: GgmlType::F16,
            type_v: GgmlType::F16,
            moe_cache_size: 0,
            moe_cache_cpu_share: 0.0,
            abort_callback: None,
            embeddings: false,
            offload_kqv: true,
            no_perf: true,
            op_offload: true,
            swa_full: true,
            kv_unified: false,
            samplers: Vec::new(),
            ctx_other: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorOverride {
    pub pattern: String,
    pub type_: GgmlType,
}

#[derive(Clone, Debug)]
pub struct ImatrixData {
    pub name: String,
    pub data: Vec<f32>,
    pub size: usize,
}

#[derive(Clone)]
pub struct ModelQuantizeParams {
    pub nthread: i32,
    pub ftype: Ftype,
    pub output_tensor_type: GgmlType,
    pub token_embedding_type: GgmlType,
    pub allow_requantize: bool,
    pub quantize_output_tensor: bool,
    pub only_copy: bool,
    pub pure: bool,
    pub keep_split: bool,
    pub dry_run: bool,
    pub imatrix: Option<ImatrixData>,
    pub kv_overrides: Vec<ModelKvOverride>,
    pub tt_overrides: Vec<TensorOverride>,
    pub prune_layers: Vec<i32>,
    pub max_buf_size: usize,
    pub activations: Option<ImatrixData>,
    pub statistics: Option<ImatrixData>,
    pub stats_schema: Vec<String>,
    pub target_bpw: f32,
    pub target_size: i64,
    pub state_file: String,
    pub target_exclude: Vec<GgmlType>,
    pub abort_callback: Option<AbortCallback>,
}

impl Default for ModelQuantizeParams {
    fn default() -> Self {
        Self {
            nthread: 0,
            ftype: Ftype::MostlyQ8_0,
            output_tensor_type: GgmlType::Count,
            token_embedding_type: GgmlType::Count,
            allow_requantize: false,
            quantize_output_tensor: true,
            only_copy: false,
            pure: false,
            keep_split: false,
            dry_run: false,
            imatrix: None,
            kv_overrides: Vec::new(),
            tt_overrides: Vec::new(),
            prune_layers: Vec::new(),
            max_buf_size: LLAMA_QUANT_MAX_BUF_SIZE,
            activations: None,
            statistics: None,
            stats_schema: Vec::new(),
            target_bpw: -1.0,
            target_size: -1,
            state_file: String::new(),
            target_exclude: Vec::new(),
            abort_callback: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogitBias {
    pub token: Token,
    pub bias: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerChainParams {
    pub no_perf: bool,
}

impl Default for SamplerChainParams {
    fn default() -> Self {
        Self { no_perf: true }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Stand-in for `gguf_context`.
#[derive(Clone, Debug, Default)]
pub struct GgufContext {
    pub kv: HashMap<String, String>,
    pub tensors: Vec<Tensor>,
}

/// Stand-in for `ggml_tensor`.
#[derive(Clone, Debug, Default)]
pub struct Tensor {
    pub name: String,
    pub data: Vec<f32>,
}

/// Stand-in for tensor data supplied to `llama_model_init_from_user`.
#[derive(Clone, Debug, Default)]
pub struct TensorData {
    pub bytes: Vec<u8>,
}

pub type ModelSetTensorData = fn(&mut Tensor, &mut TensorData);

/// Stand-in for `ggml_context`.
#[derive(Clone, Debug, Default)]
pub struct GgmlContext {
    pub tensors: Vec<Tensor>,
}

/// Stand-in for `ggml_cgraph`.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct SamplerData {
    pub logits: Option<Tensor>,
    pub probs: Option<Tensor>,
    pub sampled: Option<Tensor>,
    pub candidates: Option<Tensor>,
}

#[derive(Clone, Debug, Default)]
pub struct ThreadPool {
    pub n_threads: i32,
}

/// Stand-in for the sampler's opaque context pointer.
pub enum SamplerContext {
    Empty,
    Builtin(Box<dyn crate::sampler::Sampler>),
    Chain(ChainSampler),
    Custom(Box<dyn std::any::Any>),
}

impl Clone for SamplerContext {
    fn clone(&self) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Builtin(s) => Self::Builtin(s.clone_sampler()),
            Self::Chain(c) => Self::Chain(c.clone()),
            Self::Custom(_) => Self::Empty,
        }
    }
}

impl SamplerContext {
    pub fn as_builtin(&self) -> Option<&dyn crate::sampler::Sampler> {
        match self {
            Self::Builtin(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub fn as_builtin_mut(&mut self) -> Option<&mut dyn crate::sampler::Sampler> {
        match self {
            Self::Builtin(s) => Some(s.as_mut()),
            _ => None,
        }
    }

    pub fn as_chain(&self) -> Option<&ChainSampler> {
        match self {
            Self::Chain(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_chain_mut(&mut self) -> Option<&mut ChainSampler> {
        match self {
            Self::Chain(c) => Some(c),
            _ => None,
        }
    }

    pub fn custom<T: std::any::Any>(&self) -> Option<&T> {
        match self {
            Self::Custom(b) => b.downcast_ref::<T>(),
            _ => None,
        }
    }

    pub fn custom_mut<T: std::any::Any>(&mut self) -> Option<&mut T> {
        match self {
            Self::Custom(b) => b.downcast_mut::<T>(),
            _ => None,
        }
    }
}

/// Mirror of `llama_sampler_i`. State lives in the `SamplerContext`.
pub trait SamplerI {
    fn name<'a>(&self, _ctx: &'a SamplerContext) -> Option<&'a str> {
        None
    }
    fn accept(&mut self, _ctx: &mut SamplerContext, _token: Token) {}
    fn apply(&mut self, ctx: &mut SamplerContext, cur: &mut TokenDataArray);
    fn reset(&mut self, _ctx: &mut SamplerContext) {}
    fn clone_sampler(&self, _ctx: &SamplerContext) -> Option<Sampler> {
        None
    }
    fn copy_state(&self, _src: &SamplerContext, _dst: &mut SamplerContext) {}

    fn backend_init(
        &mut self,
        _ctx: &mut SamplerContext,
        _buft: &Buft,
        _n_outputs_max_per_seq: u32,
    ) -> bool {
        false
    }
    fn backend_accept(
        &mut self,
        _ctx: &mut SamplerContext,
        _ggml_ctx: &mut GgmlContext,
        _gf: &mut Graph,
        _selected_token: &Tensor,
    ) {
    }
    fn backend_apply(
        &mut self,
        _ctx: &mut SamplerContext,
        _ggml_ctx: &mut GgmlContext,
        _gf: &mut Graph,
        _data: &mut SamplerData,
    ) {
    }
    fn backend_set_input(&mut self, _ctx: &mut SamplerContext) {}
    fn backend_reset(&mut self, _ctx: &mut SamplerContext) {}
}

pub struct Sampler {
    pub iface: Box<dyn SamplerI>,
    pub ctx: SamplerContext,
}

impl Clone for Sampler {
    fn clone(&self) -> Self {
        self.iface.clone_sampler(&self.ctx).unwrap_or_else(|| Sampler {
            iface: Box::new(BuiltinIface),
            ctx: SamplerContext::Empty,
        })
    }
}

impl Sampler {
    pub fn new(iface: Box<dyn SamplerI>, ctx: SamplerContext) -> Self {
        Self { iface, ctx }
    }

    pub fn name(&self) -> Option<&str> {
        self.iface.name(&self.ctx)
    }

    pub fn apply(&mut self, cur: &mut TokenDataArray) {
        self.iface.apply(&mut self.ctx, cur);
    }
}

#[derive(Clone)]
pub struct ChainSampler {
    pub params: SamplerChainParams,
    pub samplers: Vec<Sampler>,
    pub cur: Vec<TokenData>,
    pub t_sample_ms: f64,
    pub n_sample: i32,
}

impl ChainSampler {
    pub fn new(params: SamplerChainParams) -> Self {
        Self { params, samplers: Vec::new(), cur: Vec::new(), t_sample_ms: 0.0, n_sample: 0 }
    }

    pub fn len(&self) -> i32 {
        self.samplers.len() as i32
    }

    pub fn is_empty(&self) -> bool {
        self.samplers.is_empty()
    }
}

struct BuiltinIface;

fn to_inner_array(cur: &TokenDataArray) -> crate::sampler::TokenDataArray {
    let selected = cur.selected.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    crate::sampler::TokenDataArray {
        data: cur
            .data
            .iter()
            .map(|t| crate::sampler::TokenData { id: t.id, logit: t.logit, p: t.p })
            .collect(),
        selected,
        sorted: cur.sorted,
    }
}

fn from_inner_array(cur: crate::sampler::TokenDataArray) -> TokenDataArray {
    TokenDataArray {
        data: cur
            .data
            .into_iter()
            .map(|t| TokenData { id: t.id, logit: t.logit, p: t.p })
            .collect(),
        selected: cur.selected as i64,
        sorted: cur.sorted,
    }
}

fn builtin_sampler(sampler: Box<dyn crate::sampler::Sampler>) -> Sampler {
    Sampler::new(Box::new(BuiltinIface), SamplerContext::Builtin(sampler))
}

impl SamplerI for BuiltinIface {
    fn name<'a>(&self, ctx: &'a SamplerContext) -> Option<&'a str> {
        ctx.as_builtin().map(|s| s.name())
    }

    fn accept(&mut self, ctx: &mut SamplerContext, token: Token) {
        if let Some(s) = ctx.as_builtin_mut() {
            s.accept(token);
        }
    }

    fn apply(&mut self, ctx: &mut SamplerContext, cur: &mut TokenDataArray) {
        if let Some(s) = ctx.as_builtin_mut() {
            let mut inner = to_inner_array(cur);
            s.apply(&mut inner);
            *cur = from_inner_array(inner);
        }
    }

    fn reset(&mut self, ctx: &mut SamplerContext) {
        if let Some(s) = ctx.as_builtin_mut() {
            s.reset();
        }
    }

    fn clone_sampler(&self, ctx: &SamplerContext) -> Option<Sampler> {
        ctx.as_builtin().map(|s| builtin_sampler(s.clone_sampler()))
    }

    fn copy_state(&self, src: &SamplerContext, dst: &mut SamplerContext) {
        if let Some(s) = src.as_builtin() {
            *dst = SamplerContext::Builtin(s.clone_sampler());
        }
    }

    fn backend_init(
        &mut self,
        ctx: &mut SamplerContext,
        _buft: &Buft,
        _n_outputs_max_per_seq: u32,
    ) -> bool {
        if let Some(s) = ctx.as_builtin_mut() {
            s.backend_begin();
            true
        } else {
            false
        }
    }
}

struct ChainIface;

impl SamplerI for ChainIface {
    fn name<'a>(&self, _ctx: &'a SamplerContext) -> Option<&'a str> {
        Some("chain")
    }

    fn apply(&mut self, ctx: &mut SamplerContext, cur: &mut TokenDataArray) {
        let start = Instant::now();
        if let Some(chain) = ctx.as_chain_mut() {
            for sampler in chain.samplers.iter_mut() {
                sampler.apply(cur);
                if cur.selected >= 0 {
                    break;
                }
            }
            chain.cur = cur.data.clone();
            chain.n_sample += 1;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if !chain.params.no_perf {
                chain.t_sample_ms += elapsed;
            }
        }
    }

    fn accept(&mut self, ctx: &mut SamplerContext, token: Token) {
        if let Some(chain) = ctx.as_chain_mut() {
            for sampler in chain.samplers.iter_mut() {
                sampler.iface.accept(&mut sampler.ctx, token);
            }
        }
    }

    fn reset(&mut self, ctx: &mut SamplerContext) {
        if let Some(chain) = ctx.as_chain_mut() {
            for sampler in chain.samplers.iter_mut() {
                sampler.iface.reset(&mut sampler.ctx);
            }
        }
    }

    fn clone_sampler(&self, ctx: &SamplerContext) -> Option<Sampler> {
        ctx.as_chain().map(|c| Sampler::new(Box::new(ChainIface), SamplerContext::Chain(c.clone())))
    }

    fn copy_state(&self, src: &SamplerContext, dst: &mut SamplerContext) {
        if let (Some(s), Some(d)) = (src.as_chain(), dst.as_chain_mut()) {
            d.cur = s.cur.clone();
            d.n_sample = s.n_sample;
            d.t_sample_ms = s.t_sample_ms;
        }
    }
}

fn as_chain_mut(sampler: &mut Sampler) -> Option<&mut ChainSampler> {
    if sampler.name() == Some("chain") {
        sampler.ctx.as_chain_mut()
    } else {
        None
    }
}

/// One token recorded in the KV-cache stand-in.
#[derive(Clone, Debug)]
pub struct MemoryToken {
    pub id: Token,
    pub pos: LlamaPos,
    pub seq_ids: Vec<LlamaSeqId>,
}

#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub tokens: Vec<MemoryToken>,
    /// Sliding-window size used by the `PARTIAL_ONLY` state flag.
    pub swa: i32,
}

#[derive(Clone, Debug, Default)]
pub struct SpecialTokens {
    pub bos: Token,
    pub eos: Token,
    pub eot: Token,
    pub sep: Token,
    pub nl: Token,
    pub pad: Token,
    pub mask: Token,
    pub fim_pre: Token,
    pub fim_suf: Token,
    pub fim_mid: Token,
    pub fim_pad: Token,
    pub fim_rep: Token,
    pub fim_sep: Token,
}

#[derive(Clone, Debug)]
pub struct Vocab {
    pub pieces: Vec<String>,
    pub scores: Vec<f32>,
    pub attrs: Vec<TokenAttr>,
    pub special: SpecialTokens,
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub suppress: Vec<Token>,
    vocab_type: VocabType,
}

impl Default for Vocab {
    fn default() -> Self {
        let pieces = (0..=255u32).map(|i| format!(" {}", i as u8 as char)).collect();
        Self::new(VocabType::Spm, pieces)
    }
}

impl Vocab {
    /// Byte-level stand-in vocabulary. Token 0 is unknown; byte tokens follow.
    pub fn new(vocab_type: VocabType, pieces: Vec<String>) -> Self {
        let n = pieces.len();
        let attrs = (0..n)
            .map(|i| {
                if (1..=13).contains(&i) {
                    TokenAttr::from_flag(TokenAttrFlag::Control)
                } else {
                    TokenAttr::from_flag(TokenAttrFlag::Normal)
                }
            })
            .collect();
        Self {
            pieces,
            scores: vec![0.0; n],
            attrs,
            special: SpecialTokens {
                bos: 1,
                eos: 2,
                eot: 3,
                sep: 4,
                nl: 5,
                pad: 6,
                mask: 7,
                fim_pre: 8,
                fim_suf: 9,
                fim_mid: 10,
                fim_pad: 11,
                fim_rep: 12,
                fim_sep: 13,
            },
            add_bos: false,
            add_eos: false,
            add_sep: false,
            suppress: Vec::new(),
            vocab_type,
        }
    }

    fn longest_match(&self, bytes: &[u8]) -> Option<(Token, usize)> {
        let mut best: Option<(Token, usize)> = None;
        for (id, piece) in self.pieces.iter().enumerate() {
            let p = piece.as_bytes();
            if p.len() <= bytes.len() && bytes.starts_with(p) {
                if best.map_or(true, |(_, n)| p.len() > n) {
                    best = Some((id as Token, p.len()));
                }
            }
        }
        best
    }
}

impl crate::sampler::Vocab for Vocab {
    fn n_tokens(&self) -> usize {
        self.pieces.len()
    }
    fn detokenize(&self, token: Token, _special: bool) -> Vec<u8> {
        self.pieces.get(token as usize).map(|s| s.as_bytes().to_vec()).unwrap_or_default()
    }
    fn tokenize(&self, text: &[u8], bos: bool, _special: bool) -> Vec<Token> {
        let mut out = Vec::new();
        if bos {
            out.push(self.special.bos);
        }
        let mut i = 0;
        while i < text.len() {
            match self.longest_match(&text[i..]) {
                Some((id, n)) if n > 0 => {
                    out.push(id);
                    i += n;
                }
                _ => {
                    out.push(0);
                    i += 1;
                }
            }
        }
        out
    }
    fn is_eog(&self, token: Token) -> bool {
        token == self.special.eos || token == self.special.eot
    }
    fn token_eot(&self) -> Token {
        self.special.eot
    }
    fn token_eos(&self) -> Token {
        self.special.eos
    }
    fn token_to_piece(&self, token: Token) -> Vec<u8> {
        self.detokenize(token, false)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub vocab: Vocab,
    pub meta: HashMap<String, String>,
    pub desc: String,
    pub ftype: Ftype,
    pub size: u64,
    pub n_params: u64,
    pub n_ctx_train: i32,
    pub n_embd: i32,
    pub n_embd_inp: i32,
    pub n_embd_out: i32,
    pub n_layer: i32,
    pub n_layer_nextn: i32,
    pub n_head: i32,
    pub n_head_kv: i32,
    pub n_swa: i32,
    pub rope_freq_scale_train: f32,
    pub rope_type: RopeType,
    pub n_cls_out: u32,
    pub cls_labels: Vec<String>,
    pub chat_templates: HashMap<String, String>,
    pub has_encoder: bool,
    pub has_decoder: bool,
    pub decoder_start_token: Token,
    pub is_recurrent: bool,
    pub is_hybrid: bool,
    pub is_diffusion: bool,
    pub dflash_variant: DFlashVariant,
    pub dflash_selector_top_k: i32,
    pub target_layer_ids: Vec<i32>,
    pub path: String,
    pub bytes: Vec<u8>,
    pub params: ModelParams,
    pub adapters: Vec<AdapterLora>,
}

#[derive(Clone, Debug, Default)]
pub struct AdapterLora {
    pub path: String,
    pub meta: HashMap<String, String>,
    pub invocation_tokens: Vec<Token>,
    pub rank: i32,
}

#[derive(Clone, Debug, Default)]
pub struct PerfContextData {
    pub t_start_ms: f64,
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
    pub n_reused: i32,
    pub n_reallocs_same: i32,
    pub n_reallocs_diff: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PerfSamplerData {
    pub t_sample_ms: f64,
    pub n_sample: i32,
}

#[derive(Clone, Debug, Default)]
pub struct SampledOutputs {
    pub tokens: Vec<Token>,
    pub probs: Vec<f32>,
    pub logits: Vec<f32>,
    pub candidates: Vec<Token>,
    pub counts: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct OptResult {
    pub loss: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub rows: usize,
}

#[derive(Clone)]
pub struct OptParams {
    pub n_ctx_train: u32,
    pub param_filter: Option<ParamFilter>,
    pub get_opt_pars: Option<fn() -> usize>,
    pub optimizer_type: OptOptimizerType,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            n_ctx_train: 0,
            param_filter: None,
            get_opt_pars: None,
            optimizer_type: OptOptimizerType::Adamw,
        }
    }
}

pub type EpochCallback = fn(&Dataset, &OptResult) -> bool;

pub struct Context {
    pub model: Model,
    pub params: ContextParams,
    pub memory: Memory,
    pub logits: Vec<f32>,
    pub embeddings: Vec<f32>,
    pub embeddings_nextn: Vec<f32>,
    pub embeddings_layer_inp: Vec<f32>,
    pub sampled: SampledOutputs,
    pub n_ctx: u32,
    pub n_ctx_seq: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub causal_attn: bool,
    pub embeddings_enabled: bool,
    pub embeddings_nextn_enabled: bool,
    pub embeddings_nextn_masked: bool,
    pub nextn_layer_offset: i32,
    pub embeddings_layer_inp_enabled: Vec<bool>,
    pub embeddings_layer_inp_dev: bool,
    pub embeddings_nextn_dev: bool,
    pub abort_callback: Option<AbortCallback>,
    pub threadpool: Option<ThreadPool>,
    pub threadpool_batch: Option<ThreadPool>,
    pub perf: PerfContextData,
    pub samplers: Vec<SamplerSeqConfig>,
    pub ctx_other: Option<Rc<RefCell<Context>>>,
    pub ctx_state_src: Option<Rc<RefCell<Context>>>,
    pub opt_params: Option<OptParams>,
    pub enc_output: Vec<f32>,
    pub cvec: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct BatchEntry {
    pub id: Token,
    pub embd: Vec<f32>,
    pub embd_state: Vec<f32>,
    pub embd_row: Option<i32>,
    pub embd_state_row: Option<i32>,
    pub pos: Vec<LlamaPos>,
    pub seq_ids: Vec<LlamaSeqId>,
    pub output_embd: bool,
    pub output_logits: bool,
}

#[derive(Clone, Debug)]
pub struct Embedding {
    pub data: Vec<f32>,
    pub n_rows: usize,
    pub n_embd: usize,
}

impl Embedding {
    pub fn new(data: Vec<f32>, n_rows: usize, n_embd: usize) -> Self {
        Self { data, n_rows, n_embd }
    }
}

#[derive(Clone, Debug)]
pub struct BatchExt {
    pub entries: Vec<BatchEntry>,
    pub capacity: usize,
    pub embd_size: usize,
    pub n_seq_max: usize,
    pub rows: Vec<Vec<f32>>,
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_i32(out: &mut Vec<u8>, v: i32) {
    push_u32(out, v as u32);
}

fn take_bytes<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(n)?;
    let slice = data.get(*pos..end)?;
    *pos = end;
    Some(slice)
}

fn pop_u32(data: &[u8], pos: &mut usize) -> Option<u32> {
    let b = take_bytes(data, pos, 4)?;
    let mut a = [0u8; 4];
    a.copy_from_slice(b);
    Some(u32::from_le_bytes(a))
}

fn pop_u64(data: &[u8], pos: &mut usize) -> Option<u64> {
    let b = take_bytes(data, pos, 8)?;
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    Some(u64::from_le_bytes(a))
}

fn pop_i32(data: &[u8], pos: &mut usize) -> Option<i32> {
    pop_u32(data, pos).map(|v| v as i32)
}

fn write_c_str(dst: &mut [u8], value: &str) -> i32 {
    for (out, b) in dst.iter_mut().zip(value.as_bytes().iter()) {
        *out = *b;
    }
    if let Some(last) = dst.last_mut() {
        *last = 0;
    }
    value.len() as i32
}

fn clear_buf(dst: &mut [u8]) {
    if let Some(first) = dst.first_mut() {
        *first = 0;
    }
}

fn now_ms() -> f64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs_f64() * 1000.0,
        Err(_) => 0.0,
    }
}

/// `llama_version`.
pub fn llama_version() -> &'static str {
    LLAMA_VERSION
}

/// `llama_ftype_name`.
pub fn llama_ftype_name(ftype: Ftype) -> &'static str {
    let guessed = (ftype as i32) & (Ftype::Guessed as i32) != 0;
    let (base, guessed_name) = match (ftype as i32) & !(Ftype::Guessed as i32) {
        0 => ("all F32", "(guessed) all F32"),
        1 => ("F16", "(guessed) F16"),
        2 => ("Q4_0", "(guessed) Q4_0"),
        3 => ("Q4_1", "(guessed) Q4_1"),
        7 => ("Q8_0", "(guessed) Q8_0"),
        8 => ("Q5_0", "(guessed) Q5_0"),
        9 => ("Q5_1", "(guessed) Q5_1"),
        10 => ("Q2_K - Medium", "(guessed) Q2_K - Medium"),
        11 => ("Q3_K - Small", "(guessed) Q3_K - Small"),
        12 => ("Q3_K - Medium", "(guessed) Q3_K - Medium"),
        13 => ("Q3_K - Large", "(guessed) Q3_K - Large"),
        14 => ("Q4_K - Small", "(guessed) Q4_K - Small"),
        15 => ("Q4_K - Medium", "(guessed) Q4_K - Medium"),
        16 => ("Q5_K - Small", "(guessed) Q5_K - Small"),
        17 => ("Q5_K - Medium", "(guessed) Q5_K - Medium"),
        18 => ("Q6_K", "(guessed) Q6_K"),
        19 => ("IQ2_XXS - 2.0625 bpw", "(guessed) IQ2_XXS - 2.0625 bpw"),
        20 => ("IQ2_XS - 2.3125 bpw", "(guessed) IQ2_XS - 2.3125 bpw"),
        21 => ("Q2_K - Small", "(guessed) Q2_K - Small"),
        22 => ("IQ3_XS - 3.3 bpw", "(guessed) IQ3_XS - 3.3 bpw"),
        23 => ("IQ3_XXS - 3.0625 bpw", "(guessed) IQ3_XXS - 3.0625 bpw"),
        24 => ("IQ1_S - 1.5625 bpw", "(guessed) IQ1_S - 1.5625 bpw"),
        25 => ("IQ4_NL - 4.5 bpw", "(guessed) IQ4_NL - 4.5 bpw"),
        26 => ("IQ3_S - 3.4375 bpw", "(guessed) IQ3_S - 3.4375 bpw"),
        27 => ("IQ3_S mix - 3.66 bpw", "(guessed) IQ3_S mix - 3.66 bpw"),
        28 => ("IQ2_S - 2.5 bpw", "(guessed) IQ2_S - 2.5 bpw"),
        29 => ("IQ2_M - 2.7 bpw", "(guessed) IQ2_M - 2.7 bpw"),
        30 => ("IQ4_XS - 4.25 bpw", "(guessed) IQ4_XS - 4.25 bpw"),
        31 => ("IQ1_M - 1.75 bpw", "(guessed) IQ1_M - 1.75 bpw"),
        32 => ("BF16", "(guessed) BF16"),
        36 => ("TQ1_0 - 1.69 bpw ternary", "(guessed) TQ1_0 - 1.69 bpw ternary"),
        37 => ("TQ2_0 - 2.06 bpw ternary", "(guessed) TQ2_0 - 2.06 bpw ternary"),
        38 => ("MXFP4 MoE", "(guessed) MXFP4 MoE"),
        39 => ("NVFP4", "(guessed) NVFP4"),
        40 => ("Q1_0", "(guessed) Q1_0"),
        41 => ("Q2_0", "(guessed) Q2_0"),
        42 => ("F8_E4M3", "(guessed) F8_E4M3"),
        _ => ("unknown, may not work", "(guessed) unknown, may not work"),
    };
    if guessed { guessed_name } else { base }
}

/// `llama_flash_attn_type_name`.
pub fn llama_flash_attn_type_name(kind: FlashAttnType) -> &'static str {
    match kind {
        FlashAttnType::Auto => "auto",
        FlashAttnType::Disabled => "disabled",
        FlashAttnType::Enabled => "enabled",
    }
}

/// `llama_load_mode_name`.
pub fn llama_load_mode_name(mode: LoadMode) -> &'static str {
    match mode {
        LoadMode::Auto => "auto",
        LoadMode::None => "none",
        LoadMode::Mmap => "mmap",
        LoadMode::Mlock => "mlock",
        LoadMode::MmapMlock => "mmap+mlock",
        LoadMode::DirectIo => "dio",
    }
}

/// `llama_load_mode_from_str`.
pub fn llama_load_mode_from_str(s: &str) -> Result<LoadMode> {
    match s {
        "auto" => Ok(LoadMode::Auto),
        "none" => Ok(LoadMode::None),
        "mmap" => Ok(LoadMode::Mmap),
        "mlock" => Ok(LoadMode::Mlock),
        "mmap+mlock" => Ok(LoadMode::MmapMlock),
        "dio" => Ok(LoadMode::DirectIo),
        _ => Err(Error::new(format!("unknown load mode: {}", s))),
    }
}

pub fn llama_model_default_params() -> ModelParams {
    ModelParams::default()
}

pub fn llama_context_default_params() -> ContextParams {
    ContextParams::default()
}

pub fn llama_sampler_chain_default_params() -> SamplerChainParams {
    SamplerChainParams::default()
}

pub fn llama_model_quantize_default_params() -> ModelQuantizeParams {
    ModelQuantizeParams::default()
}

pub fn llama_backend_init() {}

pub fn llama_backend_free() {}

pub fn llama_numa_init(_numa: NumaStrategy) {}

pub fn llama_attach_threadpool(
    ctx: &mut Context,
    threadpool: Option<&ThreadPool>,
    threadpool_batch: Option<&ThreadPool>,
) {
    ctx.threadpool = threadpool.cloned();
    ctx.threadpool_batch = threadpool_batch.cloned();
}

pub fn llama_detach_threadpool(ctx: &mut Context) {
    ctx.threadpool = None;
    ctx.threadpool_batch = None;
}

pub fn llama_time_us() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_micros().min(i64::MAX as u128) as i64,
        Err(_) => 0,
    }
}

pub fn llama_max_devices() -> usize {
    LLAMA_MAX_DEVICES
}

pub fn llama_max_parallel_sequences() -> usize {
    crate::cparams::LLAMA_MAX_SEQ as usize
}

pub fn llama_max_tensor_buft_overrides() -> usize {
    LLAMA_MAX_TENSOR_BUFT_OVERRIDES
}

pub fn llama_supports_mmap() -> bool {
    cfg!(any(unix, windows))
}

pub fn llama_supports_mlock() -> bool {
    cfg!(unix)
}

pub fn llama_supports_gpu_offload() -> bool {
    false
}

pub fn llama_supports_rpc() -> bool {
    false
}

fn model_from_bytes(path: &str, bytes: Vec<u8>, params: ModelParams) -> Result<Model> {
    if let Some(cb) = params.progress_callback {
        if !cb(0.25) {
            return Err(Error::new("model loading aborted"));
        }
        if !cb(1.0) {
            return Err(Error::new("model loading aborted"));
        }
    }
    let mut model = Model {
        vocab: Vocab::default(),
        meta: HashMap::new(),
        desc: String::new(),
        ftype: Ftype::MostlyQ8_0,
        size: bytes.len() as u64,
        n_params: 0,
        n_ctx_train: 4096,
        n_embd: 4096,
        n_embd_inp: 4096,
        n_embd_out: 4096,
        n_layer: 32,
        n_layer_nextn: 0,
        n_head: 32,
        n_head_kv: 8,
        n_swa: 0,
        rope_freq_scale_train: 1.0,
        rope_type: RopeType::Norm,
        n_cls_out: 0,
        cls_labels: Vec::new(),
        chat_templates: HashMap::new(),
        has_encoder: false,
        has_decoder: true,
        decoder_start_token: LLAMA_TOKEN_NULL,
        is_recurrent: false,
        is_hybrid: false,
        is_diffusion: false,
        dflash_variant: DFlashVariant::None,
        dflash_selector_top_k: 0,
        target_layer_ids: Vec::new(),
        path: path.to_owned(),
        bytes,
        params,
        adapters: Vec::new(),
    };
    for kv in &model.params.kv_overrides {
        model.meta.insert(kv.key.clone(), match &kv.value {
            KvOverrideValue::Str(s) => s.clone(),
            KvOverrideValue::I64(v) => v.to_string(),
            KvOverrideValue::F64(v) => v.to_string(),
            KvOverrideValue::Bool(v) => v.to_string(),
        });
    }
    if let Some(v) = model.meta.get("general.file_type").and_then(|s| s.parse::<i32>().ok()) {
        if let Some(ftype) = all_ftypes().into_iter().find(|f| *f as i32 == v) {
            model.ftype = ftype;
        }
    }
    model.desc = format!("{} ({} bytes)", model.path, model.size);
    Ok(model)
}

fn all_ftypes() -> Vec<Ftype> {
    vec![
        Ftype::AllF32, Ftype::MostlyF16, Ftype::MostlyQ4_0, Ftype::MostlyQ4_1,
        Ftype::MostlyQ8_0, Ftype::MostlyQ5_0, Ftype::MostlyQ5_1, Ftype::MostlyQ2K,
        Ftype::MostlyQ3KS, Ftype::MostlyQ3KM, Ftype::MostlyQ3KL, Ftype::MostlyQ4KS,
        Ftype::MostlyQ4KM, Ftype::MostlyQ5KS, Ftype::MostlyQ5KM, Ftype::MostlyQ6K,
        Ftype::MostlyIq2Xxs, Ftype::MostlyIq2Xs, Ftype::MostlyQ2KS, Ftype::MostlyIq3Xs,
        Ftype::MostlyIq3Xxs, Ftype::MostlyIq1S, Ftype::MostlyIq4Nl, Ftype::MostlyIq3S,
        Ftype::MostlyIq3M, Ftype::MostlyIq2S, Ftype::MostlyIq2M, Ftype::MostlyIq4Xs,
        Ftype::MostlyIq1M, Ftype::MostlyBf16, Ftype::MostlyTq1_0, Ftype::MostlyTq2_0,
        Ftype::MostlyMxfp4Moe, Ftype::MostlyNvfp4, Ftype::MostlyQ1_0, Ftype::MostlyQ2_0,
        Ftype::MostlyF8E4M3, Ftype::Guessed,
    ]
}

/// `llama_model_init_from_user`.
pub fn llama_model_init_from_user(
    metadata: &GgufContext,
    set_tensor_data: Option<ModelSetTensorData>,
    userdata: &mut TensorData,
    params: ModelParams,
) -> Result<Model> {
    let mut model = model_from_bytes("<user>", Vec::new(), params)?;
    model.meta = metadata.kv.clone();
    model.bytes = metadata
        .kv
        .iter()
        .map(|(k, v)| format!("{}={};", k, v))
        .collect::<String>()
        .into_bytes();
    if let Some(cb) = set_tensor_data {
        let mut tensors = metadata.tensors.clone();
        for tensor in &mut tensors {
            cb(tensor, userdata);
        }
    }
    Ok(model)
}

/// `llama_model_load_from_file`.
pub fn llama_model_load_from_file(path: &str, params: ModelParams) -> Result<Model> {
    let bytes = fs::read(path).map_err(|e| Error::new(format!("failed to read {}: {}", path, e)))?;
    model_from_bytes(path, bytes, params)
}

/// `llama_model_load_from_file_ptr`.
pub fn llama_model_load_from_file_ptr(file: &mut File, params: ModelParams) -> Result<Model> {
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| Error::new(format!("failed to read model FILE: {}", e)))?;
    model_from_bytes("<file pointer>", bytes, params)
}

/// `llama_model_load_from_splits`.
pub fn llama_model_load_from_splits(paths: &[&str], params: ModelParams) -> Result<Model> {
    let mut bytes = Vec::new();
    for path in paths {
        let part = fs::read(path)
            .map_err(|e| Error::new(format!("failed to read {}: {}", path, e)))?;
        bytes.extend_from_slice(&part);
    }
    let joined = paths.join("|");
    model_from_bytes(&joined, bytes, params)
}

/// `llama_model_save_to_file`.
pub fn llama_model_save_to_file(model: &Model, path: &str) -> Result<()> {
    fs::write(path, &model.bytes).map_err(|e| Error::new(format!("failed to write {}: {}", path, e)))
}

/// `llama_model_free`. Rust drops the model when this function consumes it.
pub fn llama_model_free(_model: Model) {}

/// `llama_init_from_model`.
pub fn llama_init_from_model(model: &Model, params: ContextParams) -> Result<Context> {
    if model.vocab.pieces.is_empty() {
        return Err(Error::new("model has no vocabulary"));
    }
    let n_ctx = if params.n_ctx == 0 { model.n_ctx_train.max(1) as u32 } else { params.n_ctx };
    let n_batch = if params.n_batch == 0 { n_ctx } else { params.n_batch };
    let n_ubatch = if params.n_ubatch == 0 { n_batch } else { params.n_ubatch };
    let n_outputs_max = if params.n_outputs_max == 0 { n_batch } else { params.n_outputs_max };
    let mut ctx_params = params;
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_ubatch;
    ctx_params.n_outputs_max = n_outputs_max;
    let samplers = ctx_params.samplers.clone();
    let ctx_other = ctx_params.ctx_other.clone();
    let n_threads = ctx_params.n_threads;
    let n_threads_batch = ctx_params.n_threads_batch;
    let abort_callback = ctx_params.abort_callback;
    let mut memory = Memory::default();
    memory.swa = model.n_swa;
    Ok(Context {
        model: model.clone(),
        params: ctx_params.clone(),
        memory,
        logits: Vec::new(),
        embeddings: Vec::new(),
        embeddings_nextn: Vec::new(),
        embeddings_layer_inp: Vec::new(),
        sampled: SampledOutputs::default(),
        n_ctx,
        n_ctx_seq: n_ctx,
        n_batch,
        n_ubatch,
        n_seq_max: ctx_params.n_seq_max,
        n_rs_seq: ctx_params.n_rs_seq,
        n_threads,
        n_threads_batch,
        causal_attn: ctx_params.attention_type != AttentionType::NonCausal,
        embeddings_enabled: ctx_params.embeddings,
        embeddings_nextn_enabled: false,
        embeddings_nextn_masked: false,
        nextn_layer_offset: 0,
        embeddings_layer_inp_enabled: Vec::new(),
        embeddings_layer_inp_dev: false,
        embeddings_nextn_dev: false,
        abort_callback,
        threadpool: None,
        threadpool_batch: None,
        perf: PerfContextData {
            t_start_ms: now_ms(),
            n_reused: 0,
            ..PerfContextData::default()
        },
        samplers,
        ctx_other,
        ctx_state_src: None,
        opt_params: None,
        enc_output: Vec::new(),
        cvec: None,
    })
}

/// `llama_free`. Rust drops the context when this function consumes it.
pub fn llama_free(_ctx: Context) {}

pub fn llama_n_ctx(ctx: &Context) -> u32 {
    ctx.n_ctx
}

pub fn llama_n_ctx_seq(ctx: &Context) -> u32 {
    ctx.n_ctx_seq
}

pub fn llama_n_batch(ctx: &Context) -> u32 {
    ctx.n_batch
}

pub fn llama_n_ubatch(ctx: &Context) -> u32 {
    ctx.n_ubatch
}

pub fn llama_n_seq_max(ctx: &Context) -> u32 {
    ctx.n_seq_max
}

pub fn llama_n_rs_seq(ctx: &Context) -> u32 {
    ctx.n_rs_seq
}

pub fn llama_get_model(ctx: &Context) -> &Model {
    &ctx.model
}

pub fn llama_get_memory(ctx: &mut Context) -> &mut Memory {
    &mut ctx.memory
}

pub fn llama_pooling_type(ctx: &Context) -> PoolingType {
    ctx.params.pooling_type
}

pub fn llama_model_get_vocab(model: &Model) -> &Vocab {
    &model.vocab
}

pub fn llama_model_rope_type(model: &Model) -> RopeType {
    model.rope_type
}

pub fn llama_model_n_ctx_train(model: &Model) -> i32 {
    model.n_ctx_train
}

pub fn llama_model_n_embd(model: &Model) -> i32 {
    model.n_embd
}

pub fn llama_model_n_embd_inp(model: &Model) -> i32 {
    model.n_embd_inp
}

pub fn llama_model_n_embd_out(model: &Model) -> i32 {
    model.n_embd_out
}

pub fn llama_model_n_layer(model: &Model) -> i32 {
    model.n_layer
}

pub fn llama_model_n_layer_nextn(model: &Model) -> i32 {
    model.n_layer_nextn
}

pub fn llama_model_n_head(model: &Model) -> i32 {
    model.n_head
}

pub fn llama_model_n_head_kv(model: &Model) -> i32 {
    model.n_head_kv
}

pub fn llama_model_n_swa(model: &Model) -> i32 {
    model.n_swa
}

pub fn llama_model_rope_freq_scale_train(model: &Model) -> f32 {
    model.rope_freq_scale_train
}

pub fn llama_model_n_cls_out(model: &Model) -> u32 {
    model.n_cls_out
}

pub fn llama_model_cls_label(model: &Model, i: u32) -> Option<&str> {
    model.cls_labels.get(i as usize).map(String::as_str)
}

pub fn llama_vocab_type(vocab: &Vocab) -> VocabType {
    vocab.vocab_type
}

pub fn llama_vocab_n_tokens(vocab: &Vocab) -> i32 {
    vocab.pieces.len() as i32
}

pub fn llama_model_meta_val_str(model: &Model, key: &str, buf: &mut [u8]) -> i32 {
    match model.meta.get(key) {
        Some(value) => write_c_str(buf, value),
        None => {
            clear_buf(buf);
            -1
        }
    }
}

pub fn llama_model_meta_count(model: &Model) -> i32 {
    model.meta.len() as i32
}

pub fn llama_model_meta_key_str(key: ModelMetaKey) -> Option<&'static str> {
    match key {
        ModelMetaKey::SamplingSequence => Some("general.sampling.sequence"),
        ModelMetaKey::SamplingTopK => Some("general.sampling.top_k"),
        ModelMetaKey::SamplingTopP => Some("general.sampling.top_p"),
        ModelMetaKey::SamplingMinP => Some("general.sampling.min_p"),
        ModelMetaKey::SamplingXtcProbability => Some("general.sampling.xtc_probability"),
        ModelMetaKey::SamplingXtcThreshold => Some("general.sampling.xtc_threshold"),
        ModelMetaKey::SamplingTemp => Some("general.sampling.temp"),
        ModelMetaKey::SamplingPenaltyLastN => Some("general.sampling.penalty_last_n"),
        ModelMetaKey::SamplingPenaltyRepeat => Some("general.sampling.penalty_repeat"),
        ModelMetaKey::SamplingMirostat => Some("general.sampling.mirostat"),
        ModelMetaKey::SamplingMirostatTau => Some("general.sampling.mirostat_tau"),
        ModelMetaKey::SamplingMirostatEta => Some("general.sampling.mirostat_eta"),
    }
}

fn sorted_meta_keys(model: &Model) -> Vec<&String> {
    let mut keys: Vec<&String> = model.meta.keys().collect();
    keys.sort();
    keys
}

pub fn llama_model_meta_key_by_index(model: &Model, i: i32, buf: &mut [u8]) -> i32 {
    let keys = sorted_meta_keys(model);
    match usize::try_from(i).ok().and_then(|i| keys.get(i)) {
        Some(key) => write_c_str(buf, key),
        None => {
            clear_buf(buf);
            -1
        }
    }
}

pub fn llama_model_meta_val_str_by_index(model: &Model, i: i32, buf: &mut [u8]) -> i32 {
    let keys = sorted_meta_keys(model);
    match usize::try_from(i).ok().and_then(|i| keys.get(i)).and_then(|k| model.meta.get(k.as_str())) {
        Some(value) => write_c_str(buf, value),
        None => {
            clear_buf(buf);
            -1
        }
    }
}

pub fn llama_model_meta_arr_n(model: &Model, key: &str) -> i32 {
    match model.meta.get(key) {
        Some(value) => value.split(';').count() as i32,
        None => -1,
    }
}

pub fn llama_model_meta_arr_str<'a>(model: &'a Model, key: &str, i: i32) -> Option<&'a str> {
    model
        .meta
        .get(key)
        .and_then(|v| v.split(';').nth(usize::try_from(i).unwrap_or(usize::MAX)))
}

pub fn llama_model_desc(model: &Model, buf: &mut [u8]) -> i32 {
    write_c_str(buf, &model.desc)
}

pub fn llama_model_ftype(model: &Model) -> Ftype {
    model.ftype
}

pub fn llama_model_size(model: &Model) -> u64 {
    model.size
}

pub fn llama_model_chat_template<'a>(model: &'a Model, name: Option<&str>) -> Option<&'a str> {
    match name {
        Some(name) => model.chat_templates.get(name).map(String::as_str),
        None => model.chat_templates.get("default").map(String::as_str),
    }
}

pub fn llama_model_n_params(model: &Model) -> u64 {
    model.n_params
}

pub fn llama_model_has_encoder(model: &Model) -> bool {
    model.has_encoder
}

pub fn llama_model_has_decoder(model: &Model) -> bool {
    model.has_decoder
}

pub fn llama_model_decoder_start_token(model: &Model) -> Token {
    model.decoder_start_token
}

pub fn llama_model_is_recurrent(model: &Model) -> bool {
    model.is_recurrent
}

pub fn llama_model_is_hybrid(model: &Model) -> bool {
    model.is_hybrid
}

pub fn llama_model_is_diffusion(model: &Model) -> bool {
    model.is_diffusion
}

/// `llama_model_quantize`. The stand-in copies the file unless `dry_run`.
pub fn llama_model_quantize(
    fname_inp: &str,
    fname_out: &str,
    params: &ModelQuantizeParams,
) -> u32 {
    let bytes = match fs::read(fname_inp) {
        Ok(b) => b,
        Err(_) => return 1,
    };
    if params.dry_run {
        return 0;
    }
    let mut out = Vec::with_capacity(bytes.len() + 16);
    out.extend_from_slice(&LLAMA_FILE_MAGIC_GGLA.to_le_bytes());
    out.extend_from_slice(&(params.ftype as i32).to_le_bytes());
    out.extend_from_slice(&bytes);
    if fs::write(fname_out, out).is_err() {
        return 1;
    }
    0
}

pub fn llama_model_dflash_variant(model: &Model) -> DFlashVariant {
    model.dflash_variant
}

pub fn llama_model_dflash_selector_top_k(model: &Model) -> i32 {
    model.dflash_selector_top_k
}

pub fn llama_model_target_layer_ids(model: &Model) -> &[i32] {
    &model.target_layer_ids
}

pub fn llama_model_target_layer_ids_n(model: &Model) -> u32 {
    model.target_layer_ids.len() as u32
}

/// `llama_adapter_lora_init`.
pub fn llama_adapter_lora_init(model: &Model, path: &str) -> Result<AdapterLora> {
    let bytes = fs::read(path)
        .map_err(|e| Error::new(format!("failed to read LoRA {}: {}", path, e)))?;
    let mut adapter = AdapterLora {
        path: path.to_owned(),
        rank: 0,
        ..AdapterLora::default()
    };
    adapter.meta.insert("adapter.file_size".to_owned(), bytes.len().to_string());
    adapter.meta.insert("adapter.base_model".to_owned(), model.path.clone());
    Ok(adapter)
}

/// `llama_adapter_lora_init_from_file_ptr`.
pub fn llama_adapter_lora_init_from_file_ptr(
    model: &Model,
    file: &mut File,
) -> Result<AdapterLora> {
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| Error::new(format!("failed to read LoRA FILE: {}", e)))?;
    let mut adapter = AdapterLora::default();
    adapter.meta.insert("adapter.file_size".to_owned(), bytes.len().to_string());
    adapter.meta.insert("adapter.base_model".to_owned(), model.path.clone());
    Ok(adapter)
}

pub fn llama_adapter_meta_val_str(
    adapter: &AdapterLora,
    key: &str,
    buf: &mut [u8],
) -> i32 {
    match adapter.meta.get(key) {
        Some(value) => write_c_str(buf, value),
        None => {
            clear_buf(buf);
            -1
        }
    }
}

pub fn llama_adapter_meta_count(adapter: &AdapterLora) -> i32 {
    adapter.meta.len() as i32
}

pub fn llama_adapter_meta_key_by_index(
    adapter: &AdapterLora,
    i: i32,
    buf: &mut [u8],
) -> i32 {
    let mut keys: Vec<&String> = adapter.meta.keys().collect();
    keys.sort();
    match usize::try_from(i).ok().and_then(|i| keys.get(i)) {
        Some(key) => write_c_str(buf, key),
        None => {
            clear_buf(buf);
            -1
        }
    }
}

pub fn llama_adapter_meta_val_str_by_index(
    adapter: &AdapterLora,
    i: i32,
    buf: &mut [u8],
) -> i32 {
    let mut keys: Vec<&String> = adapter.meta.keys().collect();
    keys.sort();
    match usize::try_from(i).ok().and_then(|i| keys.get(i)).and_then(|k| adapter.meta.get(k.as_str())) {
        Some(value) => write_c_str(buf, value),
        None => {
            clear_buf(buf);
            -1
        }
    }
}

pub fn llama_adapter_lora_free(_adapter: AdapterLora) {}

pub fn llama_adapter_get_alora_n_invocation_tokens(adapter: &AdapterLora) -> u64 {
    adapter.invocation_tokens.len() as u64
}

pub fn llama_adapter_get_alora_invocation_tokens(adapter: &AdapterLora) -> &[Token] {
    &adapter.invocation_tokens
}

pub fn llama_set_adapters_lora(
    ctx: &mut Context,
    adapters: &[AdapterLora],
    scales: &[f32],
) -> i32 {
    if adapters.len() != scales.len() {
        return -1;
    }
    let same = ctx.model.adapters.len() == adapters.len()
        && ctx
            .model
            .adapters
            .iter()
            .zip(adapters)
            .all(|(a, b)| a.path == b.path);
    if !same {
        ctx.model.adapters = adapters.to_vec();
    }
    0
}

pub fn llama_set_adapter_cvec(
    ctx: &mut Context,
    data: &[f32],
    n_embd: i32,
    il_start: i32,
    il_end: i32,
) -> i32 {
    if data.is_empty() {
        ctx.cvec = None;
        return 0;
    }
    if n_embd <= 0 || il_end < il_start {
        return -1;
    }
    let n_layers = (il_end - il_start + 1) as usize;
    if data.len() != n_embd as usize * n_layers {
        return -1;
    }
    ctx.cvec = Some(data.to_vec());
    0
}

fn normalize_range(p0: LlamaPos, p1: LlamaPos, max_pos: LlamaPos) -> (LlamaPos, LlamaPos) {
    let start = if p0 < 0 { 0 } else { p0 };
    let end = if p1 < 0 { max_pos.saturating_add(1) } else { p1 };
    (start, end)
}

fn token_matches(token: &MemoryToken, seq_id: LlamaSeqId, p0: LlamaPos, p1: LlamaPos) -> bool {
    let pos_ok = token.pos >= p0 && token.pos < p1;
    let seq_ok = seq_id < 0 || token.seq_ids.iter().any(|&s| s == seq_id);
    pos_ok && seq_ok
}

pub fn llama_memory_clear(mem: &mut Memory, data: bool) {
    mem.tokens.clear();
    if data {
        mem.swa = 0;
    }
}

pub fn llama_memory_seq_rm(
    mem: &mut Memory,
    seq_id: LlamaSeqId,
    p0: LlamaPos,
    p1: LlamaPos,
) -> bool {
    let max = mem.tokens.iter().map(|t| t.pos).max().unwrap_or(-1);
    let (start, end) = normalize_range(p0, p1, max);
    mem.tokens.retain(|t| !token_matches(t, seq_id, start, end));
    true
}

pub fn llama_memory_seq_cp(
    mem: &mut Memory,
    seq_id_src: LlamaSeqId,
    seq_id_dst: LlamaSeqId,
    p0: LlamaPos,
    p1: LlamaPos,
) {
    let max = mem.tokens.iter().map(|t| t.pos).max().unwrap_or(-1);
    let (start, end) = normalize_range(p0, p1, max);
    let mut copies = Vec::new();
    for token in &mem.tokens {
        if seq_id_src < 0 || token.seq_ids.iter().any(|&s| s == seq_id_src) {
            if token.pos >= start && token.pos < end {
                let mut copy = token.clone();
                copy.seq_ids = vec![seq_id_dst];
                copies.push(copy);
            }
        }
    }
    mem.tokens.extend(copies);
}

pub fn llama_memory_seq_keep(mem: &mut Memory, seq_id: LlamaSeqId) {
    for token in &mut mem.tokens {
        token.seq_ids.retain(|&s| s == seq_id);
    }
    mem.tokens.retain(|t| !t.seq_ids.is_empty());
}

pub fn llama_memory_seq_add(
    mem: &mut Memory,
    seq_id: LlamaSeqId,
    p0: LlamaPos,
    p1: LlamaPos,
    delta: LlamaPos,
) {
    let max = mem.tokens.iter().map(|t| t.pos).max().unwrap_or(-1);
    let (start, end) = normalize_range(p0, p1, max);
    for token in &mut mem.tokens {
        if token_matches(token, seq_id, start, end) {
            token.pos = token.pos.saturating_add(delta);
        }
    }
}

pub fn llama_memory_seq_div(
    mem: &mut Memory,
    seq_id: LlamaSeqId,
    p0: LlamaPos,
    p1: LlamaPos,
    d: i32,
) {
    if d <= 1 {
        return;
    }
    let max = mem.tokens.iter().map(|t| t.pos).max().unwrap_or(-1);
    let (start, end) = normalize_range(p0, p1, max);
    for token in &mut mem.tokens {
        if token_matches(token, seq_id, start, end) {
            token.pos /= d;
        }
    }
}

pub fn llama_memory_seq_pos_min(mem: &Memory, seq_id: LlamaSeqId) -> LlamaPos {
    mem.tokens
        .iter()
        .filter(|t| seq_id < 0 || t.seq_ids.iter().any(|&s| s == seq_id))
        .map(|t| t.pos)
        .min()
        .unwrap_or(-1)
}

pub fn llama_memory_seq_pos_max(mem: &Memory, seq_id: LlamaSeqId) -> LlamaPos {
    mem.tokens
        .iter()
        .filter(|t| seq_id < 0 || t.seq_ids.iter().any(|&s| s == seq_id))
        .map(|t| t.pos)
        .max()
        .unwrap_or(-1)
}

pub fn llama_memory_can_shift(_mem: &Memory) -> bool {
    true
}

fn serialize_memory(mem: &Memory, seq_filter: Option<LlamaSeqId>, partial_only: bool) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, LLAMA_STATE_SEQ_MAGIC);
    push_u32(&mut out, LLAMA_STATE_SEQ_VERSION);
    let max_pos = mem.tokens.iter().map(|t| t.pos).max().unwrap_or(0);
    let min_pos = if partial_only && mem.swa > 0 {
        max_pos.saturating_sub(mem.swa)
    } else {
        i32::MIN
    };
    let selected: Vec<&MemoryToken> = mem
        .tokens
        .iter()
        .filter(|t| seq_filter.map_or(true, |s| t.seq_ids.iter().any(|&x| x == s)))
        .filter(|t| t.pos >= min_pos)
        .collect();
    push_u64(&mut out, selected.len() as u64);
    for token in selected {
        push_i32(&mut out, token.id);
        push_i32(&mut out, token.pos);
        push_u32(&mut out, token.seq_ids.len() as u32);
        for &seq in &token.seq_ids {
            push_i32(&mut out, seq);
        }
    }
    out
}

fn deserialize_memory(data: &[u8]) -> Option<(Memory, usize)> {
    let mut pos = 0usize;
    if pop_u32(data, &mut pos)? != LLAMA_STATE_SEQ_MAGIC {
        return None;
    }
    let version = pop_u32(data, &mut pos)?;
    if version != LLAMA_STATE_SEQ_VERSION {
        return None;
    }
    let n = pop_u64(data, &mut pos)? as usize;
    let mut mem = Memory::default();
    for _ in 0..n {
        let id = pop_i32(data, &mut pos)?;
        let p = pop_i32(data, &mut pos)?;
        let n_seq = pop_u32(data, &mut pos)? as usize;
        let mut seq_ids = Vec::with_capacity(n_seq);
        for _ in 0..n_seq {
            seq_ids.push(pop_i32(data, &mut pos)?);
        }
        mem.tokens.push(MemoryToken { id, pos: p, seq_ids });
    }
    Some((mem, pos))
}

fn state_tokens(mem: &Memory) -> Vec<Token> {
    mem.tokens.iter().map(|t| t.id).collect()
}

fn serialize_state(mem: &Memory) -> Vec<u8> {
    let tokens = state_tokens(mem);
    let kv = serialize_memory(mem, None, false);
    let mut out = Vec::new();
    push_u32(&mut out, LLAMA_SESSION_MAGIC);
    push_u32(&mut out, LLAMA_SESSION_VERSION);
    push_u64(&mut out, tokens.len() as u64);
    for token in tokens {
        push_i32(&mut out, token);
    }
    push_u64(&mut out, kv.len() as u64);
    out.extend_from_slice(&kv);
    out
}

fn deserialize_state(data: &[u8]) -> Option<(Vec<Token>, Memory, usize)> {
    let mut pos = 0usize;
    if pop_u32(data, &mut pos)? != LLAMA_SESSION_MAGIC {
        return None;
    }
    if pop_u32(data, &mut pos)? != LLAMA_SESSION_VERSION {
        return None;
    }
    let n_tokens = pop_u64(data, &mut pos)? as usize;
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(pop_i32(data, &mut pos)?);
    }
    let n_state = pop_u64(data, &mut pos)? as usize;
    let kv = take_bytes(data, &mut pos, n_state)?;
    let (mem, _) = deserialize_memory(kv)?;
    Some((tokens, mem, pos))
}

pub fn llama_state_get_size(ctx: &Context) -> usize {
    serialize_state(&ctx.memory).len()
}

pub fn llama_state_get_data(ctx: &Context, dst: &mut [u8]) -> usize {
    let state = serialize_state(&ctx.memory);
    let n = state.len().min(dst.len());
    dst[..n].copy_from_slice(&state[..n]);
    n
}

pub fn llama_state_set_data(ctx: &mut Context, src: &[u8]) -> usize {
    match deserialize_state(src) {
        Some((_tokens, mem, used)) => {
            ctx.memory = mem;
            used
        }
        None => 0,
    }
}

pub fn llama_state_load_file(
    ctx: &mut Context,
    path: &str,
    tokens_out: Option<&mut Vec<Token>>,
    n_token_capacity: usize,
) -> Result<bool> {
    let data = fs::read(path)
        .map_err(|e| Error::new(format!("failed to read session {}: {}", path, e)))?;
    let (tokens, mem, _) = deserialize_state(&data)
        .ok_or_else(|| Error::new("invalid session file"))?;
    ctx.memory = mem;
    if let Some(out) = tokens_out {
        out.clear();
        out.extend_from_slice(&tokens[..tokens.len().min(n_token_capacity)]);
    }
    Ok(true)
}

pub fn llama_state_save_file(
    ctx: &Context,
    path: &str,
    tokens: &[Token],
    n_token_count: usize,
) -> Result<()> {
    let mut state = serialize_state(&ctx.memory);
    let n = n_token_count.min(tokens.len());
    let mut out = Vec::new();
    push_u32(&mut out, LLAMA_SESSION_MAGIC);
    push_u32(&mut out, LLAMA_SESSION_VERSION);
    push_u64(&mut out, n as u64);
    for &token in &tokens[..n] {
        push_i32(&mut out, token);
    }
    push_u64(&mut out, state.len() as u64);
    out.append(&mut state);
    fs::write(path, out).map_err(|e| Error::new(format!("failed to write session {}: {}", path, e)))
}

pub fn llama_state_seq_get_size(ctx: &Context, seq_id: LlamaSeqId) -> usize {
    serialize_memory(&ctx.memory, Some(seq_id), false).len()
}

pub fn llama_state_seq_get_data(
    ctx: &Context,
    dst: &mut [u8],
    seq_id: LlamaSeqId,
) -> usize {
    let data = serialize_memory(&ctx.memory, Some(seq_id), false);
    let n = data.len().min(dst.len());
    dst[..n].copy_from_slice(&data[..n]);
    n
}

pub fn llama_state_seq_set_data(
    ctx: &mut Context,
    src: &[u8],
    dest_seq_id: LlamaSeqId,
) -> usize {
    match deserialize_memory(src) {
        Some((mem, used)) => {
            for mut token in mem.tokens {
                token.seq_ids = vec![dest_seq_id];
                ctx.memory.tokens.push(token);
            }
            used
        }
        None => 0,
    }
}

pub fn llama_state_seq_save_file(
    ctx: &Context,
    path: &str,
    seq_id: LlamaSeqId,
    tokens: &[Token],
    n_token_count: usize,
) -> Result<()> {
    let state = serialize_memory(&ctx.memory, Some(seq_id), false);
    let n = n_token_count.min(tokens.len());
    let mut out = Vec::new();
    push_u32(&mut out, LLAMA_STATE_SEQ_MAGIC);
    push_u32(&mut out, LLAMA_STATE_SEQ_VERSION);
    push_i32(&mut out, seq_id);
    push_u64(&mut out, n as u64);
    for &token in &tokens[..n] {
        push_i32(&mut out, token);
    }
    push_u64(&mut out, state.len() as u64);
    out.extend_from_slice(&state);
    fs::write(path, out)
        .map_err(|e| Error::new(format!("failed to write sequence state {}: {}", path, e)))
}

pub fn llama_state_seq_load_file(
    ctx: &mut Context,
    path: &str,
    dest_seq_id: LlamaSeqId,
    tokens_out: Option<&mut Vec<Token>>,
    n_token_capacity: usize,
) -> Result<usize> {
    let data = fs::read(path)
        .map_err(|e| Error::new(format!("failed to read sequence state {}: {}", path, e)))?;
    let mut pos = 0usize;
    if pop_u32(&data, &mut pos) != Some(LLAMA_STATE_SEQ_MAGIC)
        || pop_u32(&data, &mut pos) != Some(LLAMA_STATE_SEQ_VERSION)
    {
        return Err(Error::new("invalid sequence state file"));
    }
    let _saved_seq = pop_i32(&data, &mut pos).unwrap_or(-1);
    let n_tokens = pop_u64(&data, &mut pos).unwrap_or(0) as usize;
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(pop_i32(&data, &mut pos).unwrap_or(LLAMA_TOKEN_NULL));
    }
    let n_state = pop_u64(&data, &mut pos).unwrap_or(0) as usize;
    let mut kv = take_bytes(&data, &mut pos, n_state)
        .ok_or_else(|| Error::new("truncated sequence state"))?
        .to_vec();
    let (mut mem, _) = deserialize_memory(&mut kv)
        .ok_or_else(|| Error::new("invalid sequence state payload"))?;
    for token in &mut mem.tokens {
        token.seq_ids = vec![dest_seq_id];
    }
    ctx.memory.tokens.extend(mem.tokens);
    if let Some(out) = tokens_out {
        out.clear();
        out.extend_from_slice(&tokens[..tokens.len().min(n_token_capacity)]);
    }
    Ok(tokens.len())
}

pub fn llama_state_seq_get_size_ext(
    ctx: &Context,
    seq_id: LlamaSeqId,
    flags: StateSeqFlags,
) -> usize {
    if flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 {
        return 0;
    }
    let partial = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY != 0;
    serialize_memory(&ctx.memory, Some(seq_id), partial).len()
}

pub fn llama_state_seq_get_data_ext(
    ctx: &Context,
    dst: &mut [u8],
    seq_id: LlamaSeqId,
    flags: StateSeqFlags,
) -> usize {
    if flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 {
        return 0;
    }
    let partial = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY != 0;
    let data = serialize_memory(&ctx.memory, Some(seq_id), partial);
    let n = data.len().min(dst.len());
    dst[..n].copy_from_slice(&data[..n]);
    n
}

pub fn llama_state_seq_set_data_ext(
    ctx: &mut Context,
    src: &[u8],
    dest_seq_id: LlamaSeqId,
    flags: StateSeqFlags,
) -> usize {
    if flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 {
        return 0;
    }
    llama_state_seq_set_data(ctx, src, dest_seq_id)
}

/// `llama_batch_get_one`. The slice length replaces `n_tokens`.
pub fn llama_batch_get_one(tokens: &[Token]) -> Batch {
    let n = tokens.len() as i32;
    Batch {
        n_tokens: n,
        token: tokens.to_vec(),
        embd: Vec::new(),
        pos: Vec::new(),
        n_seq_id: Vec::new(),
        seq_id: Vec::new(),
        logits: Vec::new(),
    }
}

/// `llama_batch_init`. All elements start zeroed in the Rust stand-in.
pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> Batch {
    let n = n_tokens.max(0) as usize;
    let n_seq = n_seq_max.max(0) as usize;
    let mut batch = Batch {
        n_tokens: 0,
        token: Vec::new(),
        embd: Vec::new(),
        pos: vec![0; n],
        n_seq_id: vec![0; n],
        seq_id: vec![vec![0; n_seq]; n],
        logits: vec![0; n],
    };
    if embd != 0 {
        batch.embd = vec![0.0; n * embd.max(0) as usize];
    } else {
        batch.token = vec![LLAMA_TOKEN_NULL; n];
    }
    batch
}

/// `llama_batch_free`. Rust drops the batch when this function consumes it.
pub fn llama_batch_free(_batch: Batch) {}

fn decode_impl(ctx: &mut Context, batch: &Batch, is_encode: bool) -> i32 {
    if batch.n_tokens < 0 || batch.n_tokens as usize > batch.token.len() {
        return -1;
    }
    let n = batch.n_tokens as usize;
    if n == 0 {
        return 0;
    }
    if !batch.embd.is_empty() && !batch.token.is_empty() {
        // Both inputs are allowed by the extended API, but not by llama_batch.
        if batch.embd.len() % n != 0 {
            return -1;
        }
    }
    if let Some(cb) = ctx.abort_callback {
        if cb() {
            return 2;
        }
    }
    let start = Instant::now();
    let n_vocab = ctx.model.vocab.pieces.len();
    let n_embd = ctx.model.n_embd.max(0) as usize;
    let mut outputs = Vec::new();
    let mut seqs = Vec::new();
    for i in 0..n {
        let seq_ids: Vec<LlamaSeqId> = if batch.n_seq_id.get(i).copied().unwrap_or(0) > 0
            && i < batch.seq_id.len()
        {
            let count = batch.n_seq_id[i].max(0) as usize;
            batch.seq_id[i].iter().take(count).copied().collect()
        } else {
            vec![0]
        };
        let pos = match batch.pos.get(i).copied() {
            Some(p) => p,
            None => {
                let s = seq_ids.first().copied().unwrap_or(0);
                llama_memory_seq_pos_max(&ctx.memory, s).saturating_add(1).max(0)
            }
        };
        if !is_encode {
            let id = batch.token.get(i).copied().unwrap_or(LLAMA_TOKEN_NULL);
            ctx.memory.tokens.push(MemoryToken {
                id,
                pos,
                seq_ids: seq_ids.clone(),
            });
        }
        if batch.logits.get(i).copied().unwrap_or(0) != 0 {
            outputs.push(i);
        }
        for s in seq_ids {
            if !seqs.contains(&s) {
                seqs.push(s);
            }
        }
    }
    if outputs.is_empty() {
        outputs.push(n - 1);
    }
    ctx.logits = vec![0.0; outputs.len() * n_vocab];
    if ctx.embeddings_enabled || ctx.params.pooling_type != PoolingType::None {
        let rows = if ctx.params.pooling_type == PoolingType::None {
            outputs.len()
        } else {
            seqs.len().max(1)
        };
        ctx.embeddings = vec![0.0; rows * n_embd];
        ctx.enc_output = ctx.embeddings.clone();
    } else {
        ctx.embeddings.clear();
        ctx.enc_output.clear();
    }
    if let Some(cb) = ctx.params.cb_eval {
        let tensor = Tensor::default();
        let _ = cb(&tensor, true);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    if is_encode {
        ctx.perf.t_p_eval_ms += elapsed;
        ctx.perf.n_p_eval += n as i32;
    } else {
        ctx.perf.t_eval_ms += elapsed;
        ctx.perf.n_eval += n as i32;
    }
    0
}

pub fn llama_encode(ctx: &mut Context, batch: &Batch) -> i32 {
    decode_impl(ctx, batch, true)
}

pub fn llama_decode(ctx: &mut Context, batch: &Batch) -> i32 {
    decode_impl(ctx, batch, false)
}

/// `llama_batch_ext_init`.
pub fn llama_batch_ext_init(ctx: &Context) -> BatchExt {
    BatchExt {
        entries: Vec::new(),
        capacity: ctx.n_batch as usize,
        embd_size: ctx.model.n_embd.max(0) as usize,
        n_seq_max: ctx.n_seq_max as usize,
        rows: Vec::new(),
    }
}

/// `llama_batch_ext_free`. Rust drops the batch when this function consumes it.
pub fn llama_batch_ext_free(_batch: BatchExt) {}

pub fn llama_batch_ext_clear(batch: &mut BatchExt) {
    batch.entries.clear();
}

fn batch_ext_add_entry(batch: &mut BatchExt, seq_id: LlamaSeqId) -> Result<i32> {
    if batch.entries.len() >= batch.capacity {
        return Ok(-1);
    }
    if seq_id < -1 {
        return Ok(-3);
    }
    let idx = batch.entries.len() as i32;
    batch.entries.push(BatchEntry {
        id: LLAMA_TOKEN_NULL,
        embd: Vec::new(),
        embd_state: Vec::new(),
        embd_row: None,
        embd_state_row: None,
        pos: Vec::new(),
        seq_ids: vec![seq_id],
        output_embd: false,
        output_logits: false,
    });
    Ok(idx)
}

pub fn llama_batch_ext_add(batch: &mut BatchExt, seq_id: LlamaSeqId) -> i32 {
    batch_ext_add_entry(batch, seq_id).unwrap_or(-1)
}

pub fn llama_batch_ext_add_token(
    batch: &mut BatchExt,
    seq_id: LlamaSeqId,
    id: Token,
    n_vocab: i32,
) -> i32 {
    if id == LLAMA_TOKEN_NULL || id < 0 || id >= n_vocab {
        return -2;
    }
    match batch_ext_add_entry(batch, seq_id) {
        Ok(idx @ 0..) => {
            batch.entries[idx as usize].id = id;
            idx
        }
        Ok(err) => err,
        Err(_) => -1,
    }
}

pub fn llama_batch_ext_add_embd(
    batch: &mut BatchExt,
    seq_id: LlamaSeqId,
    embd: &Embedding,
) -> i32 {
    if embd.n_rows == 0
        || embd.n_embd == 0
        || embd.data.len() != embd.n_rows * embd.n_embd
        || (batch.embd_size != 0 && embd.n_embd != batch.embd_size)
    {
        return -2;
    }
    match batch_ext_add_entry(batch, seq_id) {
        Ok(idx @ 0..) => {
            batch.entries[idx as usize].embd = embd.data.clone();
            idx
        }
        Ok(err) => err,
        Err(_) => -1,
    }
}

pub fn llama_batch_ext_add_seq(
    batch: &mut BatchExt,
    idx: i32,
    seq_id: LlamaSeqId,
) -> bool {
    let entry = match usize::try_from(idx).ok().and_then(|i| batch.entries.get_mut(i)) {
        Some(e) => e,
        None => return false,
    };
    if seq_id < -1 || entry.seq_ids.contains(&seq_id) {
        return false;
    }
    entry.seq_ids.push(seq_id);
    true
}

pub fn llama_batch_ext_set_embd_token(
    batch: &mut BatchExt,
    idx: i32,
    embd: &Embedding,
) -> bool {
    let entry = match usize::try_from(idx).ok().and_then(|i| batch.entries.get_mut(i)) {
        Some(e) => e,
        None => return false,
    };
    if embd.n_rows == 0 || embd.data.len() != embd.n_rows * embd.n_embd {
        return false;
    }
    entry.embd = embd.data.clone();
    true
}

pub fn llama_batch_ext_set_embd_state(
    batch: &mut BatchExt,
    idx: i32,
    embd: &Embedding,
) -> bool {
    let entry = match usize::try_from(idx).ok().and_then(|i| batch.entries.get_mut(i)) {
        Some(e) => e,
        None => return false,
    };
    if embd.data.is_empty() {
        return false;
    }
    entry.embd_state = embd.data.clone();
    true
}

pub fn llama_batch_ext_add_embd_row(
    batch: &mut BatchExt,
    seq_id: LlamaSeqId,
    row: i32,
) -> i32 {
    match batch_ext_add_entry(batch, seq_id) {
        Ok(idx @ 0..) => {
            batch.entries[idx as usize].embd_row = Some(row);
            idx
        }
        Ok(err) => err,
        Err(_) => -1,
    }
}

pub fn llama_batch_ext_set_embd_state_row(
    batch: &mut BatchExt,
    idx: i32,
    row: i32,
) -> bool {
    match usize::try_from(idx).ok().and_then(|i| batch.entries.get_mut(i)) {
        Some(entry) => {
            entry.embd_state_row = Some(row);
            true
        }
        None => false,
    }
}

pub fn llama_batch_ext_set_output_embd(
    batch: &mut BatchExt,
    idx: i32,
    value: bool,
) -> bool {
    match usize::try_from(idx).ok().and_then(|i| batch.entries.get_mut(i)) {
        Some(entry) => {
            entry.output_embd = value;
            true
        }
        None => false,
    }
}

pub fn llama_batch_ext_set_output_logits(
    batch: &mut BatchExt,
    idx: i32,
    value: bool,
) -> bool {
    match usize::try_from(idx).ok().and_then(|i| batch.entries.get_mut(i)) {
        Some(entry) => {
            entry.output_logits = value;
            true
        }
        None => false,
    }
}

pub fn llama_batch_ext_set_pos(
    batch: &mut BatchExt,
    idx: i32,
    pos: &[LlamaPos],
) -> bool {
    match usize::try_from(idx).ok().and_then(|i| batch.entries.get_mut(i)) {
        Some(entry) => {
            entry.pos = pos.to_vec();
            true
        }
        None => false,
    }
}

fn batch_from_ext(batch: &BatchExt) -> Batch {
    let n = batch.entries.len();
    let mut out = Batch {
        n_tokens: n as i32,
        token: Vec::with_capacity(n),
        embd: Vec::new(),
        pos: Vec::with_capacity(n),
        n_seq_id: Vec::with_capacity(n),
        seq_id: Vec::with_capacity(n),
        logits: Vec::with_capacity(n),
    };
    for entry in &batch.entries {
        out.token.push(entry.id);
        out.pos.push(entry.pos.first().copied().unwrap_or(0));
        out.n_seq_id.push(entry.seq_ids.len() as i32);
        out.seq_id.push(entry.seq_ids.clone());
        out.logits.push(if entry.output_logits || entry.output_embd { 1 } else { 0 });
        if !entry.embd.is_empty() {
            out.embd.extend_from_slice(&entry.embd);
        }
    }
    out
}

/// `llama_process`. Return values match `llama_decode`.
pub fn llama_process(ctx: &mut Context, kind: ProcessType, batch: &BatchExt) -> i32 {
    let converted = batch_from_ext(batch);
    decode_impl(ctx, &converted, kind == ProcessType::Encode)
}

pub fn llama_set_n_threads(ctx: &mut Context, n_threads: i32, n_threads_batch: i32) {
    ctx.n_threads = n_threads;
    ctx.n_threads_batch = n_threads_batch;
    ctx.params.n_threads = n_threads;
    ctx.params.n_threads_batch = n_threads_batch;
}

pub fn llama_n_threads(ctx: &Context) -> i32 {
    ctx.n_threads
}

pub fn llama_n_threads_batch(ctx: &Context) -> i32 {
    ctx.n_threads_batch
}

pub fn llama_set_embeddings(ctx: &mut Context, embeddings: bool) {
    ctx.embeddings_enabled = embeddings;
    ctx.params.embeddings = embeddings;
}

pub fn llama_set_causal_attn(ctx: &mut Context, causal_attn: bool) {
    ctx.causal_attn = causal_attn;
    ctx.params.attention_type = if causal_attn {
        AttentionType::Causal
    } else {
        AttentionType::NonCausal
    };
}

pub fn llama_get_causal_attn(ctx: &Context) -> bool {
    ctx.causal_attn
}

pub fn llama_set_abort_callback(ctx: &mut Context, callback: Option<AbortCallback>) {
    ctx.abort_callback = callback;
    ctx.params.abort_callback = callback;
}

/// `llama_synchronize`: all stand-in operations are synchronous.
pub fn llama_synchronize(_ctx: &Context) {}

fn ith_row<'a>(data: &'a [f32], row_size: usize, i: i32) -> Option<&'a [f32]> {
    if row_size == 0 {
        return None;
    }
    let n_rows = data.len() / row_size;
    let idx = if i < 0 { n_rows as i32 + i } else { i };
    if idx < 0 {
        return None;
    }
    let start = idx as usize * row_size;
    data.get(start..start + row_size)
}

pub fn llama_get_logits(ctx: &Context) -> &[f32] {
    &ctx.logits
}

pub fn llama_get_logits_ith(ctx: &Context, i: i32) -> Option<&[f32]> {
    ith_row(&ctx.logits, ctx.model.vocab.pieces.len(), i)
}

pub fn llama_get_embeddings(ctx: &Context) -> &[f32] {
    &ctx.embeddings
}

pub fn llama_get_embeddings_ith(ctx: &Context, i: i32) -> Option<&[f32]> {
    ith_row(&ctx.embeddings, ctx.model.n_embd.max(0) as usize, i)
}

pub fn llama_get_embeddings_seq(ctx: &Context, seq_id: LlamaSeqId) -> Option<&[f32]> {
    if ctx.params.pooling_type == PoolingType::None {
        return None;
    }
    let n = if ctx.params.pooling_type == PoolingType::Rank {
        ctx.model.n_cls_out as usize
    } else {
        ctx.model.n_embd.max(0) as usize
    };
    let idx = seq_id.max(0) as usize;
    ctx.embeddings.get(idx * n..(idx + 1) * n)
}

pub fn llama_set_embeddings_nextn(ctx: &mut Context, value: bool, masked: bool) {
    ctx.embeddings_nextn_enabled = value;
    ctx.embeddings_nextn_masked = masked;
}

pub fn llama_set_nextn_layer_offset(ctx: &mut Context, offset: i32) {
    ctx.nextn_layer_offset = offset;
}

pub fn llama_get_embeddings_nextn(ctx: &Context) -> &[f32] {
    &ctx.embeddings_nextn
}

pub fn llama_get_embeddings_nextn_ith(ctx: &Context, i: i32) -> Option<&[f32]> {
    ith_row(&ctx.embeddings_nextn, ctx.model.n_embd_out.max(0) as usize, i)
}

pub fn llama_set_embeddings_layer_inp(ctx: &mut Context, lid: u32, value: bool) {
    let n = ctx.model.n_layer.max(0) as usize;
    if ctx.embeddings_layer_inp_enabled.len() < n {
        ctx.embeddings_layer_inp_enabled.resize(n, false);
    }
    if let Some(slot) = ctx.embeddings_layer_inp_enabled.get_mut(lid as usize) {
        *slot = value;
    }
}

pub fn llama_get_embeddings_layer_inp(ctx: &Context, lid: u32) -> Option<&[f32]> {
    let n = ctx.model.n_embd.max(0) as usize;
    let row = lid as usize;
    ctx.embeddings_layer_inp.get(row * n..(row + 1) * n)
}

pub fn llama_set_embeddings_layer_inp_dev(ctx: &mut Context, value: bool) {
    ctx.embeddings_layer_inp_dev = value;
}

pub fn llama_supports_embd_row(ctx: &Context) -> bool {
    ctx.ctx_other.is_some()
        && matches!(ctx.model.dflash_variant, DFlashVariant::Dflash | DFlashVariant::Dspark)
}

pub fn llama_set_embeddings_nextn_dev(ctx: &mut Context, value: bool) -> bool {
    ctx.embeddings_nextn_dev = value && ctx.model.n_layer_nextn > 0;
    ctx.embeddings_nextn_dev
}

pub fn llama_get_embeddings_nextn_rows(
    ctx: &Context,
    row0: i64,
    n_rows: i64,
    out: &mut [f32],
) -> bool {
    let width = ctx.model.n_embd_out.max(0) as usize;
    if width == 0 || row0 < 0 || n_rows < 0 || out.len() < n_rows as usize * width {
        return false;
    }
    let src = ctx.embeddings_nextn.get(row0 as usize * width..(row0 as usize + n_rows as usize) * width);
    match src {
        Some(s) => {
            out[..s.len()].copy_from_slice(s);
            true
        }
        None => false,
    }
}

pub fn llama_set_embeddings_nextn_rows(
    ctx: &mut Context,
    row0: i64,
    n_rows: i64,
    data: &[f32],
) -> bool {
    let width = ctx.model.n_embd_out.max(0) as usize;
    if width == 0 || row0 < 0 || n_rows < 0 || data.len() < n_rows as usize * width {
        return false;
    }
    let needed = row0 as usize * width + n_rows as usize * width;
    if ctx.embeddings_nextn.len() < needed {
        ctx.embeddings_nextn.resize(needed, 0.0);
    }
    let dst = &mut ctx.embeddings_nextn[row0 as usize * width..needed];
    dst.copy_from_slice(&data[..n_rows as usize * width]);
    true
}

pub fn llama_set_ctx_state_src(
    ctx: &mut Context,
    ctx_src: Option<Rc<RefCell<Context>>>,
) -> bool {
    ctx.ctx_state_src = ctx_src;
    ctx.ctx_state_src.is_some()
}

pub fn llama_get_ctx_other(ctx: &Context) -> Option<Rc<RefCell<Context>>> {
    ctx.ctx_other.clone()
}

pub fn llama_get_sampled_token_ith(ctx: &Context, i: i32) -> Token {
    let n = ctx.sampled.tokens.len();
    let idx = if i < 0 { n as i32 + i } else { i };
    if idx < 0 || idx as usize >= n {
        LLAMA_TOKEN_NULL
    } else {
        ctx.sampled.tokens[idx as usize]
    }
}

pub fn llama_get_sampled_probs_ith(ctx: &Context, i: i32) -> Option<&[f32]> {
    let n = ctx.sampled.counts.len();
    let idx = if i < 0 { n as i32 + i } else { i };
    if idx < 0 || idx as usize >= n {
        return None;
    }
    let count = ctx.sampled.counts[idx as usize];
    let start: usize = ctx.sampled.counts[..idx as usize].iter().sum();
    ctx.sampled.probs.get(start..start + count)
}

pub fn llama_get_sampled_probs_count_ith(ctx: &Context, i: i32) -> u32 {
    let n = ctx.sampled.counts.len();
    let idx = if i < 0 { n as i32 + i } else { i };
    if idx < 0 || idx as usize >= n { 0 } else { ctx.sampled.counts[idx as usize] as u32 }
}

pub fn llama_get_sampled_logits_ith(ctx: &Context, i: i32) -> Option<&[f32]> {
    let n = ctx.sampled.counts.len();
    let idx = if i < 0 { n as i32 + i } else { i };
    if idx < 0 || idx as usize >= n {
        return None;
    }
    let count = ctx.sampled.counts[idx as usize];
    let start: usize = ctx.sampled.counts[..idx as usize].iter().sum();
    ctx.sampled.logits.get(start..start + count)
}

pub fn llama_get_sampled_logits_count_ith(ctx: &Context, i: i32) -> u32 {
    llama_get_sampled_probs_count_ith(ctx, i)
}

pub fn llama_get_sampled_candidates_ith(ctx: &Context, i: i32) -> Option<&[Token]> {
    let n = ctx.sampled.counts.len();
    let idx = if i < 0 { n as i32 + i } else { i };
    if idx < 0 || idx as usize >= n {
        return None;
    }
    let count = ctx.sampled.counts[idx as usize];
    let start: usize = ctx.sampled.counts[..idx as usize].iter().sum();
    ctx.sampled.candidates.get(start..start + count)
}

pub fn llama_get_sampled_candidates_count_ith(ctx: &Context, i: i32) -> u32 {
    llama_get_sampled_probs_count_ith(ctx, i)
}

pub fn llama_vocab_get_text(vocab: &Vocab, token: Token) -> Option<&str> {
    vocab.pieces.get(token as usize).map(String::as_str)
}

pub fn llama_vocab_get_score(vocab: &Vocab, token: Token) -> f32 {
    vocab.scores.get(token as usize).copied().unwrap_or(0.0)
}

pub fn llama_vocab_get_attr(vocab: &Vocab, token: Token) -> TokenAttr {
    vocab.attrs.get(token as usize).copied().unwrap_or_default()
}

pub fn llama_vocab_is_eog(vocab: &Vocab, token: Token) -> bool {
    token == vocab.special.eos || token == vocab.special.eot
}

pub fn llama_vocab_is_control(vocab: &Vocab, token: Token) -> bool {
    llama_vocab_get_attr(vocab, token).contains(TokenAttrFlag::Control)
}

pub fn llama_vocab_bos(vocab: &Vocab) -> Token {
    vocab.special.bos
}

pub fn llama_vocab_eos(vocab: &Vocab) -> Token {
    vocab.special.eos
}

pub fn llama_vocab_eot(vocab: &Vocab) -> Token {
    vocab.special.eot
}

pub fn llama_vocab_sep(vocab: &Vocab) -> Token {
    vocab.special.sep
}

pub fn llama_vocab_nl(vocab: &Vocab) -> Token {
    vocab.special.nl
}

pub fn llama_vocab_pad(vocab: &Vocab) -> Token {
    vocab.special.pad
}

pub fn llama_vocab_mask(vocab: &Vocab) -> Token {
    vocab.special.mask
}

pub fn llama_vocab_get_add_bos(vocab: &Vocab) -> bool {
    vocab.add_bos
}

pub fn llama_vocab_get_add_eos(vocab: &Vocab) -> bool {
    vocab.add_eos
}

pub fn llama_vocab_get_add_sep(vocab: &Vocab) -> bool {
    vocab.add_sep
}

pub fn llama_vocab_get_suppress_tokens(vocab: &Vocab) -> &[Token] {
    &vocab.suppress
}

pub fn llama_vocab_fim_pre(vocab: &Vocab) -> Token {
    vocab.special.fim_pre
}

pub fn llama_vocab_fim_suf(vocab: &Vocab) -> Token {
    vocab.special.fim_suf
}

pub fn llama_vocab_fim_mid(vocab: &Vocab) -> Token {
    vocab.special.fim_mid
}

pub fn llama_vocab_fim_pad(vocab: &Vocab) -> Token {
    vocab.special.fim_pad
}

pub fn llama_vocab_fim_rep(vocab: &Vocab) -> Token {
    vocab.special.fim_rep
}

pub fn llama_vocab_fim_sep(vocab: &Vocab) -> Token {
    vocab.special.fim_sep
}

/// `llama_tokenize`. Returns the token ids; use `Result` for the C error codes.
pub fn llama_tokenize(
    vocab: &Vocab,
    text: &[u8],
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<Token>> {
    use crate::sampler::Vocab as _;
    let mut tokens = vocab.tokenize(text, add_special && vocab.add_bos, parse_special);
    if add_special && vocab.add_eos {
        tokens.push(vocab.special.eos);
    }
    Ok(tokens)
}

pub fn llama_token_to_piece(
    vocab: &Vocab,
    token: Token,
    buf: &mut [u8],
    lstrip: i32,
    special: bool,
) -> i32 {
    let piece = match vocab.pieces.get(token as usize) {
        Some(p) => p.as_bytes(),
        None => return -1,
    };
    let mut skip = lstrip.max(0) as usize;
    let leading = piece.iter().take_while(|&&b| b == b' ').count();
    skip = skip.min(leading);
    let payload = &piece[skip..];
    let n = payload.len().min(buf.len());
    buf[..n].copy_from_slice(&payload[..n]);
    let _ = special;
    payload.len() as i32
}

pub fn llama_detokenize(
    vocab: &Vocab,
    tokens: &[Token],
    text: &mut [u8],
    remove_special: bool,
    unparse_special: bool,
) -> i32 {
    let mut out = Vec::new();
    for &token in tokens {
        if remove_special
            && (llama_vocab_is_eog(vocab, token) || (vocab.add_bos && token == vocab.special.bos))
        {
            continue;
        }
        if let Some(piece) = vocab.pieces.get(token as usize) {
            out.extend_from_slice(piece.as_bytes());
        }
    }
    let n = out.len().min(text.len());
    text[..n].copy_from_slice(&out[..n]);
    let _ = unparse_special;
    out.len() as i32
}

const BUILTIN_TEMPLATES: &[&str] = &[
    "chatml", "llama2", "mistral", "gemma", "vicuna", "phind", "deepseek", "zephyr",
    "qwen", "falcon", "stablelm", "orca",
];

pub fn llama_chat_builtin_templates(output: &mut Vec<String>, len: usize) -> i32 {
    output.clear();
    let n = len.min(BUILTIN_TEMPLATES.len());
    for name in BUILTIN_TEMPLATES.iter().take(n) {
        output.push((*name).to_owned());
    }
    BUILTIN_TEMPLATES.len() as i32
}

fn render_message(name: &str, role: &str, content: &str) -> String {
    match name {
        "chatml" | "qwen" | "deepseek" => {
            format!("<|im_start|>{}\n{}<|im_end|>\n", role, content)
        }
        "llama2" | "mistral" | "zephyr" => {
            if role == "system" {
                format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n[/INST]", content)
            } else {
                format!("[INST] {} [/INST]", content)
            }
        }
        "gemma" => format!("<start_of_turn>{}\n{}\n<end_of_turn>\n", role, content),
        "vicuna" => format!("{}: {}\n", if role == "user" { "USER" } else { "ASSISTANT" }, content),
        "phind" => format!("### {}:\n{}\n", role, content),
        "falcon" => format!("{}: {}\n", role, content),
        "stablelm" => format!("<|{}-token|>{}{}", role, content, "\n"),
        "orca" => format!("<|im_start|>{}\n{}<|im_end|>\n", role, content),
        _ => format!("{}: {}\n", role, content),
    }
}

/// `llama_chat_apply_template`. Supports the built-in templates of the port.
pub fn llama_chat_apply_template(
    tmpl: &str,
    chat: &[ChatMessage],
    add_ass: bool,
    buf: &mut [u8],
) -> i32 {
    let name = if tmpl.is_empty() { "chatml" } else { tmpl };
    if name != "chatml" && !BUILTIN_TEMPLATES.contains(&name) {
        clear_buf(buf);
        return -1;
    }
    let mut rendered = String::new();
    for message in chat {
        rendered.push_str(&render_message(name, &message.role, &message.content));
    }
    if add_ass {
        rendered.push_str(&render_message(name, "assistant", ""));
    }
    write_c_str(buf, &rendered)
}

pub fn llama_sampler_init(iface: Box<dyn SamplerI>, ctx: SamplerContext) -> Sampler {
    Sampler::new(iface, ctx)
}

pub fn llama_set_sampler(ctx: &mut Context, seq_id: LlamaSeqId, smpl: Sampler) -> bool {
    if seq_id < -1 {
        return false;
    }
    ctx.samplers.push(SamplerSeqConfig { seq_id, sampler: smpl });
    true
}

pub fn llama_sampler_name(smpl: &Sampler) -> Option<&str> {
    smpl.name()
}

pub fn llama_sampler_accept(smpl: &mut Sampler, token: Token) {
    smpl.iface.accept(&mut smpl.ctx, token);
}

pub fn llama_sampler_apply(smpl: &mut Sampler, cur: &mut TokenDataArray) {
    smpl.apply(cur);
}

pub fn llama_sampler_reset(smpl: &mut Sampler) {
    smpl.iface.reset(&mut smpl.ctx);
}

pub fn llama_sampler_clone(smpl: &Sampler) -> Sampler {
    smpl.clone()
}

pub fn llama_sampler_copy(src: &Sampler, dst: &mut Sampler) {
    src.iface.copy_state(&src.ctx, &mut dst.ctx);
}

pub fn llama_sampler_free(_smpl: Sampler) {}

pub fn llama_sampler_chain_init(params: SamplerChainParams) -> Sampler {
    Sampler::new(Box::new(ChainIface), SamplerContext::Chain(ChainSampler::new(params)))
}

pub fn llama_sampler_chain_add(chain: &mut Sampler, smpl: Sampler) -> bool {
    match as_chain_mut(chain) {
        Some(c) => {
            c.samplers.push(smpl);
            true
        }
        None => false,
    }
}

pub fn llama_sampler_chain_get(chain: &mut Sampler, i: i32) -> Option<&mut Sampler> {
    if i == -1 {
        return Some(chain);
    }
    let chain = as_chain_mut(chain)?;
    usize::try_from(i).ok().and_then(|i| chain.samplers.get_mut(i))
}

pub fn llama_sampler_chain_n(chain: &Sampler) -> i32 {
    chain.ctx.as_chain().map_or(0, |c| c.len())
}

pub fn llama_sampler_chain_remove(chain: &mut Sampler, i: i32) -> Option<Sampler> {
    let chain = as_chain_mut(chain)?;
    let idx = usize::try_from(i).ok()?;
    if idx < chain.samplers.len() {
        Some(chain.samplers.remove(idx))
    } else {
        None
    }
}

pub fn llama_sampler_init_greedy() -> Sampler {
    builtin_sampler(crate::sampler::init_greedy())
}

pub fn llama_sampler_init_dist(seed: u32) -> Sampler {
    let seed = if seed == LLAMA_DEFAULT_SEED { crate::sampler::LLAMA_DEFAULT_SEED } else { seed };
    builtin_sampler(crate::sampler::init_dist(seed))
}

pub fn llama_sampler_init_top_k(k: i32) -> Sampler {
    builtin_sampler(crate::sampler::init_top_k(k))
}

pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> Sampler {
    builtin_sampler(crate::sampler::init_top_p(p, min_keep))
}

pub fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> Sampler {
    builtin_sampler(crate::sampler::init_min_p(p, min_keep))
}

pub fn llama_sampler_init_typical(p: f32, min_keep: usize) -> Sampler {
    builtin_sampler(crate::sampler::init_typical(p, min_keep))
}

pub fn llama_sampler_init_temp(t: f32) -> Sampler {
    builtin_sampler(crate::sampler::init_temp(t))
}

pub fn llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> Sampler {
    builtin_sampler(crate::sampler::init_temp_ext(t, delta, exponent))
}

pub fn llama_sampler_init_xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Sampler {
    builtin_sampler(crate::sampler::init_xtc(p, t, min_keep, seed))
}

pub fn llama_sampler_init_top_n_sigma(n: f32) -> Sampler {
    builtin_sampler(crate::sampler::init_top_n_sigma(n))
}

pub fn llama_sampler_init_mirostat(
    n_vocab: i32,
    seed: u32,
    tau: f32,
    eta: f32,
    m: i32,
) -> Sampler {
    builtin_sampler(crate::sampler::init_mirostat(n_vocab, seed, tau, eta, m))
}

pub fn llama_sampler_init_mirostat_v2(seed: u32, tau: f32, eta: f32) -> Sampler {
    builtin_sampler(crate::sampler::init_mirostat_v2(seed, tau, eta))
}

pub fn llama_sampler_init_grammar(
    grammar_str: &str,
    grammar_root: &str,
    grammar: Option<Box<dyn crate::sampler::Grammar>>,
) -> Sampler {
    builtin_sampler(crate::sampler::init_grammar(Some(grammar_str), Some(grammar_root), grammar))
}

/// Lazy grammar: filtering starts once an accepted piece contains a trigger.
pub fn llama_sampler_init_grammar_lazy_patterns(
    vocab: Rc<Vocab>,
    grammar_str: &str,
    grammar_root: &str,
    trigger_patterns: &[&str],
    trigger_tokens: &[Token],
    grammar: Option<Box<dyn crate::sampler::Grammar>>,
) -> Sampler {
    let sampler = LazyGrammar {
        vocab,
        patterns: trigger_patterns.iter().map(|s| (*s).to_owned()).collect(),
        trigger_tokens: trigger_tokens.to_vec(),
        active: false,
        text: String::new(),
        grammar: crate::sampler::init_grammar(Some(grammar_str), Some(grammar_root), grammar),
    };
    Sampler::new(Box::new(LazyGrammarIface), SamplerContext::Custom(Box::new(sampler)))
}

pub fn llama_sampler_init_penalties(
    n_vocab: i32,
    penalty_last_n: i32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
) -> Sampler {
    builtin_sampler(crate::sampler::init_penalties(
        n_vocab,
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present,
    ))
}

pub fn llama_sampler_init_dry(
    vocab: &Vocab,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n: i32,
    seq_breakers: &[&str],
) -> Sampler {
    builtin_sampler(crate::sampler::init_dry(
        vocab,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_penalty_last_n,
        seq_breakers,
    ))
}

pub fn llama_sampler_init_adaptive_p(target: f32, decay: f32, seed: u32) -> Sampler {
    builtin_sampler(crate::sampler::init_adaptive_p(target, decay, seed))
}

pub fn llama_sampler_init_logit_bias(n_vocab: i32, logit_bias: &[LogitBias]) -> Sampler {
    let biases: Vec<crate::sampler::LogitBias> = logit_bias
        .iter()
        .map(|b| crate::sampler::LogitBias { token: b.token, bias: b.bias })
        .collect();
    builtin_sampler(crate::sampler::init_logit_bias(n_vocab, &biases))
}

pub fn llama_sampler_init_infill(vocab: Rc<Vocab>) -> Sampler {
    builtin_sampler(crate::sampler::init_infill(vocab))
}

pub fn llama_sampler_get_seed(smpl: &Sampler) -> u32 {
    smpl.ctx
        .as_builtin()
        .map(crate::sampler::get_seed)
        .unwrap_or(LLAMA_DEFAULT_SEED)
}

/// `llama_sampler_sample`: sample and accept one token from the ith output.
pub fn llama_sampler_sample(smpl: &mut Sampler, ctx: &Context, idx: i32) -> Token {
    let row = match llama_get_logits_ith(ctx, idx) {
        Some(r) => r.to_vec(),
        None => return LLAMA_TOKEN_NULL,
    };
    let mut cur = TokenDataArray::new(
        row.into_iter()
            .enumerate()
            .map(|(id, logit)| TokenData { id: id as Token, logit, p: 0.0 })
            .collect(),
    );
    smpl.apply(&mut cur);
    if cur.selected < 0 || cur.selected as usize >= cur.data.len() {
        return LLAMA_TOKEN_NULL;
    }
    let token = cur.data[cur.selected as usize].id;
    smpl.iface.accept(&mut smpl.ctx, token);
    token
}

pub fn llama_sampler_backend_init(
    smpl: &mut Sampler,
    buft: &Buft,
    n_outputs_max_per_seq: u32,
) -> bool {
    smpl.iface.backend_init(&mut smpl.ctx, buft, n_outputs_max_per_seq)
}

pub fn llama_sampler_backend_accept(
    smpl: &mut Sampler,
    ggml_ctx: &mut GgmlContext,
    gf: &mut Graph,
    selected_token: &Tensor,
) {
    smpl.iface.backend_accept(&mut smpl.ctx, ggml_ctx, gf, selected_token);
}

pub fn llama_sampler_backend_apply(
    smpl: &mut Sampler,
    ggml_ctx: &mut GgmlContext,
    gf: &mut Graph,
    data: &mut SamplerData,
) {
    smpl.iface.backend_apply(&mut smpl.ctx, ggml_ctx, gf, data);
}

pub fn llama_sampler_backend_set_input(smpl: &mut Sampler) {
    smpl.iface.backend_set_input(&mut smpl.ctx);
}

pub fn llama_sampler_backend_reset(smpl: &mut Sampler) {
    smpl.iface.backend_reset(&mut smpl.ctx);
}

/// `llama_split_path`. Returns the formatted split path.
pub fn llama_split_path(
    path_prefix: &str,
    split_no: i32,
    split_count: i32,
    maxlen: usize,
) -> String {
    let path = format!("{}-{:05}-of-{:05}.gguf", path_prefix, split_no, split_count);
    path.chars().take(maxlen).collect()
}

/// `llama_split_prefix`. Returns the recovered path prefix.
pub fn llama_split_prefix(
    split_path: &str,
    split_no: i32,
    split_count: i32,
    maxlen: usize,
) -> Option<String> {
    let suffix = format!("-{:05}-of-{:05}.gguf", split_no, split_count);
    split_path.strip_suffix(&suffix).map(|p| p.chars().take(maxlen).collect())
}

pub fn llama_print_system_info() -> &'static str {
    "llama.cpp Rust API port: CPU stand-in backend"
}

pub type LogCallback = fn(i32, &str);

static LOG_CALLBACK: std::sync::Mutex<Option<LogCallback>> = std::sync::Mutex::new(None);

pub fn llama_log_get() -> Option<LogCallback> {
    LOG_CALLBACK.lock().map(|g| *g).unwrap_or(None)
}

pub fn llama_log_set(callback: Option<LogCallback>) {
    if let Ok(mut g) = LOG_CALLBACK.lock() {
        *g = callback;
    }
}

fn log_message(level: i32, message: &str) {
    if let Some(cb) = llama_log_get() {
        cb(level, message);
    }
}

pub fn llama_perf_context(ctx: &Context) -> PerfContextData {
    ctx.perf.clone()
}

pub fn llama_perf_context_print(ctx: &Context) {
    let p = llama_perf_context(ctx);
    log_message(
        2,
        &format!(
            "load: {:.2} ms, prompt: {:.2} ms, eval: {:.2} ms, n_eval: {}",
            p.t_load_ms, p.t_p_eval_ms, p.t_eval_ms, p.n_eval
        ),
    );
}

pub fn llama_perf_context_reset(ctx: &mut Context) {
    ctx.perf.t_start_ms = now_ms();
    ctx.perf.t_p_eval_ms = 0.0;
    ctx.perf.t_eval_ms = 0.0;
    ctx.perf.n_p_eval = 0;
    ctx.perf.n_eval = 0;
    ctx.perf.n_reused = 0;
    ctx.perf.n_reallocs_same = 0;
    ctx.perf.n_reallocs_diff = 0;
}

pub fn llama_perf_sampler(chain: &Sampler) -> PerfSamplerData {
    match chain.ctx.as_chain() {
        Some(c) if !c.params.no_perf => PerfSamplerData {
            t_sample_ms: c.t_sample_ms,
            n_sample: c.n_sample,
        },
        _ => PerfSamplerData::default(),
    }
}

pub fn llama_perf_sampler_print(chain: &Sampler) {
    let p = llama_perf_sampler(chain);
    log_message(2, &format!("sampling: {:.2} ms, n_sample: {}", p.t_sample_ms, p.n_sample));
}

pub fn llama_perf_sampler_reset(chain: &mut Sampler) {
    if let Some(c) = as_chain_mut(chain) {
        c.t_sample_ms = 0.0;
        c.n_sample = 0;
    }
}

pub fn llama_opt_param_filter_all(_tensor: &Tensor) -> bool {
    true
}

pub fn llama_opt_init(ctx: &mut Context, model: &Model, params: OptParams) {
    let mut params = params;
    if params.n_ctx_train == 0 {
        params.n_ctx_train = ctx.n_ctx;
    }
    if params.param_filter.is_none() {
        params.param_filter = Some(llama_opt_param_filter_all);
    }
    ctx.model = model.clone();
    ctx.opt_params = Some(params);
}

/// `llama_opt_epoch`. The stand-in updates the result and calls both callbacks.
pub fn llama_opt_epoch(
    ctx: &Context,
    dataset: &Dataset,
    result_train: &mut OptResult,
    result_eval: &mut OptResult,
    idata_split: i64,
    callback_train: Option<EpochCallback>,
    callback_eval: Option<EpochCallback>,
) {
    if idata_split < 0 {
        return;
    }
    let n = dataset.rows.min(idata_split.max(0) as usize);
    if let Some(cb) = callback_train {
        result_train.loss = 1.0 / (n.max(1) as f64);
        let _ = cb(dataset, result_train);
    }
    if let Some(cb) = callback_eval {
        result_eval.loss = result_train.loss;
        let _ = cb(dataset, result_eval);
    }
    let _ = ctx.opt_params.as_ref();
}

struct LazyGrammar {
    vocab: Rc<Vocab>,
    patterns: Vec<String>,
    trigger_tokens: Vec<Token>,
    active: bool,
    text: String,
    grammar: Box<dyn crate::sampler::Sampler>,
}

struct LazyGrammarIface;

impl SamplerI for LazyGrammarIface {
    fn name<'a>(&self, _ctx: &'a SamplerContext) -> Option<&'a str> {
        Some("grammar-lazy-patterns")
    }

    fn apply(&mut self, ctx: &mut SamplerContext, cur: &mut TokenDataArray) {
        if let Some(s) = ctx.custom_mut::<LazyGrammar>() {
            if s.active {
                let mut inner = to_inner_array(cur);
                s.grammar.apply(&mut inner);
                *cur = from_inner_array(inner);
            }
        }
    }

    fn accept(&mut self, ctx: &mut SamplerContext, token: Token) {
        if let Some(s) = ctx.custom_mut::<LazyGrammar>() {
            if s.trigger_tokens.contains(&token) {
                s.active = true;
            }
            if let Some(piece) = s.vocab.pieces.get(token as usize) {
                s.text.push_str(piece);
            }
            if !s.active && s.patterns.iter().any(|p| s.text.contains(p)) {
                s.active = true;
            }
            s.grammar.accept(token);
        }
    }

    fn reset(&mut self, ctx: &mut SamplerContext) {
        if let Some(s) = ctx.custom_mut::<LazyGrammar>() {
            s.active = false;
            s.text.clear();
            s.grammar.reset();
        }
    }

    fn clone_sampler(&self, ctx: &SamplerContext) -> Option<Sampler> {
        let s = ctx.custom::<LazyGrammar>()?;
        Some(Sampler::new(
            Box::new(LazyGrammarIface),
            SamplerContext::Custom(Box::new(LazyGrammar {
                vocab: s.vocab.clone(),
                patterns: s.patterns.clone(),
                trigger_tokens: s.trigger_tokens.clone(),
                active: s.active,
                text: s.text.clone(),
                grammar: s.grammar.clone_sampler(),
            })),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_values_match_header() {
        assert_eq!(RopeType::None as i32, -1);
        assert_eq!(RopeType::Neox as i32, 2);
        assert_eq!(RopeType::Mrope as i32, 8);
        assert_eq!(RopeType::Imrope as i32, 40);
        assert_eq!(RopeType::Vision as i32, 24);
        assert_eq!(Ftype::Guessed as i32, 1024);
        assert_eq!(TokenAttrFlag::SingleWord as i32, 1 << 9);
        assert_eq!(LoadMode::Auto as i32, -1);
        assert_eq!(LLAMA_DEFAULT_SEED, u32::MAX);
        assert_eq!(LLAMA_SESSION_VERSION, 11);
    }

    #[test]
    fn defaults_match_header() {
        let m = llama_model_default_params();
        assert_eq!(m.n_gpu_layers, -1);
        assert_eq!(m.split_mode, SplitMode::Layer);
        assert!(m.use_extra_bufts);
        let c = llama_context_default_params();
        assert_eq!(c.n_ctx, 512);
        assert_eq!(c.n_batch, 2048);
        assert_eq!(c.n_ubatch, 512);
        assert!(llama_sampler_chain_default_params().no_perf);
    }

    #[test]
    fn decode_and_state_roundtrip() {
        let mut model = Model::default();
        model.n_embd = 4;
        let mut ctx = llama_init_from_model(&model, ContextParams::default()).unwrap();
        let batch = llama_batch_get_one(&[11, 12]);
        assert_eq!(llama_decode(&mut ctx, &batch), 0);
        assert_eq!(llama_get_logits(&ctx).len(), 256);
        assert_eq!(llama_get_memory(&mut ctx).tokens.len(), 2);

        let mut buf = vec![0u8; llama_state_get_size(&ctx)];
        assert_eq!(llama_state_get_data(&ctx, &mut buf), buf.len());
        let mut restored = llama_init_from_model(&model, ContextParams::default()).unwrap();
        assert_eq!(llama_state_set_data(&mut restored, &buf), buf.len());
        assert_eq!(llama_get_memory(&mut restored).tokens[1].id, 12);
    }

    #[test]
    fn sampler_chain_selects_token() {
        let model = Model::default();
        let mut ctx = llama_init_from_model(&model, ContextParams::default()).unwrap();
        assert_eq!(llama_decode(&mut ctx, &llama_batch_get_one(&[1])), 0);
        let mut chain = llama_sampler_chain_init(SamplerChainParams::default());
        assert!(llama_sampler_chain_add(&mut chain, llama_sampler_init_greedy()));
        assert_eq!(llama_sampler_chain_n(&chain), 1);
        assert_eq!(llama_sampler_sample(&mut chain, &ctx, -1), 0);
        let taken = llama_sampler_chain_remove(&mut chain, 0).unwrap();
        llama_sampler_free(taken);
    }
}
