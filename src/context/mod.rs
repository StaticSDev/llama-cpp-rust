//! Rust port of `llama-context.cpp` / `llama-context.h`.
//!
//! The C++ context owns GPU backends, a scheduler and raw output pointers.
//! Here those roles are filled by host stand-ins: a small tensor graph
//! executor in `graph.rs`, backend descriptors in this file, and shared
//! `Rc<RefCell<_>>` output arenas. Control flow, parameter resolution,
//! batching, KV wiring, state I/O and getters follow the reference.

mod graph;
mod memory;
mod model;
mod output;
mod params;
mod perf;
mod process;
mod setters;

pub use graph::{
    build_graph, graph_n_input_tensors, ubatch_prepare_reserve, verify_hadamard_graph,
    CopyExpertsInfo, FusedNode, Graph, GraphOp, GraphParams, GraphResult, GraphSignature,
    MemorySnapshot, SchedParams, Scheduler, TensorDesc,
};
pub use memory::{
    create_memory, kv_hparams, to_kv_ubatch, MemoryBreakdownData, MemoryContext,
    LLAMA_SESSION_MAGIC, LLAMA_SESSION_VERSION, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE,
    LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION,
};
pub use model::{
    Arch, CVecAdapter, HadamardInfo, LoraAdapter, LoraRef, Model, SplitMode, VocabInfo,
};
pub use output::BufferView;
pub use params::{
    clamp_seq_max, default_params, AttentionType, BackendSampler, ContextParams, FlashAttnType,
    FusedOp, FusedOpProbe, GraphType, SamplerConfig, FUSED_PROBE_DSV4_HC_COMB,
    FUSED_PROBE_DSV4_HC_POST, FUSED_PROBE_DSV4_HC_PRE, FUSED_PROBE_FLASH_ATTN,
    FUSED_PROBE_FLASH_ATTN_KV_ROWS, FUSED_PROBE_TOP_K_MASK,
};
pub use perf::{
    CrossEntropy, OptContext, OptParams, PerfContextData, TrainCallback, TrainDataset, TrainResult,
};
pub use process::needs_raw_logits;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::batch::BatchAllocr;
use crate::core::{Error, Result, SeqId, Token};
use crate::cparams::{self, CParams, PoolingType, RopeScalingType};
use crate::kv_cache::KvCache;

use graph::{
    CopyExpertsInfo as CopyExperts, GraphResult as Gres, SchedParams as SchedCfg,
    Scheduler as Sched,
};
use model::LoraRef as ActiveLora;
use params::ContextParams as CtxParams;
use perf::OptContext as Opt;

/// Backend descriptor. `id` indexes `backend_buf_exp_size`.
#[derive(Clone, Debug)]
pub struct Backend {
    pub id: usize,
    pub name: String,
    pub is_cpu: bool,
    pub n_threads: i32,
    pub supports: BTreeSet<FusedOp>,
    pub abortable: bool,
}

/// `llama_cross` stand-in for T5 encoder output.
#[derive(Clone, Debug, Default)]
pub struct CrossInfo {
    pub n_embd: usize,
    pub n_enc: usize,
    pub v_embd: Vec<f32>,
    pub seq_ids_enc: Vec<Vec<SeqId>>,
}

/// `llama_moe_cache` stand-in: expert cache sizes and copied experts.
#[derive(Clone, Debug, Default)]
pub struct MoeCache {
    pub size: usize,
    pub cpu_share: f32,
    pub copied: Vec<usize>,
}

impl MoeCache {
    pub fn new(size: usize, cpu_share: f32) -> Self {
        Self {
            size,
            cpu_share,
            copied: Vec::new(),
        }
    }

    /// `llama_moe_cache::copy`: record an expert copy.
    pub fn copy(&mut self, expert: usize) -> bool {
        if self.size == 0 {
            return false;
        }
        if !self.copied.contains(&expert) {
            self.copied.push(expert);
        }
        true
    }
}

/// `llama_context`.
pub struct LlamaContext {
    pub(crate) model: Model,
    pub(crate) params: CtxParams,
    pub(crate) cparams: CParams,
    pub(crate) balloc: BatchAllocr,
    pub(crate) cvec: CVecAdapter,
    pub(crate) loras: Vec<ActiveLora>,
    pub(crate) cross: CrossInfo,
    pub(crate) memory: Option<Rc<RefCell<KvCache>>>,
    pub(crate) moe_cache: MoeCache,
    pub(crate) logits: output::BufferView<f32>,
    pub(crate) embd: output::BufferView<f32>,
    pub(crate) embd_nextn: output::BufferView<f32>,
    pub(crate) embd_layer_inp: Vec<output::BufferView<f32>>,
    pub(crate) embd_batch_idxs: Vec<usize>,
    pub(crate) sampling: output::SamplingInfo,
    pub(crate) embd_seq: BTreeMap<SeqId, Vec<f32>>,
    pub(crate) output_f32: Rc<RefCell<Vec<f32>>>,
    pub(crate) output_tokens: Rc<RefCell<Vec<Token>>>,
    pub(crate) n_input_tensors: u32,
    pub(crate) n_outputs: i64,
    pub(crate) output_ids: Vec<i64>,
    pub(crate) output_swaps: Vec<(u32, u32)>,
    pub(crate) sched: Sched,
    pub(crate) sched_params_cur: SchedCfg,
    pub(crate) sched_need_reserve: bool,
    pub(crate) graph_has_logits: bool,
    pub(crate) gf_res_prev: [Gres; 2],
    pub(crate) gf_res_reserve: Gres,
    pub(crate) gf_res_prev_active: Option<usize>,
    pub(crate) copy_experts: CopyExperts,
    pub(crate) n_reallocs_same: i32,
    pub(crate) n_reallocs_diff: i32,
    pub(crate) backends: Vec<Backend>,
    pub(crate) backend_buf_exp_size: Vec<usize>,
    pub(crate) set_n_threads_fns: Vec<usize>,
    pub(crate) opt_ctx: Option<Opt>,
    pub(crate) threadpool: Option<usize>,
    pub(crate) threadpool_batch: Option<usize>,
    pub(crate) abort_callback: Option<std::sync::Arc<dyn Fn() -> bool>>,
    pub(crate) layer_inp_dev: Option<Vec<f32>>,
    pub(crate) h_nextn_dev: Option<Vec<f32>>,
    pub(crate) mem_storage: BTreeMap<SeqId, Vec<u8>>,
    pub(crate) has_evaluated_once: bool,
    pub(crate) graph_reuse_disable: bool,
    pub(crate) hadamard_verified: bool,
    pub(crate) t_start_us: i64,
    pub(crate) t_load_us: i64,
    pub(crate) t_p_eval_us: i64,
    pub(crate) t_eval_us: i64,
    pub(crate) t_compute_start_us: i64,
    pub(crate) n_queued_tokens: i64,
    pub(crate) n_p_eval: i32,
    pub(crate) n_eval: i32,
    pub(crate) n_reused: i32,
}

fn ggml_pad(x: u32, n: u32) -> u32 {
    if n == 0 {
        return x;
    }
    ((x + n - 1) / n) * n
}

fn map_rope_scaling(ty: crate::hparams::RopeScalingType) -> RopeScalingType {
    match ty {
        crate::hparams::RopeScalingType::None => RopeScalingType::None,
        crate::hparams::RopeScalingType::Linear => RopeScalingType::Linear,
        crate::hparams::RopeScalingType::Yarn => RopeScalingType::Yarn,
        crate::hparams::RopeScalingType::Extrapolation => RopeScalingType::Longrope,
    }
}

fn map_pooling(ty: crate::hparams::PoolingType) -> PoolingType {
    match ty {
        crate::hparams::PoolingType::None => PoolingType::None,
        crate::hparams::PoolingType::Mean => PoolingType::Mean,
        crate::hparams::PoolingType::Cls => PoolingType::Cls,
        crate::hparams::PoolingType::Last => PoolingType::Last,
        crate::hparams::PoolingType::Rank => PoolingType::Rank,
        crate::hparams::PoolingType::Unspecified => PoolingType::Unspecified,
    }
}

fn yarn_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

impl LlamaContext {
    /// `llama_init_from_model` plus the `llama_context` constructor.
    pub fn new(model: &Model, params: CtxParams) -> Result<Self> {
        if params.n_batch == 0 && params.n_ubatch == 0 {
            return Err(Error::new("n_batch and n_ubatch cannot both be zero"));
        }
        let hparams = model.hparams.clone();
        let n_ctx_requested = if params.n_ctx == 0 {
            hparams.n_ctx_train
        } else {
            params.n_ctx
        };
        if n_ctx_requested == 0 {
            return Err(Error::new("n_ctx and n_ctx_train cannot both be zero"));
        }
        if params.n_seq_max == 0 && params.flash_attn_type != FlashAttnType::Disabled {
            crate::llama_log_debug!("n_seq_max = 0 with flash attention");
        }

        let mut cparams = CParams::default();
        cparams.n_seq_max = params.n_seq_max.max(1).min(cparams::LLAMA_MAX_SEQ);
        cparams.n_rs_seq = params.n_rs_seq;
        if cparams.n_rs_seq > 0 && (!model.arch.supports_rs_rollback() || !hparams.causal_attn) {
            crate::llama_log_debug!(
                "n_rs_seq requested but model does not support rollback; clamping to 0"
            );
            cparams.n_rs_seq = 0;
        }
        cparams.rs_replay = cparams.n_rs_seq > 0 && model.arch.uses_rs_replay();

        cparams.n_threads = params.n_threads;
        cparams.n_threads_batch = params.n_threads_batch;
        cparams.yarn_ext_factor = if params.yarn_ext_factor >= 0.0 {
            params.yarn_ext_factor
        } else {
            hparams.yarn_ext_factor
        };
        cparams.yarn_attn_factor = if params.yarn_attn_factor >= 0.0 {
            params.yarn_attn_factor
        } else {
            hparams.rope_attn_factor
        };
        cparams.yarn_beta_fast = if params.yarn_beta_fast >= 0.0 {
            params.yarn_beta_fast
        } else {
            hparams.yarn_beta_fast
        };
        cparams.yarn_beta_slow = if params.yarn_beta_slow >= 0.0 {
            params.yarn_beta_slow
        } else {
            hparams.yarn_beta_slow
        };
        cparams.embeddings = params.embeddings;
        cparams.offload_kqv = params.offload_kqv;
        cparams.no_perf = params.no_perf;
        cparams.embeddings_layer_inp = vec![false; hparams.n_layer() as usize + 1];
        cparams.embeddings_layer_inp_dev = false;
        cparams.embeddings_nextn_dev = false;
        cparams.ctx_type = params.ctx_type;
        cparams.rope_scaling_type = if params.rope_scaling_type == RopeScalingType::Unspecified {
            map_rope_scaling(hparams.rope_scaling_type_train)
        } else {
            params.rope_scaling_type
        };
        cparams.pooling_type = params.pooling_type;
        cparams.n_ctx = n_ctx_requested;
        cparams.rope_freq_base = if params.rope_freq_base == 0.0 {
            hparams.rope_freq_base_train
        } else {
            params.rope_freq_base
        };
        cparams.rope_freq_scale = if params.rope_freq_scale == 0.0 {
            hparams.rope_freq_scale_train
        } else {
            params.rope_freq_scale
        };
        cparams.n_ctx_orig_yarn = if params.yarn_orig_ctx != 0 {
            params.yarn_orig_ctx
        } else if hparams.n_ctx_orig_yarn != 0 {
            hparams.n_ctx_orig_yarn
        } else {
            hparams.n_ctx_train
        };

        if cparams.rope_scaling_type == RopeScalingType::None {
            cparams.rope_freq_scale = 1.0;
        }
        if cparams.yarn_ext_factor < 0.0 {
            cparams.yarn_ext_factor = if cparams.rope_scaling_type == RopeScalingType::Yarn {
                1.0
            } else {
                0.0
            };
        }
        if cparams.yarn_ext_factor != 0.0 {
            let factor = 1.0 / cparams.rope_freq_scale.max(f32::MIN_POSITIVE);
            if hparams.rope_yarn_log_mul != 0.0 {
                let mscale_all = hparams.rope_yarn_log_mul;
                let mscale = if model.arch == Arch::Deepseek2 && mscale_all != 1.0 {
                    mscale_all
                } else {
                    1.0
                };
                cparams.yarn_attn_factor =
                    yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all);
                crate::llama_log_warn!(
                    "setting new yarn_attn_factor = {:.4} (mscale = {:.1}, mscale_all = {:.1})",
                    cparams.yarn_attn_factor,
                    mscale,
                    mscale_all
                );
            } else {
                cparams.yarn_attn_factor = yarn_mscale(factor, 1.0);
            }
            cparams.yarn_attn_factor *= 1.0 / (1.0 + 0.1 * factor.ln());
        }
        cparams.yarn_attn_factor *= hparams.rope_attn_factor;

        if cparams.pooling_type == PoolingType::Unspecified {
            cparams.pooling_type =
                if hparams.pooling_type == crate::hparams::PoolingType::Unspecified {
                    PoolingType::None
                } else {
                    map_pooling(hparams.pooling_type)
                };
        }
        cparams.causal_attn = match params.attention_type {
            AttentionType::Unspecified => hparams.causal_attn,
            AttentionType::Causal => true,
            AttentionType::NonCausal => false,
        };
        cparams.flash_attn = params.flash_attn_type != FlashAttnType::Disabled;
        cparams.auto_fa = params.flash_attn_type == FlashAttnType::Auto;

        if model.arch == Arch::Gemma4Assistant {
            if params.ctx_other.is_none() {
                return Err(Error::new("Gemma4Assistant requires ctx_other to be set"));
            }
            if let Some(other) = params.ctx_other.as_ref() {
                let other = other.borrow();
                if !cparams.auto_fa && cparams.flash_attn != other.cparams.flash_attn {
                    crate::llama_log_warn!("flash_attn does not match the shared KV context");
                }
                cparams.flash_attn = other.cparams.flash_attn;
                cparams.auto_fa = false;
            }
        }
        if (model.arch == Arch::Eagle3 || model.arch == Arch::Dflash)
            && (!model.has_tok_embd || !model.has_output)
            && params.ctx_other.is_none()
        {
            return Err(Error::new(format!(
                "{} requires ctx_other to be set",
                model.arch.name()
            )));
        }

        cparams.fused_dsv4_hc_pre = true;
        cparams.fused_dsv4_hc_comb = true;
        cparams.fused_dsv4_hc_post = true;
        cparams.auto_fhc = true;
        cparams.fused_top_k_mask = true;
        cparams.auto_ftkm = hparams.indexer_top_k > 0;

        let n_batch = if params.n_batch == 0 {
            n_ctx_requested
        } else {
            params.n_batch
        };
        cparams.n_batch = if cparams.causal_attn {
            n_ctx_requested.min(n_batch)
        } else {
            n_batch
        };
        let n_ubatch = if params.n_ubatch == 0 {
            n_batch
        } else {
            params.n_ubatch
        };
        cparams.n_ubatch = cparams.n_batch.min(n_ubatch);
        let has_encoder = matches!(model.arch, Arch::T5 | Arch::Dflash | Arch::Eagle3);
        cparams.n_outputs_max = if params.n_outputs_max == 0 || has_encoder {
            cparams.n_batch
        } else {
            params.n_outputs_max
        };
        cparams.n_outputs_max_per_seq = if params.n_outputs_max_per_seq == 0 {
            cparams.n_outputs_max
        } else {
            params.n_outputs_max_per_seq.min(cparams.n_outputs_max)
        };

        cparams.op_offload = params.op_offload;
        cparams.kv_unified = params.kv_unified;
        if model.arch == Arch::HyV4Flash && !cparams.kv_unified && cparams.n_seq_max > 1 {
            crate::llama_log_warn!("{} needs a unified KV cache", model.arch.name());
            cparams.kv_unified = true;
        }
        cparams.moe_cache_size = params.moe_cache_size;
        cparams.moe_cache_cpu_share = params.moe_cache_cpu_share;

        cparams.fused_kv_rows = false;
        cparams.auto_fkvr = std::env::var("LLAMA_KV_ROWS_DISABLE").is_err();
        cparams.pipeline_parallel = false;
        cparams.training = false;
        cparams.moe_random_routing = std::env::var("LLAMA_MOE_RANDOM_ROUTING")
            .map(|v| v.parse::<i32>().unwrap_or(0) != 0)
            .unwrap_or(false);
        if cparams.moe_random_routing && hparams.n_expert > 0 {
            crate::llama_log_warn!("MoE random routing enabled: output is not valid");
        }

        let graph_reuse_disable = std::env::var("LLAMA_GRAPH_REUSE_DISABLE")
            .map(|v| v.parse::<i32>().unwrap_or(0) != 0)
            .unwrap_or(false);
        if graph_reuse_disable {
            crate::llama_log_warn!("graph reuse disabled");
        }

        cparams.n_ctx = ggml_pad(cparams.n_ctx, 256);
        if cparams.kv_unified {
            cparams.n_ctx_seq = cparams.n_ctx;
        } else {
            cparams.n_ctx_seq = ggml_pad(cparams.n_ctx / cparams.n_seq_max.max(1), 256);
            if cparams.n_ctx_seq == 0 {
                return Err(Error::new("n_ctx_seq == 0"));
            }
            if cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max {
                cparams.n_ctx = cparams.n_ctx_seq * cparams.n_seq_max;
                crate::llama_log_warn!(
                    "n_ctx is not divisible by n_seq_max, rounded down to {}",
                    cparams.n_ctx
                );
            }
        }

        crate::llama_log_info!(
            "n_seq_max = {}, n_ctx = {}, n_ctx_seq = {}, n_batch = {}, n_ubatch = {}",
            cparams.n_seq_max,
            cparams.n_ctx,
            cparams.n_ctx_seq,
            cparams.n_batch,
            cparams.n_ubatch
        );
        if cparams.n_ctx_seq < hparams.n_ctx_train {
            crate::llama_log_info!(
                "n_ctx_seq ({}) < n_ctx_train ({})",
                cparams.n_ctx_seq,
                hparams.n_ctx_train
            );
        } else if cparams.n_ctx_seq > hparams.n_ctx_train {
            crate::llama_log_warn!(
                "n_ctx_seq ({}) > n_ctx_train ({})",
                cparams.n_ctx_seq,
                hparams.n_ctx_train
            );
        }

        let mut backends = Vec::new();
        for (i, dev) in model.devices.iter().enumerate() {
            backends.push(Backend {
                id: i,
                name: dev.clone(),
                is_cpu: dev == "CPU",
                n_threads: params.n_threads,
                supports: model.fused_support.clone(),
                abortable: params.abort_callback.is_some(),
            });
        }
        if let Some(other) = params.ctx_other.as_ref() {
            let other = other.borrow();
            for dev in &other.model.devices {
                if !model.devices.contains(dev) {
                    let id = backends.len();
                    backends.push(Backend {
                        id,
                        name: dev.clone(),
                        is_cpu: dev == "CPU",
                        n_threads: params.n_threads,
                        supports: model.fused_support.clone(),
                        abortable: params.abort_callback.is_some(),
                    });
                }
            }
        }
        if std::env::var("LLAMA_ACCEL").is_ok() {
            let id = backends.len();
            backends.push(Backend {
                id,
                name: "ACCEL".to_string(),
                is_cpu: false,
                n_threads: params.n_threads,
                supports: BTreeSet::new(),
                abortable: params.abort_callback.is_some(),
            });
        }
        if !backends.iter().any(|b| b.is_cpu) {
            let id = backends.len();
            backends.push(Backend {
                id,
                name: "CPU".to_string(),
                is_cpu: true,
                n_threads: params.n_threads,
                supports: BTreeSet::new(),
                abortable: params.abort_callback.is_some(),
            });
        }
        let set_n_threads_fns: Vec<usize> = backends.iter().map(|b| b.id).collect();
        let backend_buf_exp_size = vec![0usize; backends.len()];

        let sched_params = SchedCfg {
            max_nodes: 1024,
            parallel: false,
            op_offload: cparams.op_offload,
        };
        let sched = Sched::new(sched_params);
        let mut this = Self {
            model: model.clone(),
            params: params.clone(),
            cparams,
            balloc: BatchAllocr::new(
                hparams.n_pos_per_embd() as usize,
                model.arch.supports_mixed_batch(),
            ),
            cvec: CVecAdapter::default(),
            loras: Vec::new(),
            cross: CrossInfo::default(),
            memory: None,
            moe_cache: MoeCache::default(),
            logits: output::BufferView::default(),
            embd: output::BufferView::default(),
            embd_nextn: output::BufferView::default(),
            embd_layer_inp: Vec::new(),
            embd_batch_idxs: Vec::new(),
            sampling: output::SamplingInfo::default(),
            embd_seq: BTreeMap::new(),
            output_f32: Rc::new(RefCell::new(Vec::new())),
            output_tokens: Rc::new(RefCell::new(Vec::new())),
            n_input_tensors: 0,
            n_outputs: 0,
            output_ids: Vec::new(),
            output_swaps: Vec::new(),
            sched,
            sched_params_cur: sched_params,
            sched_need_reserve: true,
            graph_has_logits: true,
            gf_res_prev: [Gres::default(), Gres::default()],
            gf_res_reserve: Gres::default(),
            gf_res_prev_active: None,
            copy_experts: CopyExperts::default(),
            n_reallocs_same: 0,
            n_reallocs_diff: 0,
            backends,
            backend_buf_exp_size,
            set_n_threads_fns,
            opt_ctx: None,
            threadpool: None,
            threadpool_batch: None,
            abort_callback: params.abort_callback.clone(),
            layer_inp_dev: None,
            h_nextn_dev: None,
            mem_storage: BTreeMap::new(),
            has_evaluated_once: false,
            graph_reuse_disable,
            hadamard_verified: false,
            t_start_us: now_us(),
            t_load_us: 0,
            t_p_eval_us: 0,
            t_eval_us: 0,
            t_compute_start_us: 0,
            n_queued_tokens: 0,
            n_p_eval: 0,
            n_eval: 0,
            n_reused: 0,
        };

        let n_vocab = model.vocab.n_tokens.max(1) as usize;
        this.sampling.token_ids_full_vocab = (0..n_vocab).map(|i| i as Token).collect();

        for config in &params.samplers {
            if config.sampler.n_chain == 0 {
                return Err(Error::new("the backend samplers must be chains"));
            }
            this.set_sampler(config.seq_id, Some(config.sampler.clone()));
        }

        if this.output_reserve(params.n_seq_max as i32) < params.n_seq_max {
            return Err(Error::new("failed to reserve initial output buffer"));
        }

        if !hparams.vocab_only {
            let other = this
                .params
                .ctx_other
                .as_ref()
                .and_then(|ctx| ctx.borrow().memory.clone());
            this.memory = Some(Rc::new(RefCell::new(memory::create_memory(
                model,
                &this.cparams,
                &this.params,
                other,
            )?)));
        }

        let pipeline_parallel = model.n_devices() > 1
            && model.n_gpu_layers > model.hparams.n_layer_all
            && model.split_mode == SplitMode::Layer
            && cparams_offload(&this.cparams)
            && !model.has_tensor_overrides
            && this.cparams.moe_cache_size == 0;
        this.cparams.pipeline_parallel = pipeline_parallel;
        if this.cparams.moe_cache_size > 0 {
            this.moe_cache = MoeCache::new(
                this.cparams.moe_cache_size,
                this.cparams.moe_cache_cpu_share,
            );
        }
        this.sched_reserve()?;
        if !this.cparams.flash_attn && this.params.type_v.is_quantized() {
            return Err(Error::new("quantized V cache requires Flash Attention"));
        }
        Ok(this)
    }

    /// `llama_synchronize`.
    pub fn synchronize(&mut self) {
        self.sched.synchronize();
        if self.n_queued_tokens == 1 {
            if !self.cparams.no_perf {
                self.t_eval_us += now_us() - self.t_compute_start_us;
            }
            self.n_eval += 1;
        } else if self.n_queued_tokens > 1 {
            if !self.cparams.no_perf {
                self.t_p_eval_us += now_us() - self.t_compute_start_us;
            }
            self.n_p_eval += self.n_queued_tokens as i32;
        }
        if self.n_queued_tokens > 0 && !self.has_evaluated_once {
            self.t_load_us = now_us() - self.t_start_us;
            self.has_evaluated_once = true;
        }
        self.n_queued_tokens = 0;
        self.t_compute_start_us = 0;
    }

    /// `llama_get_model`.
    pub fn get_model(&self) -> &Model {
        &self.model
    }

    /// `get_cparams`.
    pub fn get_cparams(&self) -> &CParams {
        &self.cparams
    }

    /// `llama_n_ctx`.
    pub fn n_ctx(&self) -> u32 {
        self.cparams.n_ctx
    }

    /// `llama_n_ctx_seq`.
    pub fn n_ctx_seq(&self) -> u32 {
        self.cparams.n_ctx_seq
    }

    /// `llama_n_batch`.
    pub fn n_batch(&self) -> u32 {
        self.cparams.n_batch
    }

    /// `llama_n_ubatch`.
    pub fn n_ubatch(&self) -> u32 {
        self.cparams.n_ubatch
    }

    /// `llama_n_seq_max`.
    pub fn n_seq_max(&self) -> u32 {
        self.cparams.n_seq_max
    }

    /// `llama_n_threads`.
    pub fn n_threads(&self) -> i32 {
        self.cparams.n_threads
    }

    /// `llama_n_threads_batch`.
    pub fn n_threads_batch(&self) -> i32 {
        self.cparams.n_threads_batch
    }

    /// `llama_get_memory`.
    pub fn get_memory(&self) -> Option<Rc<RefCell<KvCache>>> {
        self.memory.clone()
    }

    /// `llama_pooling_type`.
    pub fn pooling_type(&self) -> PoolingType {
        self.cparams.pooling_type
    }

    /// `llama_graph_reserve` public compatibility wrapper.
    pub fn graph_reserve_public(
        &mut self,
        n_tokens: u32,
        n_seqs: u32,
        n_outputs: u32,
    ) -> Result<Graph> {
        let snapshot = self.memory.as_ref().map(|kv| {
            let cache = kv.borrow();
            MemorySnapshot {
                n_kv: cache.get_size(),
                n_kv_rows: 0,
                type_k: cache.type_k(),
                type_v: cache.type_v(),
            }
        });
        self.graph_reserve(n_tokens, n_seqs, n_outputs, snapshot, false, None)
    }

    /// `llama_process` compatibility wrapper.
    pub fn process(&mut self, gtype: GraphType, batch: &crate::batch::BatchExt) -> i32 {
        match gtype {
            GraphType::Encoder => self.encode(batch),
            _ => self.decode(batch),
        }
    }
}

fn cparams_offload(c: &CParams) -> bool {
    c.offload_kqv
}

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

impl Model {
    fn n_devices(&self) -> usize {
        self.devices.len().max(1)
    }
}

#[cfg(test)]
mod tests;
