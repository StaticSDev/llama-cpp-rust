//! Port of `tools/server/main.cpp` and the CLI entry of `server.cpp`.
//!
//! Server argument parsing, role selection, parameter preparation and the
//! `llama_server()` entry point. Signal handlers are left to the host since
//! installing them needs platform FFI; `request_shutdown()` is the std-only
//! replacement used by embedders.

use crate::common::Params;
use crate::core::{Error, Result};
use crate::cparams::PoolingType;
use crate::tools::server::common::{parse_csv_row, ErrorType, ServerRoleType};
use crate::tools::server::context::ServerContext;
use crate::tools::server::http::HttpContext;
use crate::tools::server::routes::{register_routes, register_stream_routes, setup_features, ServerRoutes};
use crate::tools::server::task::TaskResult;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// `server_run_opts` from `server.cpp`.
#[derive(Clone, Copy, Debug)]
pub struct ServerRunOpts {
    /// Off when embedded in llama-cli, which handles signals itself.
    pub install_signal_handlers: bool,
    /// Off when the caller already resolved `-hf` / docker models.
    pub resolve_model: bool,
}

impl Default for ServerRunOpts {
    fn default() -> Self {
        Self {
            install_signal_handlers: true,
            resolve_model: true,
        }
    }
}

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Trigger a graceful shutdown from the host application.
pub fn request_shutdown() {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

pub fn should_shutdown() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

/// Server usage text. Help is returned instead of starting the server.
pub fn usage() -> &'static str {
    "usage: llama-server [options]\n\
     \n\
     server-specific options:\n\
       -m, --model FNAME          model path\n\
       -hf, --hf-repo REPO        Hugging Face model repository\n\
       -c, --ctx-size N           size of the prompt context\n\
       -n, --n-predict N          number of tokens to predict\n\
       -ngl, --gpu-layers N       number of layers to store in VRAM\n\
       -t, --threads N            number of threads\n\
       -np, --parallel N          number of server slots\n\
       --host HOST                addresses to listen on, comma-separated\n\
       --port N                   port to listen (default: 8080)\n\
       --api-key KEY              API key, comma-separated list allowed\n\
       --api-prefix PREFIX        path prefix for all routes\n\
       --slots, --no-slots        enable or disable /slots\n\
       --props, --no-props        enable or disable /props\n\
       --metrics, --no-metrics    enable or disable /metrics\n\
       --path PATH                path to serve static files from\n\
       --cors-origins ORIGINS     allowed origins for CORS\n\
       --embedding, --rerank      restrict to embedding or ranking\n\
       --ui, --no-ui              enable or disable Llama Chat\n\
       --timeout N                read and write timeout in seconds\n"
}

/// Parse server CLI arguments into the shared parameter struct.
/// Returns `false` when `--help` was requested.
pub fn parse_server_args(args: &[String], params: &mut Params) -> Result<bool> {
    // llama.cpp defaults n_parallel to -1 (auto) for the server example.
    params.n_parallel = -1;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "-h" || arg == "--help" {
            print!("{}", usage());
            return Ok(false);
        }
        macro_rules! value {
            () => {{
                i += 1;
                args.get(i).ok_or_else(|| Error::new(format!("missing value for {}", arg)))?.as_str()
            }};
        }
        macro_rules! int_value {
            () => {{
                let v: i32 = value!().parse().map_err(|_| Error::new(format!("invalid integer for {}", arg)))?;
                v
            }};
        }
        macro_rules! float_value {
            () => {{
                let v: f32 = value!().parse().map_err(|_| Error::new(format!("invalid number for {}", arg)))?;
                v
            }};
        }
        match arg {
            "-m" | "--model" => params.model.path = value!().to_string(),
            "-hf" | "--hf-repo" => params.model.hf_repo = value!().to_string(),
            "-c" | "--ctx-size" => params.n_ctx = int_value!(),
            "-n" | "--n-predict" => params.n_predict = int_value!(),
            "-ngl" | "--gpu-layers" | "--n-gpu-layers" => params.n_gpu_layers = int_value!(),
            "-t" | "--threads" => params.cpuparams.n_threads = int_value!(),
            "-tb" | "--threads-batch" => params.cpuparams_batch.n_threads = int_value!(),
            "-np" | "--parallel" => {
                let n = int_value!();
                if n == 0 {
                    return Err(Error::new("error: invalid value for n_parallel"));
                }
                params.n_parallel = n;
            }
            "--temp" | "--temperature" => params.sampling.temp = float_value!(),
            "--top-k" => params.sampling.top_k = int_value!(),
            "--top-p" => params.sampling.top_p = float_value!(),
            "--min-p" => params.sampling.min_p = float_value!(),
            "--seed" => params.sampling.seed = int_value!() as u32,
            "--repeat-penalty" => params.sampling.penalty_repeat = float_value!(),
            "--repeat-last-n" => params.sampling.penalty_last_n = int_value!(),
            "--embedding" | "--embeddings" => params.embedding = true,
            "--rerank" | "--reranking" => {
                params.embedding = true;
                params.pooling_type = PoolingType::Rank;
            }
            "--alias" => {
                for alias in parse_csv_row(value!()) {
                    params.model_alias.insert(alias);
                }
            }
            "--tags" => {
                for tag in parse_csv_row(value!()) {
                    params.model_tags.insert(tag);
                }
            }
            "--host" => {
                let hosts = parse_csv_row(value!());
                if hosts.is_empty() {
                    return Err(Error::new("--host requires at least one address"));
                }
                params.hostnames = hosts;
            }
            "--port" => params.port = int_value!(),
            "--reuse-port" => params.reuse_port = true,
            "--path" => params.public_path = value!().to_string(),
            "--api-prefix" => params.api_prefix = value!().to_string(),
            "--api-key" => {
                for key in parse_csv_row(value!()) {
                    params.api_keys.push(key);
                }
            }
            "--api-key-file" => {
                let path = value!();
                let text = std::fs::read_to_string(path)
                    .map_err(|e| Error::new(format!("error: failed to open file '{}': {}", path, e)))?;
                for line in text.lines() {
                    if !line.is_empty() && !line.starts_with('#') {
                        params.api_keys.push(line.trim().to_string());
                    }
                }
            }
            "--cors-origins" => {
                params.cors_origins = value!().to_string();
                params.cors_origins_explicit = true;
            }
            "--cors-methods" => params.cors_methods = value!().to_string(),
            "--cors-headers" => params.cors_headers = value!().to_string(),
            "--cors-credentials" => params.cors_credentials = true,
            "--no-cors-credentials" => params.cors_credentials = false,
            "--slots" => params.endpoint_slots = true,
            "--no-slots" => params.endpoint_slots = false,
            "--props" => params.endpoint_props = true,
            "--no-props" => params.endpoint_props = false,
            "--metrics" => params.endpoint_metrics = true,
            "--no-metrics" => params.endpoint_metrics = false,
            "--slot-save-path" => params.slot_save_path = value!().to_string(),
            "--media-path" => params.media_path = value!().to_string(),
            "--media-allow-private-urls" => params.media_allow_private_urls = true,
            "--cache-prompt" => params.cache_prompt = true,
            "--no-cache-prompt" => params.cache_prompt = false,
            "--cache-reuse" => params.n_cache_reuse = int_value!(),
            "--cache-idle-slots" => params.cache_idle_slots = true,
            "--no-cache-idle-slots" => params.cache_idle_slots = false,
            "--kv-unified" => params.kv_unified = true,
            "--kv-unified-per-slot" => params.kv_unified_per_slot = int_value!(),
            "--timeout" | "-to" => {
                let n = int_value!();
                params.timeout_read = n;
                params.timeout_write = n;
            }
            "--sse-ping-interval" => params.sse_ping_interval = int_value!(),
            "--ui" | "--webui" => params.ui = true,
            "--no-ui" | "--no-webui" => params.ui = false,
            "--ui-agent" => params.ui_agent = true,
            "--ui-config" | "--webui-config" => params.ui_config_json = value!().to_string(),
            "--ui-config-file" | "--webui-config-file" => {
                params.ui_config_json = std::fs::read_to_string(value!()).unwrap_or_default();
            }
            "--tools" => params.server_tools = parse_csv_row(value!()),
            "--tools-runtime" => params.server_tools_runtime = value!().to_string(),
            "--allow-unauthenticated-tools" => params.server_tools_allow_unauth = true,
            "--mcp-servers-config" => params.mcp_servers_config = value!().to_string(),
            "--mcp-servers-json" => params.mcp_servers_json = value!().to_string(),
            "-ag" | "--agent" => {
                params.server_tools = vec!["all".to_string()];
                params.ui_mcp_proxy = true;
            }
            "-no-ag" | "--no-agent" => {
                params.server_tools.clear();
                params.ui_mcp_proxy = false;
            }
            "--ui-mcp-proxy" | "--webui-mcp-proxy" => params.ui_mcp_proxy = true,
            "--no-ui-mcp-proxy" | "--no-webui-mcp-proxy" => params.ui_mcp_proxy = false,
            "--ui-mcp-proxy-allow-private-urls" => params.ui_mcp_proxy_allow_private_urls = true,
            "--connect" => params.server_connect = true,
            "--connect-code" => {
                let code = value!().replace('-', "").replace(' ', "");
                if code.len() != 40 {
                    return Err(Error::new(format!("error: invalid connect code, must be 40 characters")));
                }
                params.server_connect_code = code;
            }
            "--ssl-key-file" => params.ssl_file_key = value!().to_string(),
            "--ssl-cert-file" => params.ssl_file_cert = value!().to_string(),
            "--chat-template-kwargs" => {
                let json = crate::json::Json::parse(value!())?;
                if let crate::json::Json::Object(entries) = json {
                    for (key, val) in entries {
                        params.default_template_kwargs.insert(key, val.dump_safe(-1));
                    }
                }
            }
            "--models-dir" => params.models_dir = value!().to_string(),
            "--models-max" => params.models_max = int_value!(),
            "--models-preset" => params.models_preset = value!().to_string(),
            "--models-preset-hf" => params.models_preset_hf = value!().to_string(),
            "--offline" => params.offline = true,
            "--verbose" | "-v" => params.verbosity += 1,
            other => return Err(Error::new(format!("unknown argument: {}", other))),
        }
        i += 1;
    }
    Ok(true)
}

/// `server_role_detect()`: no model means router mode.
pub fn server_role_detect(params: &Params, role: &ServerRoleType) -> ServerRoleType {
    if *role == ServerRoleType::Router || *role == ServerRoleType::Download || *role == ServerRoleType::Measure {
        return *role;
    }
    let has_model = !params.model.path.is_empty()
        || !params.model.url.is_empty()
        || !params.model.hf_repo.is_empty()
        || !params.model.docker_repo.is_empty();
    if has_model {
        ServerRoleType::Instance
    } else {
        ServerRoleType::Router
    }
}

/// `server_prepare_params()`.
pub fn server_prepare_params(params: &mut Params, role: ServerRoleType) -> bool {
    if role != ServerRoleType::Router {
        if params.embedding && params.n_batch > params.n_ubatch {
            params.n_batch = params.n_ubatch;
        }
        if params.n_parallel < 0 {
            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }
    if params.kv_unified_per_slot > 0 && params.n_ctx == 0 && params.fit_params_min_ctx != i32::MAX {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
    }
    let model_name = params.model.get_name().to_string();
    if params.model_alias.is_empty() && !model_name.is_empty() {
        params.model_alias.insert(model_name);
    }
    if params.server_connect && server_connect_unavailable() {
        return false;
    }
    if !params.server_connect && !params.server_connect_code.is_empty() {
        params.server_connect_code.clear();
    }
    true
}

fn server_connect_unavailable() -> bool {
    false
}

/// Build the server state, HTTP routes and task loop, then block until shutdown.
pub fn server_run(params: &mut Params, args: &[String], role_type: ServerRoleType, opts: &ServerRunOpts) -> i32 {
    if !server_prepare_params(params, role_type) {
        return 1;
    }

    let ctx_server = Arc::new(ServerContext::new(role_type));
    let routes = ServerRoutes::new(params, &ctx_server);

    // A std-only server has no inference engine. Every posted task gets a
    // prompt error response so handlers never block on a missing result.
    let ctx_for_tasks = Arc::clone(&ctx_server);
    ctx_server.queue_tasks.on_new_task(move |task, _yielding| {
        let result = TaskResult::Error {
            id: task.id,
            kind: ErrorType::NotSupported,
            message: "no inference engine is attached to the server".to_string(),
            n_prompt_tokens: 0,
            n_ctx: 0,
        };
        ctx_for_tasks.queue_results.send(result);
        true
    });

    let sleep_callback = routes.sleeping_callback();
    ctx_server
        .queue_tasks
        .on_sleeping_state(move |is_sleeping| sleep_callback(is_sleeping));

    let mut http = HttpContext::new();
    if !http.init(params) {
        return 1;
    }
    register_routes(&http, &routes);
    register_stream_routes(&http);
    if !setup_features(&http, params) {
        return 1;
    }

    let loop_ctx = Arc::clone(&ctx_server);
    let idle_ms = params.sleep_idle_seconds as i64 * 1000;
    let loop_handle = std::thread::spawn(move || loop_ctx.queue_tasks.start_loop(idle_ms));

    if !http.start() {
        ctx_server.terminate();
        let _ = loop_handle.join();
        return 1;
    }
    http.set_ready(false);
    if role_type == ServerRoleType::Instance {
        if !opts.resolve_model && params.model.path.is_empty() && params.model.hf_repo.is_empty() {
            // Embedded callers resolve the model themselves.
        } else if !ctx_server.load_model(params) {
            http.stop();
            ctx_server.terminate();
            let _ = loop_handle.join();
            return 1;
        }
    } else {
        ctx_server.load_model(params);
    }
    routes.update_meta(&ctx_server);
    http.set_ready(true);

    for address in http.listening_addresses() {
        eprintln!("srv  : listening on {}", address);
    }
    if opts.install_signal_handlers {
        // Signal handlers need platform FFI; embedders call request_shutdown().
    }
    let _ = args;

    while !should_shutdown() && !http.is_stopped() {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    http.set_ready(false);
    ctx_server.terminate();
    http.stop();
    let _ = loop_handle.join();
    http.join();
    SHUTDOWN.store(false, Ordering::Relaxed);
    0
}

/// `llama_server(int argc, char ** argv)`.
pub fn llama_server() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut params = Params::default();
    match parse_server_args(&args, &mut params) {
        Ok(true) => {}
        Ok(false) => return 0,
        Err(e) => {
            eprintln!("error: {}", e);
            return 1;
        }
    }
    let mut role = ServerRoleType::Router;
    if std::env::var_os(crate::tools::server::common::SERVER_ENV_ROUTER_PORT).is_some() {
        role = ServerRoleType::from_str(
            &std::env::var(crate::tools::server::common::SERVER_ENV_CHILD_ROLE).unwrap_or_default(),
        );
    }
    server_run(&mut params, &args, role, &ServerRunOpts::default())
}

/// `llama_server(common_params & params, int argc, char ** argv)`.
pub fn llama_server_embedded(params: &mut Params, args: &[String]) -> i32 {
    let opts = ServerRunOpts {
        install_signal_handlers: false,
        resolve_model: false,
    };
    let role = server_role_detect(params, &ServerRoleType::Router);
    server_run(params, args, role, &opts)
}

/// `llama_server_terminate()`.
pub fn llama_server_terminate() {
    request_shutdown();
}
