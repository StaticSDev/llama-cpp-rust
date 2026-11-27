//! Rust port of llama.cpp's `tools/server` HTTP server.
//!
//! The port covers request handling, route registration, server state, the
//! JSON protocol and the server CLI. HTTP transport uses a std-only
//! `TcpListener` abstraction. Inference is supplied by the host through the
//! task queue and tokenizer hooks; this layer never uses FFI or unsafe code.

pub mod cli;
pub mod common;
pub mod context;
pub mod http;
pub mod queue;
pub mod routes;
pub mod task;

#[cfg(test)]
mod tests;

pub use cli::{
    llama_server, llama_server_embedded, llama_server_terminate, parse_server_args, request_shutdown, server_prepare_params,
    server_role_detect, server_run, should_shutdown, usage, ServerRunOpts,
};
pub use common::{
    format_error_response, gen_chatcmplid, gen_tool_call_id, random_string, safe_json_to_str, server_addr_is_public,
    server_model_architecture_json, server_model_output_modalities, ErrorType, Metrics, MetricsBucket, ServerChatParams,
    ServerPipe, ServerRole, ServerRoleType, ServerState, ServerTokens, SlotStats,
};
pub use context::{ServerContext, ServerContextMeta, ServerSlot, StateCallback};
pub use http::{
    encode_query_component, static_file, Handler, HttpContext, HttpRequest, HttpResponse, Method, RawBuffer, SseFraming,
    UploadedFile, WsConn, WsHandler,
};
pub use queue::{BatchResponse, ResponseReader, ServerQueue, ServerResponse};
pub use routes::{register_routes, register_stream_routes, setup_features, ServerRoutes};
pub use task::{
    CompletionResult, CompletionTokenOutput, GrammarTrigger, MetricsResult, PartialResult, ResultPromptProgress,
    ServerControlAction, ServerTask, ServerTaskType, StopType, TaskParams, TaskResponseType, TaskResult,
};
