//! Small self-checks for the server port.

use super::common::{
    format_error_response, json_value, safe_json_to_str, ErrorType, ServerTokens,
};
use super::http::{Handler, HttpContext, HttpRequest, HttpResponse, Method};
use super::task::{MetricsResult, TaskParams, TaskResult};
use crate::json::{Item, Json};
use std::sync::Arc;

#[test]
fn error_response_matches_protocol() {
    let error = format_error_response("bad request", ErrorType::InvalidRequest);
    assert_eq!(error.value("code", 0i64).unwrap(), 400);
    assert_eq!(error.value_str("type", "").unwrap(), "invalid_request_error");
    assert_eq!(safe_json_to_str(&error), "{\"code\":400,\"message\":\"bad request\",\"type\":\"invalid_request_error\"}");
}

#[test]
fn json_value_falls_back_on_wrong_type() {
    let body = Json::object_from(vec![Item::new("n", "not a number")]);
    assert_eq!(json_value(&body, "n", 7i32), 7);
    assert_eq!(json_value(&body, "missing", 9i32), 9);
}

#[test]
fn oai_sse_sends_one_event_per_array_item() {
    let data = Json::array_from(vec![Json::object_from(vec![Item::new("a", 1)]), Json::object_from(vec![Item::new("b", 2)])]);
    let sse = super::common::format_oai_sse(&data);
    assert_eq!(sse.matches("data: ").count(), 2);
    assert!(sse.ends_with("\n\n"));
}

#[test]
fn route_matching_fills_path_params() {
    let ctx = HttpContext::new();
    let handler: Handler = Arc::new(|req: &HttpRequest| {
        HttpResponse::ok(&Json::object_from(vec![Item::new("id", req.get_param("id_slot"))]))
    });
    ctx.post("/slots/:id_slot", handler);
    let mut req = HttpRequest::default();
    req.path = "/slots/3".to_string();
    let res = ctx.dispatch_request(Method::Post, &req);
    assert_eq!(res.status, 200);
    assert!(res.data.contains("\"id\":\"3\""));
}

#[test]
fn unknown_route_is_404() {
    let ctx = HttpContext::new();
    let req = HttpRequest { path: "/missing".to_string(), ..HttpRequest::default() };
    assert_eq!(ctx.dispatch_request(Method::Get, &req).status, 404);
}

#[test]
fn token_container_round_trip() {
    let mut tokens = ServerTokens::new(vec![1, 2, 3], false);
    tokens.push_back_media(2);
    let packed = tokens.serialize();
    let restored = ServerTokens::deserialize(&packed).unwrap();
    assert_eq!(restored.get_tokens(), tokens.get_tokens());
    assert_eq!(restored.get_text_tokens(), vec![1, 2, 3]);
}

#[test]
fn task_params_json_has_generation_fields() {
    let params = TaskParams::default();
    let json = params.to_json(true);
    assert_eq!(json.value("n_predict", 0i64).unwrap(), -1);
    assert_eq!(json.value_str("chat_format", "").unwrap(), "Content-only");
    assert!(json.contains("samplers"));
}

#[test]
fn metrics_text_has_prometheus_help() {
    let metrics = MetricsResult::default();
    let text = metrics.to_metrics();
    assert!(text.contains("# HELP llamacpp:prompt_tokens_total"));
    assert!(text.contains("# TYPE llamacpp:requests_processing gauge"));
}

#[test]
fn error_result_is_detected() {
    let result = TaskResult::Error {
        id: 1,
        kind: ErrorType::Unavailable,
        message: "no slot".to_string(),
        n_prompt_tokens: 0,
        n_ctx: 0,
    };
    assert!(result.is_error());
    assert_eq!(result.to_json().value("code", 0i64).unwrap(), 503);
}

#[test]
fn cli_parses_server_flags() {
    let mut params = crate::common::Params::default();
    let args: Vec<String> = ["--port", "8081", "--metrics", "--api-key", "a,b", "-np", "2"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert!(super::cli::parse_server_args(&args, &mut params).unwrap());
    assert_eq!(params.port, 8081);
    assert!(params.endpoint_metrics);
    assert_eq!(params.api_keys, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(params.n_parallel, 2);
}
