//! Local provider bridge dispatching to Ollama for non-standard architectures.

use runtimed_core::engine::GenerationResult;
use runtimed_core::error::RuntimedError;
use serde_json::Value;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Parse raw HTTP response body into GenerationResult.
pub fn parse_http_ollama_response(raw_body: &str) -> Result<GenerationResult, RuntimedError> {
    let json_slice = match (raw_body.find('{'), raw_body.rfind('}')) {
        (Some(start), Some(end)) if start <= end => &raw_body[start..=end],
        _ => raw_body.trim(),
    };
    let parsed: Value = serde_json::from_str(json_slice).map_err(|e| {
        RuntimedError::GenerationFailed(format!("failed to parse provider JSON: {e}"))
    })?;

    if let Some(err) = parsed.get("error").and_then(|v| v.as_str()) {
        return Err(RuntimedError::GenerationFailed(err.to_string()));
    }

    let response = parsed.get("response").and_then(|v| v.as_str()).unwrap_or("");
    let thinking = parsed.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
    let full_text = if !thinking.trim().is_empty() {
        format!("<think>\n{}\n</think>\n{}", thinking.trim(), response.trim())
    } else {
        response.to_string()
    };

    let p_toks = parsed.get("prompt_eval_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let c_toks = parsed.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let dur_ns = parsed.get("total_duration").and_then(|v| v.as_u64()).unwrap_or(0);
    let duration_ms = (dur_ns / 1_000_000).max(1);

    Ok(GenerationResult {
        text: full_text,
        prompt_tokens: p_toks,
        completion_tokens: c_toks,
        finish_reason: "stop".to_string(),
        duration_ms,
    })
}

/// Dispatches a generation request to the local provider daemon.
pub async fn dispatch_ollama(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    draft_model: Option<&str>,
) -> Result<GenerationResult, RuntimedError> {
    let connect_future = TcpStream::connect("127.0.0.1:11434");
    let mut stream = timeout(Duration::from_secs(5), connect_future)
        .await
        .map_err(|_| RuntimedError::GenerationFailed("provider daemon connect timed out".into()))?
        .map_err(|e| RuntimedError::GenerationFailed(format!("connect to provider failed: {e}")))?;

    let mut payload = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    });
    if let Some(draft) = draft_model {
        payload["draft_model"] = serde_json::json!(draft);
    }

    let body_str = serde_json::to_string(&payload)
        .map_err(|e| RuntimedError::GenerationFailed(format!("JSON encode failed: {e}")))?;

    let http_req = format!(
        "POST /api/generate HTTP/1.1\r\nHost: 127.0.0.1:11434\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body_str.len(),
        body_str
    );

    stream
        .write_all(http_req.as_bytes())
        .await
        .map_err(|e| RuntimedError::GenerationFailed(format!("HTTP write failed: {e}")))?;
    let _ = stream.flush().await;

    let mut raw_resp = Vec::with_capacity(8192);
    let read_future = stream.read_to_end(&mut raw_resp);
    timeout(Duration::from_secs(180), read_future)
        .await
        .map_err(|_| RuntimedError::GenerationFailed("provider response timed out".into()))?
        .map_err(|e| RuntimedError::GenerationFailed(format!("HTTP read failed: {e}")))?;

    let resp_str = String::from_utf8_lossy(&raw_resp);
    if resp_str.is_empty() {
        return Err(RuntimedError::GenerationFailed("provider daemon returned empty response (EOF)".into()));
    }
    let body_part = match resp_str.split_once("\r\n\r\n") {
        Some((_header, body)) => body,
        None => resp_str.as_ref(),
    };

    parse_http_ollama_response(body_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_http_ollama_response_with_thinking() {
        let json_str = r#"{"model":"qwen3.5:9b","response":"4","thinking":"Let me think: 2+2 is 4","prompt_eval_count":12,"eval_count":15,"total_duration":50000000}"#;
        let res = parse_http_ollama_response(json_str).unwrap();
        assert_eq!(res.prompt_tokens, 12);
        assert_eq!(res.completion_tokens, 15);
        assert_eq!(res.duration_ms, 50);
        assert!(res.text.contains("<think>"));
        assert!(res.text.contains("Let me think: 2+2 is 4"));
        assert!(res.text.ends_with("4"));
    }

    #[test]
    fn test_parse_http_ollama_response_without_thinking() {
        let json_str = r#"{"model":"granite","response":"Hello world","prompt_eval_count":5,"eval_count":2,"total_duration":20000000}"#;
        let res = parse_http_ollama_response(json_str).unwrap();
        assert_eq!(res.text, "Hello world");
        assert_eq!(res.prompt_tokens, 5);
        assert_eq!(res.completion_tokens, 2);
        assert_eq!(res.duration_ms, 20);
    }

    #[test]
    fn test_parse_http_ollama_response_error() {
        let json_str = r#"{"error":"model not found"}"#;
        let err = parse_http_ollama_response(json_str).unwrap_err();
        match err {
            RuntimedError::GenerationFailed(msg) => assert_eq!(msg, "model not found"),
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_parse_http_ollama_response_chunked() {
        let chunked = "1f4\r\n{\"model\":\"qwen3.5:9b\",\"response\":\"1591\",\"prompt_eval_count\":10,\"eval_count\":2,\"total_duration\":1000000}\r\n0\r\n\r\n";
        let res = parse_http_ollama_response(chunked).unwrap();
        assert_eq!(res.text, "1591");
        assert_eq!(res.prompt_tokens, 10);
        assert_eq!(res.completion_tokens, 2);
    }
}
