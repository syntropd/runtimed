//! HTTP client and schema adapters for Ollama /v1/systemone endpoint.

use serde_json::{json, Map, Value};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Transform generic questions map into Ollama /v1/systemone schema.
pub fn build_systemone_questions(questions: &Map<String, Value>) -> Value {
    let mut out = Map::new();
    for (k, q) in questions {
        let q_obj = match q.as_object() {
            Some(o) => o,
            None => continue,
        };
        let q_type = q_obj.get("type").and_then(|v| v.as_str()).unwrap_or("bool");
        let instructions = q_obj
            .get("instructions")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let mut item = Map::new();
        item.insert("instructions".to_string(), json!(instructions));

        match q_type {
            "bool" | "noul" => {
                item.insert("type".to_string(), json!("noul"));
            }
            "choice" => {
                item.insert("type".to_string(), json!("choice"));
                if let Some(crit) = q_obj.get("criteria").filter(|c| c.is_object()) {
                    item.insert("criteria".to_string(), crit.clone());
                } else if let Some(choices) = q_obj.get("choices").and_then(|v| v.as_array()) {
                    let mut crit_map = Map::new();
                    for c in choices {
                        if let Some(c_str) = c.as_str() {
                            crit_map.insert(c_str.to_string(), json!(c_str));
                        }
                    }
                    item.insert("criteria".to_string(), Value::Object(crit_map));
                }
            }
            "score" => {
                item.insert("type".to_string(), json!("score"));
                if let Some(crit) = q_obj.get("criteria").filter(|c| c.is_array()) {
                    item.insert("criteria".to_string(), crit.clone());
                } else {
                    item.insert(
                        "criteria".to_string(),
                        json!([
                            "Level 0 (Minimal / None)",
                            "Level 1 (Low / Minor)",
                            "Level 2 (Moderate / Average)",
                            "Level 3 (High / Severe)",
                            "Level 4 (Critical / Maximum)"
                        ]),
                    );
                }
            }
            other => {
                item.insert("type".to_string(), json!(other));
            }
        }
        out.insert(k.clone(), Value::Object(item));
    }
    Value::Object(out)
}

/// Normalize Ollama /v1/systemone answers into deterministic typed format.
pub fn normalize_systemone_answers(
    questions: &Map<String, Value>,
    raw_answers: Option<&Map<String, Value>>,
) -> Value {
    let mut out = Map::new();
    let empty_map = Map::new();
    let raw = raw_answers.unwrap_or(&empty_map);

    for (k, q) in questions {
        let q_obj = q.as_object();
        let orig_type = q_obj
            .and_then(|o| o.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("bool");

        if let Some(ans) = raw.get(k).and_then(|v| v.as_object()) {
            let mut item = Map::new();
            if orig_type == "bool" {
                item.insert("type".to_string(), json!("bool"));
                let noul = ans.get("noul").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b_val = noul >= 0.5;
                item.insert("bool".to_string(), json!(b_val));
                let conf = (noul - 0.5).abs() * 2.0;
                item.insert("confidence".to_string(), json!(conf));
                item.insert(
                    "probabilities".to_string(),
                    json!({ "true": noul, "false": 1.0 - noul }),
                );
                item.insert("score".to_string(), json!(noul));
            } else {
                for (prop_k, prop_v) in ans {
                    item.insert(prop_k.clone(), prop_v.clone());
                }
            }
            out.insert(k.clone(), Value::Object(item));
        }
    }
    Value::Object(out)
}

/// Send request to Ollama /v1/systemone HTTP endpoint.
pub async fn call_systemone_endpoint(payload: &Value) -> Result<Value, String> {
    let connect_fut = TcpStream::connect("127.0.0.1:11434");
    let mut stream = timeout(Duration::from_secs(30), connect_fut)
        .await
        .map_err(|_| "systemone provider connect timed out".to_string())?
        .map_err(|e| format!("systemone connect failed: {e}"))?;

    let json_bytes = serde_json::to_vec(payload).map_err(|e| e.to_string())?;
    let req = format!(
        "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1:11434\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        json_bytes.len()
    );

    stream
        .write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream.write_all(&json_bytes).await.map_err(|e| e.to_string())?;
    stream.flush().await.map_err(|e| e.to_string())?;

    let mut response_bytes = Vec::new();
    let read_fut = stream.read_to_end(&mut response_bytes);
    timeout(Duration::from_secs(180), read_fut)
        .await
        .map_err(|_| "systemone provider read timed out".to_string())?
        .map_err(|e| format!("systemone read failed: {e}"))?;

    let response_str = String::from_utf8_lossy(&response_bytes);
    parse_systemone_http_response(&response_str)
}

/// Parse HTTP response body for /v1/systemone with chunk framing protection.
pub fn parse_systemone_http_response(raw_resp: &str) -> Result<Value, String> {
    if raw_resp.is_empty() {
        return Err("systemone provider returned empty response".to_string());
    }
    let body = raw_resp
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(raw_resp);

    let json_slice = match (body.find('{'), body.rfind('}')) {
        (Some(start), Some(end)) if start <= end => &body[start..=end],
        _ => body.trim(),
    };

    let parsed: Value = serde_json::from_str(json_slice)
        .map_err(|e| format!("parse response JSON failed: {e}"))?;

    if let Some(err) = parsed.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_systemone_questions_bool() {
        let mut questions = Map::new();
        questions.insert(
            "is_down".to_string(),
            json!({
                "type": "bool",
                "instructions": "Is service down?"
            }),
        );
        let s1 = build_systemone_questions(&questions);
        let q_down = s1.pointer("/is_down").unwrap();
        assert_eq!(q_down["type"], "noul");
        assert_eq!(q_down["instructions"], "Is service down?");
    }

    #[test]
    fn test_build_systemone_questions_choice() {
        let mut questions = Map::new();
        questions.insert(
            "category".to_string(),
            json!({
                "type": "choice",
                "instructions": "Pick category",
                "choices": ["network", "disk"]
            }),
        );
        let s1 = build_systemone_questions(&questions);
        let q_cat = s1.pointer("/category").unwrap();
        assert_eq!(q_cat["type"], "choice");
        assert_eq!(q_cat["criteria"]["network"], "network");
        assert_eq!(q_cat["criteria"]["disk"], "disk");
    }

    #[test]
    fn test_normalize_systemone_answers_bool() {
        let mut questions = Map::new();
        questions.insert("critical".to_string(), json!({ "type": "bool" }));
        let mut raw = Map::new();
        raw.insert("critical".to_string(), json!({ "type": "noul", "noul": 0.95 }));

        let normalized = normalize_systemone_answers(&questions, Some(&raw));
        let ans = normalized.pointer("/critical").unwrap();
        assert_eq!(ans["type"], "bool");
        assert_eq!(ans["bool"], true);
        assert!((ans["confidence"].as_f64().unwrap() - 0.9).abs() < 1e-4);
    }

    #[test]
    fn test_parse_systemone_http_response_chunked() {
        let chunked = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1f4\r\n{\"model\":\"clef-flash\",\"answers\":{\"ready\":{\"type\":\"noul\",\"noul\":0.99}}}\r\n0\r\n\r\n";
        let res = parse_systemone_http_response(chunked).unwrap();
        assert_eq!(res["model"], "clef-flash");
        assert_eq!(res["answers"]["ready"]["type"], "noul");
    }

    #[test]
    fn test_parse_systemone_http_response_error() {
        let err_resp = "HTTP/1.1 400 Bad Request\r\n\r\n{\"error\":\"invalid question schema\"}";
        let err = parse_systemone_http_response(err_resp).unwrap_err();
        assert_eq!(err, "invalid question schema");
    }
}
