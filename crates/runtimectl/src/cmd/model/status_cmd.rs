//! Handler for querying loaded model status.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `status` command.
pub async fn exec_status(client: &RuntimedClient, model: &str, as_json: bool) -> Result<()> {
    let params = json!({
        "model": model,
    });

    let res = client
        .call("io.syntrop.Runtime1.GetModelStatus", Some(params))
        .await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    let status = res.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
    println!("Model:  {}", model);
    println!("Status: {}", status);

    if let Some(m) = res.get("model").filter(|v| !v.is_null()) {
        let arch = m.get("architecture").and_then(|v| v.as_str()).unwrap_or("-");
        let params_cnt = m.get("parameter_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let mem = m.get("memory_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
        let ctx = m.get("context_window").and_then(|v| v.as_u64()).unwrap_or(0);
        let backend = m.get("compute_backend").and_then(|v| v.as_str()).unwrap_or("-");

        println!("Architecture: {}", arch);
        println!("Parameters:   {}", crate::cmd::human_count(params_cnt));
        println!("Memory:       {:.2} GiB", (mem as f64) / (1024.0 * 1024.0 * 1024.0));
        println!("Context:      {} tokens", ctx);
        println!("Backend:      {}", backend);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::test_support::fake_daemon;

    #[tokio::test]
    async fn renders_loaded_not_loaded_human_and_json() {
        let payloads = [
            serde_json::json!({ "parameters": {
                "status": "loaded",
                "model": {
                    "architecture": "qwen2",
                    "parameter_count": 4600000000u64,
                    "memory_bytes": 2147483648u64,
                    "context_window": 8192u64,
                    "compute_backend": "cpu"
                }
            } }),
            serde_json::json!({ "parameters": { "status": "not_loaded", "model": null } }),
        ];
        for payload in payloads {
            for as_json in [false, true] {
                let (client, server) = fake_daemon(vec![payload.clone()]);
                exec_status(&client, "m", as_json).await.expect("renders");
                server.await.expect("server done");
            }
        }
    }
}
