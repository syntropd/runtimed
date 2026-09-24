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
        println!("Parameters:   {} B", params_cnt / 1_000_000_000);
        println!("Memory:       {:.2} GiB", (mem as f64) / (1024.0 * 1024.0 * 1024.0));
        println!("Context:      {} tokens", ctx);
        println!("Backend:      {}", backend);
    }

    Ok(())
}
