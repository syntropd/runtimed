//! Handler for listing all active loaded models.

use crate::client::RuntimedClient;
use anyhow::Result;

/// Executes the `list` command.
pub async fn exec_list(client: &RuntimedClient, as_json: bool) -> Result<()> {
    let res = client
        .call("io.syntrop.Runtime1.ListLoadedModels", None)
        .await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    let models = res
        .get("models")
        .and_then(|m| m.as_array())
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    if models.is_empty() {
        println!("No models currently loaded in memory.");
        return Ok(());
    }

    println!("{:<24} {:<12} {:<12} {:<10} {}", "MODEL", "BACKEND", "MEMORY", "PARAMS", "CONTEXT");
    println!("{}", "-".repeat(75));

    for m in models {
        let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("-");
        let backend = m.get("compute_backend").and_then(|v| v.as_str()).unwrap_or("-");
        let mem = m
            .get("memory_bytes")
            .and_then(|v| v.as_u64())
            .map(|b| format!("{:.1}G", (b as f64) / (1024.0 * 1024.0 * 1024.0)))
            .unwrap_or_else(|| "-".into());
        let params = m
            .get("parameter_count")
            .and_then(|v| v.as_u64())
            .map(|p| format!("{}B", p / 1_000_000_000))
            .unwrap_or_else(|| "-".into());
        let ctx = m.get("context_window").and_then(|v| v.as_u64()).unwrap_or(0);

        println!("{:<24} {:<12} {:<12} {:<10} {}", name, backend, mem, params, ctx);
    }

    Ok(())
}
