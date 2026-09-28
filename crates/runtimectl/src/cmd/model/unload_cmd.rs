//! Handler for unloading models from memory.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `unload` command.
pub async fn exec_unload(client: &RuntimedClient, model: &str, as_json: bool) -> Result<()> {
    let params = json!({
        "model": model,
    });

    let res = client
        .call("io.syntrop.Runtime1.UnloadModel", Some(params))
        .await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    let freed = res.get("freed_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
    println!(
        "Successfully unloaded model '{}' (freed {:.2} GiB).",
        model,
        (freed as f64) / (1024.0 * 1024.0 * 1024.0)
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::test_support::fake_daemon;

    #[tokio::test]
    async fn renders_human_and_json() {
        for as_json in [false, true] {
            let (client, server) = fake_daemon(vec![serde_json::json!({
                "parameters": { "freed_bytes": 1024u64 }
            })]);
            exec_unload(&client, "m", as_json).await.expect("renders");
            server.await.expect("server done");
        }
    }
}
