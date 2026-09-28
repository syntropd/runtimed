//! Handler for daemon load reporting (fleet overflow signal).

use crate::client::RuntimedClient;
use anyhow::Result;

/// Executes the `load` command.
pub async fn exec_load(client: &RuntimedClient, as_json: bool) -> Result<()> {
    let res = client.call("io.syntrop.Runtime1.GetLoad", None).await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    let avail = res.get("available_slots").and_then(|v| v.as_u64()).unwrap_or(0);
    let max = res.get("max_slots").and_then(|v| v.as_u64()).unwrap_or(0);
    let used = res.get("used_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
    println!("slots: {avail}/{max} free, resident: {used} bytes");
    if let Some(models) = res.get("models").and_then(|v| v.as_array()) {
        for m in models {
            let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let mem = m.get("memory_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("  {name} ({mem} bytes)");
        }
    }

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
                "parameters": {
                    "available_slots": 3u64,
                    "max_slots": 4u64,
                    "used_bytes": 100u64,
                    "models": [{ "name": "m", "memory_bytes": 100u64 }]
                }
            })]);
            exec_load(&client, as_json).await.expect("renders");
            server.await.expect("server done");
        }
    }
}
