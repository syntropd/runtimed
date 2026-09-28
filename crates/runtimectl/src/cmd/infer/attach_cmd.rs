//! Handler for vision projector attach.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `attach-vision` command.
pub async fn exec_attach_vision(
    client: &RuntimedClient,
    model: &str,
    mmproj: &str,
    as_json: bool,
) -> Result<()> {
    let params = json!({
        "model": model,
        "mmproj": mmproj,
    });

    let res = client
        .call("io.syntrop.Runtime1.AttachVision", Some(params))
        .await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    if let Some(model) = res.get("model") {
        let name = model.get("name").and_then(|v| v.as_str()).unwrap_or("");
        println!("vision attached to {name}");
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
                "parameters": { "model": { "name": "m" } }
            })]);
            exec_attach_vision(&client, "m", "mmproj", as_json).await.expect("renders");
            server.await.expect("server done");
        }
    }
}
