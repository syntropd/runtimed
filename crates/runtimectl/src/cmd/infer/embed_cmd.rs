//! Handler for vector embedding generation.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `embed` command.
pub async fn exec_embed(
    client: &RuntimedClient,
    model: &str,
    text: &str,
    as_json: bool,
) -> Result<()> {
    let params = json!({
        "model": model,
        "text": text,
    });

    let res = client.call("io.syntrop.Runtime1.Embed", Some(params)).await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    if let Some(embedding) = res.get("embedding").and_then(|v| v.as_array()) {
        println!("Embedding Dimensions: {}", embedding.len());
        let sample: Vec<String> = embedding
            .iter()
            .take(8)
            .filter_map(|v| v.as_f64().map(|f| format!("{:.4}", f)))
            .collect();
        println!("Vector Sample (first 8): [{}]", sample.join(", "));
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
                "parameters": { "embedding": [0.1, 0.2, 0.3] }
            })]);
            exec_embed(&client, "m", "t", as_json).await.expect("renders");
            server.await.expect("server done");
        }
    }
}
