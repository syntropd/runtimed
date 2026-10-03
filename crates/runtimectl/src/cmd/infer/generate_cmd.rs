//! Handler for text token generation.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `generate` command.
#[allow(clippy::too_many_arguments)]
pub async fn exec_generate(
    client: &RuntimedClient,
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    image: Option<&str>,
    backend: Option<&str>,
    as_json: bool,
) -> Result<()> {
    let mut params = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
    });
    if let Some(path) = image {
        use base64::Engine as _;
        let bytes = std::fs::read(path)?;
        params["image"] = json!(base64::engine::general_purpose::STANDARD.encode(bytes));
    }
    if let Some(b) = backend {
        params["backend"] = json!(b);
    }

    let res = client
        .call("io.syntrop.Runtime1.Generate", Some(params))
        .await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    if let Some(result) = res.get("result") {
        let text = result.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let prompt_toks = result.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let comp_toks = result.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let duration = result.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);

        println!("{}", text);
        eprintln!(
            "\n[Prompt: {} tokens | Output: {} tokens | Completed in {}ms]",
            prompt_toks, comp_toks, duration
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::test_support::fake_daemon;

    fn reply() -> serde_json::Value {
        serde_json::json!({ "parameters": {
            "result": { "text": "hi", "prompt_tokens": 2, "completion_tokens": 1, "duration_ms": 5 }
        } })
    }

    #[tokio::test]
    async fn renders_human_and_json() {
        for as_json in [false, true] {
            let (client, server) = fake_daemon(vec![reply()]);
            exec_generate(&client, "m", "p", 8, 0.0, 0, 1.0, 0, None, None, as_json)
                .await
                .expect("renders");
            server.await.expect("server done");
        }
    }

    #[tokio::test]
    async fn image_file_is_base64_attached() {
        let path = std::env::temp_dir().join(format!("runtimectl-test-{}-img.bin", std::process::id()));
        std::fs::write(&path, [1u8, 2, 3, 4]).unwrap();
        let (client, server) = fake_daemon(vec![reply()]);
        exec_generate(&client, "m", "p", 8, 0.0, 0, 1.0, 0, Some(path.to_str().unwrap()), None, true)
            .await
            .expect("renders with image");
        server.await.expect("server done");
        let _ = std::fs::remove_file(&path);
    }
}
