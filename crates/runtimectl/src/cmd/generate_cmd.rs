//! Handler for text token generation.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `generate` command.
pub async fn exec_generate(
    client: &RuntimedClient,
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    as_json: bool,
) -> Result<()> {
    let params = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    });

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
