//! Handler for LoRA adapter attach.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `attach-lora` command.
pub async fn exec_attach_lora(
    client: &RuntimedClient,
    model: &str,
    lora: &str,
    as_json: bool,
) -> Result<()> {
    let params = json!({
        "model": model,
        "lora": lora,
    });

    let res = client
        .call("io.syntrop.Runtime1.AttachLora", Some(params))
        .await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    if let Some(fused) = res.get("fused_tensors").and_then(|v| v.as_array()) {
        println!("fused {} LoRA tensors into {model}", fused.len());
    }

    Ok(())
}
