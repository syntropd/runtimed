//! Handler for daemon vendor information.

use crate::client::RuntimedClient;
use anyhow::Result;
use serde_json::json;

/// Executes the `info` command.
pub async fn exec_info(client: &RuntimedClient, as_json: bool) -> Result<()> {
    let info = client.call("org.varlink.service.GetInfo", None).await?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&info)?);
        return Ok(());
    }

    let vendor = info.get("vendor").and_then(|v| v.as_str()).unwrap_or("unknown");
    let product = info.get("product").and_then(|p| p.as_str()).unwrap_or("unknown");
    let version = info.get("version").and_then(|v| v.as_str()).unwrap_or("unknown");
    let url = info.get("url").and_then(|u| u.as_str()).unwrap_or("unknown");

    println!("Runtimed Daemon Info:");
    println!("  Product: {} ({})", product, vendor);
    println!("  Version: {}", version);
    println!("  URL:     {}", url);

    if let Some(ifaces) = info.get("interfaces").and_then(|i| i.as_array()) {
        println!("\nSupported Varlink Interfaces:");
        for iface in ifaces {
            if let Some(name) = iface.as_str() {
                println!("  - {}", name);
                let desc_res = client
                    .call(
                        "org.varlink.service.GetInterfaceDescription",
                        Some(json!({ "interface": name })),
                    )
                    .await;
                if let Ok(desc_val) = desc_res {
                    if let Some(desc) = desc_val.get("description").and_then(|d| d.as_str()) {
                        for line in desc.lines() {
                            println!("      {}", line);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::test_support::fake_daemon;

    fn info() -> serde_json::Value {
        serde_json::json!({ "parameters": {
            "vendor": "v",
            "product": "p",
            "version": "1",
            "url": "u",
            "interfaces": ["io.syntrop.Runtime1"]
        } })
    }

    #[tokio::test]
    async fn renders_json_with_single_call() {
        let (client, server) = fake_daemon(vec![info()]);
        exec_info(&client, true).await.expect("renders");
        server.await.expect("server done");
    }

    #[tokio::test]
    async fn renders_human_with_interface_descriptions() {
        let desc = serde_json::json!({ "parameters": { "description": "line1\nline2" } });
        let (client, server) = fake_daemon(vec![info(), desc]);
        exec_info(&client, false).await.expect("renders");
        server.await.expect("server done");
    }
}
