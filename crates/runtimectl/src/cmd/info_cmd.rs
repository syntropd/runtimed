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
