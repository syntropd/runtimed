//! Varlink wire protocol framing for runtimed.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An incoming Varlink method invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarlinkCall {
    /// Fully qualified interface method name (e.g. "io.syntrop.Runtime1.Generate").
    pub method: String,
    /// Optional parameter payload dictionary.
    pub parameters: Option<Value>,
    /// Whether this call requests streaming responses.
    #[serde(default)]
    pub more: Option<bool>,
}

/// An outgoing Varlink reply envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarlinkReply {
    /// Return parameters on successful invocation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    /// Error identifier on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Whether additional response messages follow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continues: Option<bool>,
}

impl VarlinkReply {
    /// Creates a successful method reply.
    pub fn ok(parameters: Value) -> Self {
        Self {
            parameters: Some(parameters),
            error: None,
            continues: None,
        }
    }

    /// Creates an error reply with an optional diagnostic payload.
    pub fn err(error: &str, parameters: Option<Value>) -> Self {
        Self {
            parameters,
            error: Some(error.to_string()),
            continues: None,
        }
    }

    /// Serializes the reply to NUL-terminated JSON bytes for the wire.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(self).unwrap_or_else(|_| b"{}".to_vec());
        bytes.push(0x00);
        bytes
    }
}
