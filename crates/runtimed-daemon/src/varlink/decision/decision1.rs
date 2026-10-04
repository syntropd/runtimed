//! Handler implementation for io.syntrop.Decision1 Varlink interface.

use super::systemone::{
    build_systemone_questions, call_systemone_endpoint, normalize_systemone_answers,
};
use crate::varlink::server::protocol::VarlinkReply;
use serde_json::{json, Value};

/// Default decision model when none specified.
pub const DEFAULT_DECISION_MODEL: &str = "clef-flash";

/// Shared handler for io.syntrop.Decision1 method dispatches.
#[derive(Clone, Default)]
pub struct Decision1Handler;

impl Decision1Handler {
    pub fn new() -> Self {
        Self
    }

    /// Dispatches incoming io.syntrop.Decision1 method calls.
    pub async fn handle_call(&self, method: &str, params: Option<&Value>) -> Option<VarlinkReply> {
        match method {
            "io.syntrop.Decision1.Decide" => Some(self.handle_decide(params).await),
            _ => None,
        }
    }

    /// Handles io.syntrop.Decision1.Decide call.
    pub async fn handle_decide(&self, params: Option<&Value>) -> VarlinkReply {
        let params = match params {
            Some(p) => p,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Decision1.InvalidParameter",
                    Some(json!({ "parameter": "parameters" })),
                );
            }
        };

        let state = match params.get("state").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Decision1.InvalidParameter",
                    Some(json!({ "parameter": "state" })),
                );
            }
        };

        let questions = match params.get("questions").and_then(|v| v.as_object()) {
            Some(q) => q,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Decision1.InvalidParameter",
                    Some(json!({ "parameter": "questions" })),
                );
            }
        };

        let model = params
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_DECISION_MODEL);

        let s1_questions = build_systemone_questions(questions);
        let s1_request = json!({
            "model": model,
            "state": state,
            "questions": s1_questions
        });

        match call_systemone_endpoint(&s1_request).await {
            Ok(s1_response) => {
                let model_resp = s1_response
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or(model);
                let raw_answers = s1_response.get("answers").and_then(|v| v.as_object());
                let normalized_answers = normalize_systemone_answers(questions, raw_answers);

                VarlinkReply::ok(json!({
                    "model": model_resp,
                    "answers": normalized_answers
                }))
            }
            Err(e) => VarlinkReply::err(
                "io.syntrop.Decision1.DecisionFailed",
                Some(json!({ "reason": e })),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_decision1_handler_method_mismatch() {
        let handler = Decision1Handler::new();
        assert!(handler.handle_call("unknown.Method", None).await.is_none());
    }
}
