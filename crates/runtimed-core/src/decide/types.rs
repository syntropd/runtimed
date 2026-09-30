//! Data types for System One fast candidate scoring and calibrated decisions.

use serde::{Deserialize, Serialize};

/// Candidate option for single-token classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    #[serde(alias = "id")]
    pub name: String,
    pub token: String,
}

/// Request to evaluate candidate choices given a loaded model and prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecideRequest {
    pub model: String,
    pub prompt: String,
    pub candidates: Vec<Candidate>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_temperature() -> f32 {
    1.0
}

/// A candidate scored by the decision evaluator with probability and logit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredCandidate {
    pub name: String,
    pub token: String,
    pub token_id: u32,
    pub logit: f32,
    pub probability: f32,
}

/// Calibrated decision result with confidence score S in [0.0, 1.0].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResult {
    pub winner: String,
    pub confidence: f32,
    pub raw_probability: f32,
    pub margin: f32,
    pub entropy: f32,
    pub candidates: Vec<ScoredCandidate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_serde_roundtrip() {
        let cand = Candidate {
            name: "TransientRestart".into(),
            token: "A".into(),
        };
        let json = serde_json::to_string(&cand).unwrap();
        let parsed: Candidate = serde_json::from_str(&json).unwrap();
        assert_eq!(cand, parsed);

        // Also test alias 'id'
        let from_id: Candidate =
            serde_json::from_str(r#"{"id":"ConfigDrift","token":"B"}"#).unwrap();
        assert_eq!(from_id.name, "ConfigDrift");
        assert_eq!(from_id.token, "B");
    }

    #[test]
    fn test_decide_request_default_temp() {
        let req_json = r#"{"model":"m","prompt":"p","candidates":[]}"#;
        let req: DecideRequest = serde_json::from_str(req_json).unwrap();
        assert_eq!(req.temperature, 1.0);
    }
}
