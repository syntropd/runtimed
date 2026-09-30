//! System One fast candidate scoring and decision calibration.

pub mod calibrate;
pub mod evaluate;
pub mod types;

pub use calibrate::calibrated_confidence;
pub use calibrate::decision_margin;
pub use calibrate::normalized_entropy;
pub use calibrate::softmax_with_temperature;
pub use calibrate::ALPHA;
pub use calibrate::BETA;
pub use evaluate::evaluate_decision;
pub use evaluate::resolve_candidate_token;
pub use types::Candidate;
pub use types::DecideRequest;
pub use types::DecisionResult;
pub use types::ScoredCandidate;
