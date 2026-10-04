//! Varlink interface definition for io.syntrop.Decision1.

/// Varlink interface definition text for io.syntrop.Decision1.
pub const IO_SYNTROP_DECISION1_INTERFACE: &str = r#"
interface io.syntrop.Decision1

type Question (
  type: string,
  instructions: string,
  choices: ?[]string,
  criteria: ?[string]string
)

type Answer (
  type: string,
  bool: ?bool,
  choice: ?string,
  score: ?float,
  confidence: float,
  probabilities: ?[string]float
)

method Decide(model: ?string, state: string, questions: [string]Question) -> (model: string, answers: [string]Answer)

error DecisionFailed(reason: string)
error InvalidParameter(parameter: string)
error ModelNotFound(model: string)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision1_interface_declares_methods() {
        assert!(IO_SYNTROP_DECISION1_INTERFACE.contains("interface io.syntrop.Decision1"));
        assert!(IO_SYNTROP_DECISION1_INTERFACE.contains("method Decide"));
        assert!(IO_SYNTROP_DECISION1_INTERFACE.contains("type Question"));
        assert!(IO_SYNTROP_DECISION1_INTERFACE.contains("type Answer"));
    }
}
