//! Calibration and confidence scoring for System One decisions.

pub const ALPHA: f32 = 0.20;
pub const BETA: f32 = 0.25;

/// Numerically stable softmax with temperature scaling.
pub fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = if temperature > 0.0 { temperature } else { 1.0 };
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_logit) / temp).exp())
        .collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Normalized Shannon entropy: H_norm = (-sum(p_i * ln(p_i))) / ln(K).
pub fn normalized_entropy(probs: &[f32]) -> f32 {
    let k = probs.len();
    if k <= 1 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &p in probs {
        if p > 1e-12 {
            h -= p * p.ln();
        }
    }
    let h_max = (k as f32).ln();
    if h_max <= 0.0 {
        return 0.0;
    }
    (h / h_max).clamp(0.0, 1.0)
}

/// Decision margin Delta = p_(1) - p_(2).
pub fn decision_margin(probs: &[f32]) -> (f32, f32) {
    if probs.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let p_top = sorted[0];
    let p_second = if sorted.len() > 1 { sorted[1] } else { 0.0 };
    let delta = p_top - p_second;
    (p_top, delta)
}

/// Calibrated confidence score:
/// S = P_top * [1 - alpha * (1 - Delta / P_top)] * [1 - beta * H_norm^2]
pub fn calibrated_confidence(p_top: f32, delta: f32, h_norm: f32) -> f32 {
    if p_top <= 0.0 {
        return 0.0;
    }
    let ratio = (delta / p_top).clamp(0.0, 1.0);
    let margin_penalty = 1.0 - ALPHA * (1.0 - ratio);
    let entropy_penalty = 1.0 - BETA * (h_norm * h_norm);
    let s = p_top * margin_penalty * entropy_penalty;
    s.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_temperature_unity() {
        let logits = vec![2.0, 1.0, 0.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
    }

    #[test]
    fn test_normalized_entropy_extremes() {
        // Uniform distribution: max entropy -> 1.0
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h = normalized_entropy(&uniform);
        assert!((h - 1.0).abs() < 1e-4);

        // Degenerate distribution: zero entropy -> 0.0
        let one_hot = vec![1.0, 0.0, 0.0, 0.0];
        let h_zero = normalized_entropy(&one_hot);
        assert!(h_zero < 1e-5);
    }

    #[test]
    fn test_decision_margin_and_confidence() {
        // Absolute certainty
        let (p_top, delta) = decision_margin(&[1.0, 0.0]);
        assert_eq!(p_top, 1.0);
        assert_eq!(delta, 1.0);
        let conf = calibrated_confidence(p_top, delta, 0.0);
        assert!((conf - 1.0).abs() < 1e-5);

        // Near tie (50/50 split)
        let (p_top_tie, delta_tie) = decision_margin(&[0.5, 0.5]);
        assert_eq!(p_top_tie, 0.5);
        assert_eq!(delta_tie, 0.0);
        let h_tie = normalized_entropy(&[0.5, 0.5]);
        let conf_tie = calibrated_confidence(p_top_tie, delta_tie, h_tie);
        // S = 0.5 * (1 - 0.20) * (1 - 0.25) = 0.5 * 0.8 * 0.75 = 0.30
        assert!((conf_tie - 0.30).abs() < 1e-4);
    }
}
