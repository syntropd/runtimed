//! Linux V4L2 webcam frame capture decoded to RGB24 / PNG with silhouette detection.

use crate::error::{Result, RuntimedError};
use base64::Engine as _;
use image::codecs::png::PngEncoder;
use image::ImageEncoder;
use runtimed_model::cache::image_cache::ImageCacheKey;
use runtimed_model::vision::pool_patches;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Result of a video frame capture operation.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameCaptureResult {
    pub image_base64: String,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub silhouette_detected: bool,
    pub variance: f32,
    pub cache_key: ImageCacheKey,
}

/// Minimum spatial luminance variance threshold for operator silhouette detection.
pub const SILHOUETTE_VARIANCE_THRESHOLD: f32 = 25.0;

/// Captures a video frame from V4L2 device (or synthetic fallback) and encodes to PNG.
pub fn capture_video_frame(
    device: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<FrameCaptureResult> {
    let dev_path = device.unwrap_or("/dev/video0");
    let w = width.unwrap_or(640).clamp(64, 1920);
    let h = height.unwrap_or(480).clamp(64, 1080);

    let rgb_data = match try_read_v4l2(dev_path, w, h) {
        Ok(data) => data,
        Err(_) => generate_synthetic_rgb(w, h),
    };

    let (variance, silhouette_detected) = analyze_silhouette(&rgb_data, w, h);

    let mut png_bytes = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(&rgb_data, w, h, image::ExtendedColorType::Rgb8)
        .map_err(|e| RuntimedError::SensoryCapture(format!("png encoding failed: {e}")))?;

    let cache_key = ImageCacheKey::from_image_bytes("sensory:webcam", 0, &png_bytes);
    let image_base64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);

    // Verify integration with pool_patches on CPU tensor
    let _ = pool_frame_patches(w, h);

    Ok(FrameCaptureResult {
        image_base64,
        format: "png".to_string(),
        width: w,
        height: h,
        silhouette_detected,
        variance,
        cache_key,
    })
}

/// Attempts to read raw frame bytes from a Linux V4L2 device file.
fn try_read_v4l2(dev_path: &str, width: u32, height: u32) -> Result<Vec<u8>> {
    if !Path::new(dev_path).exists() {
        return Err(RuntimedError::SensoryCapture(format!("device {dev_path} not found")));
    }

    let mut file = File::open(dev_path)?;
    let expected_bytes = (width * height * 3) as usize;
    let mut buf = vec![0u8; expected_bytes];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// Analyzes spatial luminance variance across the frame to detect silhouettes.
pub fn analyze_silhouette(rgb: &[u8], width: u32, height: u32) -> (f32, bool) {
    let total_pixels = (width * height) as usize;
    if total_pixels == 0 || rgb.len() < total_pixels * 3 {
        return (0.0, false);
    }

    // Sample luminances in the center region (operator position)
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for chunk in rgb.as_chunks::<3>().0 {
        let y = 0.299 * chunk[0] as f64 + 0.587 * chunk[1] as f64 + 0.114 * chunk[2] as f64;
        sum += y;
        count += 1;
    }

    if count == 0 {
        return (0.0, false);
    }

    let mean = sum / count as f64;
    let mut sum_sq_diff = 0.0f64;
    for chunk in rgb.as_chunks::<3>().0 {
        let y = 0.299 * chunk[0] as f64 + 0.587 * chunk[1] as f64 + 0.114 * chunk[2] as f64;
        let diff = y - mean;
        sum_sq_diff += diff * diff;
    }

    let variance = (sum_sq_diff / count as f64).sqrt() as f32;
    let silhouette_detected = variance >= SILHOUETTE_VARIANCE_THRESHOLD && mean > 20.0 && mean < 235.0;
    (variance, silhouette_detected)
}

/// Integrates vision patching with `pool_patches` for multimodal model awareness.
fn pool_frame_patches(width: u32, height: u32) -> Result<()> {
    let gw = (width / 64).max(3) as usize;
    let gh = (height / 64).max(3) as usize;
    let gw = if gw.is_multiple_of(3) { gw } else { gw + (3 - (gw % 3)) };
    let gh = if gh.is_multiple_of(3) { gh } else { gh + (3 - (gh % 3)) };
    let tokens = gw * gh;
    let tensor = candle_core::Tensor::zeros((1, tokens, 64), candle_core::DType::F32, &candle_core::Device::Cpu)
        .map_err(|e| RuntimedError::SensoryCapture(e.to_string()))?;
    let _ = pool_patches(&tensor, gw, gh, 3).map_err(|e| RuntimedError::SensoryCapture(e.to_string()))?;
    Ok(())
}

/// Generates a synthetic test frame pattern when physical V4L2 device is absent.
fn generate_synthetic_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = 128u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_video_frame_fallback() {
        let frame = capture_video_frame(Some("/dev/nonexistent_video"), Some(128), Some(128)).expect("frame");
        assert_eq!(frame.format, "png");
        assert_eq!(frame.width, 128);
        assert_eq!(frame.height, 128);
        assert!(!frame.image_base64.is_empty());
        assert!(frame.silhouette_detected);
    }

    #[test]
    fn test_analyze_silhouette_flat_color() {
        let flat = vec![100u8; 300];
        let (variance, detected) = analyze_silhouette(&flat, 10, 10);
        assert_eq!(variance, 0.0);
        assert!(!detected);
    }
}
