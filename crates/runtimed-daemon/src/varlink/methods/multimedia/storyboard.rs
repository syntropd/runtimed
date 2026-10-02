//! Storyboard keyframe synthesis and image strip stitching.

use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder, RgbImage};
use runtimed_model::visual_gen::{VisualComputeLease, VisualGenSampler};
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Resolve runtime storage directory: $XDG_RUNTIME_DIR -> /run/user/<uid> -> /run (if /run/syntrop exists) -> temp_dir().
pub async fn resolve_runtime_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg.trim());
        if !p.as_os_str().is_empty() && tokio::fs::try_exists(&p).await.unwrap_or(false) {
            return p;
        }
    }
    let uid = rustix::process::getuid().as_raw();
    let run_user = PathBuf::from(format!("/run/user/{uid}"));
    if tokio::fs::try_exists(&run_user).await.unwrap_or(false) {
        return run_user;
    }
    let run_syntrop = PathBuf::from("/run/syntrop");
    if tokio::fs::try_exists(&run_syntrop).await.unwrap_or(false) {
        return PathBuf::from("/run");
    }
    std::env::temp_dir()
}

/// Result of generating a storyboard keyframe strip.
#[derive(Debug, Clone)]
pub struct StoryboardResult {
    pub storyboard_path: PathBuf,
    pub manifest_path: PathBuf,
    pub bytes: usize,
    pub width: u32,
    pub height: u32,
    pub keyframes: usize,
}

/// Stitches a series of RGB keyframe images into a single horizontal strip.
pub fn stitch_keyframe_images(images: &[RgbImage], frame_w: u32, frame_h: u32) -> Result<Vec<u8>, String> {
    if images.is_empty() {
        return Err("no keyframe images to stitch".into());
    }
    let n = images.len() as u32;
    let total_w = frame_w.checked_mul(n).ok_or("dimensions overflow")?;
    let mut strip = RgbImage::new(total_w, frame_h);

    for (i, img) in images.iter().enumerate() {
        let offset_x = (i as u32) * frame_w;
        for y in 0..frame_h {
            for x in 0..frame_w {
                strip.put_pixel(offset_x + x, y, *img.get_pixel(x, y));
            }
        }
    }

    let mut png_bytes = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(strip.as_raw(), total_w, frame_h, ExtendedColorType::Rgb8)
        .map_err(|e| format!("PNG encoding failed: {e}"))?;

    Ok(png_bytes)
}

/// Atomically writes bytes to an artifact file with staging permissions.
pub async fn write_atomic_file(
    out_dir: &std::path::Path,
    id: &str,
    ext: &str,
    bytes: &[u8],
) -> Result<PathBuf, String> {
    tokio::fs::create_dir_all(out_dir)
        .await
        .map_err(|e| format!("create output dir: {e}"))?;
    let final_path = out_dir.join(format!("{id}.{ext}"));
    let tmp_path = out_dir.join(format!(".{id}.tmp"));

    if let Err(e) = tokio::fs::write(&tmp_path, bytes).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(format!("write temporary file: {e}"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = tokio::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o644)).await;
    }
    if let Err(e) = tokio::fs::rename(&tmp_path, &final_path).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(format!("commit file: {e}"));
    }
    Ok(final_path)
}

/// Generates `n` keyframes via 1-step fast sampler and stitches them into a strip and manifest.
pub async fn render_storyboard_strip(
    prompt: &str,
    n: usize,
    width: u32,
    height: u32,
    seed: u64,
    sampler: &VisualGenSampler,
    fallback_reason: Option<&str>,
) -> Result<StoryboardResult, String> {
    let keyframe_count = n.clamp(1, 16);
    let lease = VisualComputeLease::new("storyboard-lease");
    let mut images = Vec::with_capacity(keyframe_count);

    for i in 0..keyframe_count {
        let frame_seed = seed.wrapping_add((i as u64).wrapping_mul(1000));
        let png = sampler
            .sample_1step(prompt, width, height, frame_seed, &lease)
            .map_err(|e| format!("keyframe {i} sampling failed: {e}"))?;
        let dyn_img = image::load_from_memory(&png)
            .map_err(|e| format!("decoding keyframe {i} failed: {e}"))?;
        images.push(dyn_img.to_rgb8());
    }

    let strip_bytes = stitch_keyframe_images(&images, width, height)?;
    let out_dir = resolve_runtime_dir().await.join("syntrop").join("visual_gen");

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let id = format!("{nanos:016x}_{count}_storyboard");
    let strip_path = write_atomic_file(&out_dir, &id, "png", &strip_bytes).await?;
    let manifest_path = out_dir.join(format!("{id}_manifest.json"));

    let manifest = json!({
        "prompt": prompt,
        "keyframes": keyframe_count,
        "frame_width": width,
        "frame_height": height,
        "strip_width": width * (keyframe_count as u32),
        "strip_height": height,
        "storyboard_path": strip_path.to_string_lossy(),
        "sampler": "sd-turbo-1step",
        "fallback_reason": fallback_reason.unwrap_or("requested"),
    });

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| format!("manifest serialization: {e}"))?;
    tokio::fs::write(&manifest_path, &manifest_bytes)
        .await
        .map_err(|e| format!("write manifest: {e}"))?;

    Ok(StoryboardResult {
        storyboard_path: strip_path,
        manifest_path,
        bytes: strip_bytes.len(),
        width: width * (keyframe_count as u32),
        height,
        keyframes: keyframe_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtimed_model::visual_gen::VisualGenConfig;

    #[tokio::test]
    async fn test_stitch_keyframe_images() {
        let mut img1 = RgbImage::new(16, 16);
        let mut img2 = RgbImage::new(16, 16);
        img1.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img2.put_pixel(0, 0, image::Rgb([0, 255, 0]));

        let res = stitch_keyframe_images(&[img1, img2], 16, 16);
        assert!(res.is_ok());
        let png = res.unwrap();
        assert!(!png.is_empty());

        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 16);
    }

    #[tokio::test]
    async fn test_render_storyboard_strip() {
        let sampler = VisualGenSampler::with_config(VisualGenConfig {
            default_width: 32,
            default_height: 32,
            steps: 1,
            lora_tags: vec![],
        });
        let res = render_storyboard_strip("cyberpunk street", 2, 32, 32, 42, &sampler, None).await;
        assert!(res.is_ok());
        let story = res.unwrap();
        assert_eq!(story.keyframes, 2);
        assert_eq!(story.width, 64);
        assert_eq!(story.height, 32);
        assert!(tokio::fs::try_exists(&story.storyboard_path).await.unwrap_or(false));
        assert!(tokio::fs::try_exists(&story.manifest_path).await.unwrap_or(false));

        let _ = tokio::fs::remove_file(story.storyboard_path).await;
        let _ = tokio::fs::remove_file(story.manifest_path).await;
    }
}
