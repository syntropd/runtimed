//! Pillow-exact bicubic resizer (22-bit fixed point, u8 between passes).
//!
//! Pure pixel math behind [`crate::vision::vpre`]: target sizing plus the
//! two-pass separable resampler the vision oracle is recorded against.

/// Keys bicubic kernel with Pillow's a = -0.5, separable, edge-clamped.
fn bicubic_kernel(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x <= 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::bicubic_kernel;

    #[test]
    fn kernel_matches_keys_values() {
        // a = -0.5: f(0) = 1, f(1) = 0, f(0.5) = 0.5625, partition of unity.
        assert!((bicubic_kernel(0.0) - 1.0).abs() < 1e-12);
        assert!(bicubic_kernel(1.0).abs() < 1e-12);
        assert!((bicubic_kernel(0.5) - 0.5625).abs() < 1e-12);
        assert!((bicubic_kernel(1.5) - -0.0625).abs() < 1e-12);
        // Integer-shifted taps at any phase sum to 1 (constant-preserving).
        for phase in [0.0, 0.13, 0.5, 0.77] {
            let s: f64 = (-1..3).map(|k| bicubic_kernel(phase - k as f64)).sum();
            assert!((s - 1.0).abs() < 1e-12, "phase {phase}: {s}");
        }
    }
}

/// Target size preserving aspect ratio: align up to `align`, then fit the
/// pixel budget (upscale small images, downscale huge ones).
pub fn smart_resize(w: usize, h: usize, align: usize, min_px: usize, max_px: usize) -> (usize, usize) {
    let f = align as f64;
    let round = |x: f64| (x / f).round() as usize * align;
    let ceil = |x: f64| (x / f).ceil() as usize * align;
    let floor = |x: f64| (x / f).floor() as usize * align;
    let mut tw = round(w as f64).max(align);
    let mut th = round(h as f64).max(align);
    if tw * th > max_px {
        let beta = ((w * h) as f64 / max_px as f64).sqrt();
        tw = floor(w as f64 / beta).max(align);
        th = floor(h as f64 / beta).max(align);
    } else if tw * th < min_px {
        let beta = (min_px as f64 / (w * h) as f64).sqrt();
        tw = ceil(w as f64 * beta);
        th = ceil(h as f64 * beta);
    }
    (tw, th)
}

/// Fixed-point precision of the Pillow resampler this ports (22 fractional
/// bits: 32 - 8 pixel bits - 2 accumulation headroom).
const PILLOW_PRECISION_BITS: u32 = 22;

/// Pillow `Resample.c`-compatible bicubic resize, ported from the reference
/// `resize_pillow` (support 2.0, a = -0.5). Key details that must match:
///
/// - per-output-pixel kernel ranges truncated (not edge-repeated) and
///   renormalized, then quantized to 22-bit fixed point with round-half-
///   away-from-zero;
/// - u8 horizontal intermediate with clamp between passes (this kills the
///   ringing lobes a single float pipeline would carry into corners);
/// - rounding bias + arithmetic shift back to u8 after each pass.
///
/// Returns row-major f32 triples in [0, 1].
pub(crate) fn bicubic_resize(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    let horiz = if tw == w {
        rgb.to_vec()
    } else {
        resample_pass(rgb, w, h, tw, true)
    };
    let out = if th == h {
        horiz
    } else {
        resample_pass(&horiz, tw, h, th, false)
    };
    out.iter().map(|&v| v as f32 / 255.0).collect()
}

/// One Pillow resampling pass over packed RGB u8. `horizontal` picks the
/// axis; the other axis passes through untouched.
fn resample_pass(src: &[u8], w: usize, h: usize, t: usize, horizontal: bool) -> Vec<u8> {
    let in_size = if horizontal { w } else { h };
    let scale = in_size as f64 / t as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    // Precompute normalized fixed-point weights per output position.
    let mut bounds = Vec::with_capacity(t * 2);
    let mut pre = vec![0.0f64; t * ksize];
    for o in 0..t {
        let center = (o as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        xmax -= xmin;
        let mut ww = 0.0;
        for x in 0..xmax as usize {
            let wt = bicubic_kernel((x as f64 + xmin as f64 - center + 0.5) * ss);
            pre[o * ksize + x] = wt;
            ww += wt;
        }
        if ww != 0.0 {
            for x in 0..xmax as usize {
                pre[o * ksize + x] /= ww;
            }
        }
        bounds.push(xmin as usize);
        bounds.push(xmax as usize);
    }
    let fxp = (1u64 << PILLOW_PRECISION_BITS) as f64;
    let mut weights = vec![0i32; t * ksize];
    for (i, &wt) in pre.iter().enumerate() {
        // Round half away from zero via bias + truncation, like the reference.
        weights[i] = (wt * fxp + if wt < 0.0 { -0.5 } else { 0.5 }) as i32;
    }
    let clip8 = |v: i32| v.clamp(0, 255) as u8;
    if horizontal {
        let mut out = vec![0u8; t * h * 3];
        for y in 0..h {
            for ox in 0..t {
                let xmin = bounds[ox * 2];
                let xcnt = bounds[ox * 2 + 1];
                let k = &weights[ox * ksize..];
                let mut acc = [1i32 << (PILLOW_PRECISION_BITS - 1); 3];
                for (x, &weight) in k.iter().enumerate().take(xcnt) {
                    let p = (y * w + xmin + x) * 3;
                    acc[0] += src[p] as i32 * weight;
                    acc[1] += src[p + 1] as i32 * weight;
                    acc[2] += src[p + 2] as i32 * weight;
                }
                let d = (y * t + ox) * 3;
                out[d] = clip8(acc[0] >> PILLOW_PRECISION_BITS);
                out[d + 1] = clip8(acc[1] >> PILLOW_PRECISION_BITS);
                out[d + 2] = clip8(acc[2] >> PILLOW_PRECISION_BITS);
            }
        }
        out
    } else {
        let row_elems = w * 3;
        let mut out = vec![0u8; row_elems * t];
        let mut acc = vec![0i32; row_elems];
        for oy in 0..t {
            let ymin = bounds[oy * 2];
            let ycnt = bounds[oy * 2 + 1];
            let k = &weights[oy * ksize..];
            acc.fill(1i32 << (PILLOW_PRECISION_BITS - 1));
            for y in 0..ycnt {
                let row = &src[(ymin + y) * row_elems..][..row_elems];
                let wt = k[y];
                for (a, &s) in acc.iter_mut().zip(row.iter()) {
                    *a += s as i32 * wt;
                }
            }
            let d = oy * row_elems;
            for (o, &a) in out[d..d + row_elems].iter_mut().zip(acc.iter()) {
                *o = clip8(a >> PILLOW_PRECISION_BITS);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::smart_resize;

    #[test]
    fn resize_math_matches_reference_cases() {
        // align 48, budget [70, 1120] soft tokens (px = soft * 2304).
        let (w, h) = smart_resize(64, 64, 48, 70 * 2304, 1120 * 2304);
        assert_eq!((w, h), (432, 432)); // 27x27 patches -> 81 soft
        let (w, h) = smart_resize(96, 64, 48, 70 * 2304, 1120 * 2304);
        assert_eq!((w, h), (528, 336)); // 33x21 patches -> 77 soft
        // Huge images shrink to budget.
        let (w, h) = smart_resize(4000, 3000, 48, 70 * 2304, 1120 * 2304);
        assert!(w * h <= 1120 * 2304 + 48 * 48 * 4);
        assert_eq!(w % 48, 0);
        assert_eq!(h % 48, 0);
    }
}