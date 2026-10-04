//! Pre-assembled PTX for 128-bit vectorized INT4 GEMV kernel (`ld.global.v4.u32`).
//!
//! Evaluates `y = x @ (w * scale)^T` on NVIDIA GPUs (Compute Capability 8.0+ / Ada Lovelace 8.9).
//! Uses 128-bit wide memory transactions (`ld.global.v4.u32`) for peak memory bandwidth utilization.

pub const MARLIN_GEMV_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry marlin_gemv_kernel(
    .param .u64 param_x,
    .param .u64 param_w,
    .param .u64 param_s,
    .param .u64 param_y,
    .param .u32 param_k,
    .param .u32 param_n,
    .param .u32 param_k_chunks
) {
    .reg .pred %p0, %p_loop;
    .reg .b32 %ctaid_x, %ntid_x, %tid_x, %col, %row, %n, %k, %k_chunks, %c;
    .reg .b64 %px, %pw, %ps, %py, %x_base, %y_ptr, %s_ptr, %w_ptr, %x_ptr;
    .reg .b64 %col64, %row64, %n64, %k64, %c64, %tmp64;
    .reg .f32 %acc, %scale, %out_val;
    .reg .b32 %w0, %w1, %w2, %w3;
    .reg .b32 %q, %qs;
    .reg .f32 %wf;
    .reg .f32 %x0, %x1, %x2, %x3;

    ld.param.u32 %n, [param_n];
    mov.u32 %ctaid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %tid_x, %tid.x;
    mad.lo.u32 %col, %ctaid_x, %ntid_x, %tid_x;
    setp.ge.u32 %p0, %col, %n;
    @%p0 bra EXIT;

    ld.param.u32 %k, [param_k];
    ld.param.u32 %k_chunks, [param_k_chunks];
    ld.param.u64 %px, [param_x];
    ld.param.u64 %pw, [param_w];
    ld.param.u64 %ps, [param_s];
    ld.param.u64 %py, [param_y];

    mov.u32 %row, %ctaid.y;
    cvt.u64.u32 %col64, %col;
    cvt.u64.u32 %row64, %row;
    cvt.u64.u32 %n64, %n;
    cvt.u64.u32 %k64, %k;

    // y_ptr = py + (row * n + col) * 4
    mad.wide.u32 %tmp64, %row, %n, %col64;
    shl.b64 %tmp64, %tmp64, 2;
    add.u64 %y_ptr, %py, %tmp64;

    // x_base = px + (row * k) * 4
    mul.wide.u32 %tmp64, %row, %k;
    shl.b64 %tmp64, %tmp64, 2;
    add.u64 %x_base, %px, %tmp64;

    // s_ptr = ps + col * 4
    shl.b64 %tmp64, %col64, 2;
    add.u64 %s_ptr, %ps, %tmp64;
    ld.global.f32 %scale, [%s_ptr];

    mov.f32 %acc, 0.0;
    mov.u32 %c, 0;

LOOP_K:
    setp.ge.u32 %p_loop, %c, %k_chunks;
    @%p_loop bra END_K;

    // w_ptr = pw + (c * n + col) * 16
    cvt.u64.u32 %c64, %c;
    mad.wide.u32 %tmp64, %c, %n, %col64;
    shl.b64 %tmp64, %tmp64, 4;
    add.u64 %w_ptr, %pw, %tmp64;

    // 128-bit vectorized load of 32 INT4 weights
    ld.global.v4.u32 { %w0, %w1, %w2, %w3 }, [ %w_ptr ];

    // x_ptr = x_base + (c * 32) * 4
    shl.b64 %tmp64, %c64, 7;
    add.u64 %x_ptr, %x_base, %tmp64;

    // Process w0 (nibbles 0..7)
    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr ];
    bfe.u32 %q, %w0, 0, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w0, 4, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w0, 8, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w0, 12, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr + 16 ];
    bfe.u32 %q, %w0, 16, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w0, 20, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w0, 24, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w0, 28, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    // Process w1 (nibbles 8..15)
    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr + 32 ];
    bfe.u32 %q, %w1, 0, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w1, 4, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w1, 8, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w1, 12, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr + 48 ];
    bfe.u32 %q, %w1, 16, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w1, 20, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w1, 24, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w1, 28, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    // Process w2 (nibbles 16..23)
    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr + 64 ];
    bfe.u32 %q, %w2, 0, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w2, 4, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w2, 8, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w2, 12, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr + 80 ];
    bfe.u32 %q, %w2, 16, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w2, 20, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w2, 24, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w2, 28, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    // Process w3 (nibbles 24..31)
    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr + 96 ];
    bfe.u32 %q, %w3, 0, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w3, 4, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w3, 8, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w3, 12, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    ld.global.v4.f32 { %x0, %x1, %x2, %x3 }, [ %x_ptr + 112 ];
    bfe.u32 %q, %w3, 16, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x0, %acc;
    bfe.u32 %q, %w3, 20, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x1, %acc;
    bfe.u32 %q, %w3, 24, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x2, %acc;
    bfe.u32 %q, %w3, 28, 4; sub.s32 %qs, %q, 8; cvt.rn.f32.s32 %wf, %qs; fma.rn.f32 %acc, %wf, %x3, %acc;

    add.u32 %c, %c, 1;
    bra LOOP_K;

END_K:
    mul.rn.f32 %out_val, %acc, %scale;
    st.global.f32 [ %y_ptr ], %out_val;

EXIT:
    ret;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_marlin_contains_required_assembly() {
        assert!(MARLIN_GEMV_PTX.contains("ld.global.v4.u32"));
        assert!(MARLIN_GEMV_PTX.contains("bfe.u32"));
        assert!(MARLIN_GEMV_PTX.contains("marlin_gemv_kernel"));
    }
}
