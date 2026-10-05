//! Dual-GPU Tensor Parallel MLP and Attention execution.

use crate::error::Result;
use crate::ops::{attention, silu};
use crate::tp::column_linear::ColumnParallelLinear;
use crate::tp::ring_reduce::ring_all_reduce;
use crate::tp::row_linear::RowParallelLinear;
use candle_core::{Device, Tensor};

/// Dual GPU device context.
#[derive(Clone, Debug)]
pub struct DualGpuContext {
    pub dev0: Device,
    pub dev1: Device,
}

impl DualGpuContext {
    pub fn new(dev0: Device, dev1: Device) -> Self {
        Self { dev0, dev1 }
    }

    pub fn devices(&self) -> [Device; 2] {
        [self.dev0.clone(), self.dev1.clone()]
    }

    /// Try to initialize dual CUDA devices (cuda:0 and cuda:1).
    pub fn try_cuda() -> Option<Self> {
        #[cfg(feature = "cuda")]
        {
            let d0 = Device::new_cuda(0).ok()?;
            let d1 = Device::new_cuda(1).ok()?;
            Some(Self { dev0: d0, dev1: d1 })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Runtime activation when inferenced reports >= 2 CUDA devices with active leases.
    pub fn from_inferenced_leases(active_leases: usize) -> Option<Self> {
        if active_leases >= 2 { Self::try_cuda() } else { None }
    }

    /// Check if dual CUDA execution is active and available.
    pub fn is_available() -> bool {
        Self::try_cuda().is_some()
    }
}

/// Tensor-parallel SwiGLU MLP across two ranks.
pub struct DualGpuMlp {
    pub gate_proj: [ColumnParallelLinear; 2],
    pub up_proj: [ColumnParallelLinear; 2],
    pub down_proj: [RowParallelLinear; 2],
}

impl DualGpuMlp {
    pub fn from_full_weights(
        gate_w: &Tensor,
        up_w: &Tensor,
        down_w: &Tensor,
        devs: &[Device; 2],
    ) -> Result<Self> {
        let (gw0, gw1) = (gate_w.to_device(&devs[0])?, gate_w.to_device(&devs[1])?);
        let (uw0, uw1) = (up_w.to_device(&devs[0])?, up_w.to_device(&devs[1])?);
        let (dw0, dw1) = (down_w.to_device(&devs[0])?, down_w.to_device(&devs[1])?);

        let gate_proj = [
            ColumnParallelLinear::from_full_weight(&gw0, None, 0, 2)?,
            ColumnParallelLinear::from_full_weight(&gw1, None, 1, 2)?,
        ];
        let up_proj = [
            ColumnParallelLinear::from_full_weight(&uw0, None, 0, 2)?,
            ColumnParallelLinear::from_full_weight(&uw1, None, 1, 2)?,
        ];
        let down_proj = [
            RowParallelLinear::from_full_weight(&dw0, None, 0, 2)?,
            RowParallelLinear::from_full_weight(&dw1, None, 1, 2)?,
        ];
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x0: &Tensor, x1: &Tensor) -> Result<[Tensor; 2]> {
        let g0 = silu(&self.gate_proj[0].forward(x0)?)?;
        let inter0 = g0.broadcast_mul(&self.up_proj[0].forward(x0)?)?;
        let down0 = self.down_proj[0].forward(&inter0)?;

        let g1 = silu(&self.gate_proj[1].forward(x1)?)?;
        let inter1 = g1.broadcast_mul(&self.up_proj[1].forward(x1)?)?;
        let down1 = self.down_proj[1].forward(&inter1)?;

        let red = ring_all_reduce(&[down0, down1])?;
        Ok([red[0].clone(), red[1].clone()])
    }
}

/// Tensor-parallel Attention across two ranks.
pub struct DualGpuAttention {
    pub q_proj: [ColumnParallelLinear; 2],
    pub k_proj: [ColumnParallelLinear; 2],
    pub v_proj: [ColumnParallelLinear; 2],
    pub o_proj: [RowParallelLinear; 2],
    pub num_heads_per_rank: usize,
    pub num_kv_heads_per_rank: usize,
    pub head_dim: usize,
}

impl DualGpuAttention {
    pub fn from_full_weights(
        q_w: &Tensor, k_w: &Tensor, v_w: &Tensor, o_w: &Tensor,
        num_heads: usize, num_kv_heads: usize, devs: &[Device; 2],
    ) -> Result<Self> {
        let (qw0, qw1) = (q_w.to_device(&devs[0])?, q_w.to_device(&devs[1])?);
        let (kw0, kw1) = (k_w.to_device(&devs[0])?, k_w.to_device(&devs[1])?);
        let (vw0, vw1) = (v_w.to_device(&devs[0])?, v_w.to_device(&devs[1])?);
        let (ow0, ow1) = (o_w.to_device(&devs[0])?, o_w.to_device(&devs[1])?);
        let head_dim = q_w.dim(0)? / num_heads;
        let q_proj = [ColumnParallelLinear::from_full_weight(&qw0, None, 0, 2)?, ColumnParallelLinear::from_full_weight(&qw1, None, 1, 2)?];
        let k_proj = [ColumnParallelLinear::from_full_weight(&kw0, None, 0, 2)?, ColumnParallelLinear::from_full_weight(&kw1, None, 1, 2)?];
        let v_proj = [ColumnParallelLinear::from_full_weight(&vw0, None, 0, 2)?, ColumnParallelLinear::from_full_weight(&vw1, None, 1, 2)?];
        let o_proj = [RowParallelLinear::from_full_weight(&ow0, None, 0, 2)?, RowParallelLinear::from_full_weight(&ow1, None, 1, 2)?];
        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            num_heads_per_rank: num_heads / 2,
            num_kv_heads_per_rank: num_kv_heads / 2,
            head_dim,
        })
    }

    fn rank_forward(&self, rank: usize, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len) = (x.dim(0)?, x.dim(1)?);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let q = self.q_proj[rank].forward(x)?.reshape((b, seq_len, self.num_heads_per_rank, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj[rank].forward(x)?.reshape((b, seq_len, self.num_kv_heads_per_rank, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj[rank].forward(x)?.reshape((b, seq_len, self.num_kv_heads_per_rank, self.head_dim))?.transpose(1, 2)?;
        let att = attention(&q, &k, &v, mask, scale)?.transpose(1, 2)?.reshape((b, seq_len, self.num_heads_per_rank * self.head_dim))?;
        self.o_proj[rank].forward(&att)
    }

    pub fn forward(
        &self,
        x0: &Tensor,
        x1: &Tensor,
        mask0: Option<&Tensor>,
        mask1: Option<&Tensor>,
    ) -> Result<[Tensor; 2]> {
        let out0 = self.rank_forward(0, x0, mask0)?;
        let out1 = self.rank_forward(1, x1, mask1)?;
        let red = ring_all_reduce(&[out0, out1])?;
        Ok([red[0].clone(), red[1].clone()])
    }
}

/// Unified Dual-GPU Transformer Block combining TP=2 Attention and SwiGLU MLP.
pub struct DualGpuBlock {
    pub attn: DualGpuAttention,
    pub mlp: DualGpuMlp,
}

impl DualGpuBlock {
    pub fn new(attn: DualGpuAttention, mlp: DualGpuMlp) -> Self {
        Self { attn, mlp }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_full_weights(
        qw: &Tensor, kw: &Tensor, vw: &Tensor, ow: &Tensor,
        nh: usize, nkv: usize,
        gw: &Tensor, uw: &Tensor, dw: &Tensor,
        devs: &[Device; 2],
    ) -> Result<Self> {
        let attn = DualGpuAttention::from_full_weights(qw, kw, vw, ow, nh, nkv, devs)?;
        let mlp = DualGpuMlp::from_full_weights(gw, uw, dw, devs)?;
        Ok(Self { attn, mlp })
    }

    pub fn forward(
        &self, x0: &Tensor, x1: &Tensor, mask0: Option<&Tensor>, mask1: Option<&Tensor>,
    ) -> Result<[Tensor; 2]> {
        let [a0, a1] = self.attn.forward(x0, x1, mask0, mask1)?;
        let (h0, h1) = ((x0 + a0)?, (x1 + a1)?);
        let [m0, m1] = self.mlp.forward(&h0, &h1)?;
        Ok([(&h0 + m0)?, (&h1 + m1)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_gpu_mlp_and_block() {
        let dev = Device::Cpu;
        let devs = [dev.clone(), dev.clone()];
        let (h, inter, nh, nkv) = (16, 32, 4, 4);
        let (gw, uw, dw) = (Tensor::randn(0.0f32, 1.0, (inter, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (inter, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (h, inter), &dev).unwrap());
        let mlp = DualGpuMlp::from_full_weights(&gw, &uw, &dw, &devs).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (2, h), &dev).unwrap();
        let [y0, y1] = mlp.forward(&x, &x).unwrap();
        assert!((y0 - y1).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() < 1e-4);

        let (qw, kw, vw, ow) = (Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap());
        let blk = DualGpuBlock::from_full_weights(&qw, &kw, &vw, &ow, nh, nkv, &gw, &uw, &dw, &devs).unwrap();
        let seq_x = Tensor::randn(0.0f32, 1.0, (1, 2, h), &dev).unwrap();
        let [b0, b1] = blk.forward(&seq_x, &seq_x, None, None).unwrap();
        assert_eq!(b0.dims(), b1.dims());
    }

    #[test]
    fn test_dual_gpu_attention_equivalence() {
        let dev = Device::Cpu;
        let devs = [dev.clone(), dev.clone()];
        let (h, nh, nkv) = (16, 4, 4);
        let (qw, kw, vw, ow) = (Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap(),
            Tensor::randn(0.0f32, 1.0, (h, h), &dev).unwrap());
        let att = DualGpuAttention::from_full_weights(&qw, &kw, &vw, &ow, nh, nkv, &devs).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 3, h), &dev).unwrap();
        let [out0, out1] = att.forward(&x, &x, None, None).unwrap();
        assert!((out0 - out1).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_dual_cuda_devices_if_present() {
        if let Some(ctx) = DualGpuContext::try_cuda() {
            let devs = ctx.devices();
            let (h, inter) = (8, 16);
            let (gw, uw, dw) = (Tensor::randn(0.0f32, 1.0, (inter, h), &Device::Cpu).unwrap(),
                Tensor::randn(0.0f32, 1.0, (inter, h), &Device::Cpu).unwrap(),
                Tensor::randn(0.0f32, 1.0, (h, inter), &Device::Cpu).unwrap());
            let mlp = DualGpuMlp::from_full_weights(&gw, &uw, &dw, &devs).unwrap();
            let x0 = Tensor::randn(0.0f32, 1.0, (2, h), &devs[0]).unwrap();
            let [y0, y1] = mlp.forward(&x0, &x0.to_device(&devs[1]).unwrap()).unwrap();
            assert_eq!(y0.dims(), &[2, h]);
            assert_eq!(y1.dims(), &[2, h]);
        }
    }
}
