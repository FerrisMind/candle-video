//! RMSNorm across attention heads (`rms_norm_across_heads`).

use std::sync::RwLock;

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::engine::wan_inference_compute_dtype;

#[derive(Debug)]
pub struct AttnRmsNorm {
    weight: Tensor,
    eps: f64,
    /// ponytail: cache weight in compute dtype and the precomputed broadcast shape.
    /// `to_dtype` and `Vec::with_capacity + resize` were a measurable per-forward cost
    /// (called twice per block × 30 blocks × 50 steps).
    cached: RwLock<Vec<(DType, Tensor)>>,
    /// 1-D weight length is fixed after construction; broadcast shape only needs to
    /// depend on rank. Pre-allocate one Vec of length 8 (max realistic rank) once.
    bcast_shapes: std::cell::RefCell<Vec<[usize; 8]>>,
}

impl AttnRmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
            cached: RwLock::new(Vec::new()),
            bcast_shapes: std::cell::RefCell::new(Vec::new()),
        })
    }

    fn weight_in(&self, compute: DType) -> Result<Tensor> {
        if self.weight.dtype() == compute {
            return Ok(self.weight.clone());
        }
        {
            let guard = self.cached.read().expect("attn_rms_norm cached poisoned");
            for (dt, t) in guard.iter() {
                if *dt == compute {
                    return Ok(t.clone());
                }
            }
        }
        let mut guard = self.cached.write().expect("attn_rms_norm cached poisoned");
        for (dt, t) in guard.iter() {
            if *dt == compute {
                return Ok(t.clone());
            }
        }
        let t = self.weight.to_dtype(compute)?;
        guard.push((compute, t.clone()));
        Ok(t)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let compute = wan_inference_compute_dtype(xs.device(), dtype);
        let xs_c = if dtype == compute {
            xs.clone()
        } else {
            xs.to_dtype(compute)?
        };
        let dim = xs_c.dim(D::Minus1)? as f64;
        let ms = xs_c.sqr()?.sum_keepdim(D::Minus1)?.affine(1.0 / dim, 0.0)?;
        let denom = ms.affine(1.0, self.eps)?.sqrt()?;
        let ys = xs_c.broadcast_div(&denom)?;
        let w = self.weight_in(compute)?;
        let rank = ys.rank();
        let mut shapes = self.bcast_shapes.borrow_mut();
        let arr = if rank <= shapes.len() {
            shapes[rank - 1]
        } else {
            shapes.push([1usize; 8]);
            shapes[rank - 1]
        };
        let mut shape = [1usize; 8];
        shape[..rank].copy_from_slice(&arr[..rank]);
        shape[rank - 1] = w.dims1()?;
        ys.broadcast_mul(&w.reshape(&shape[..rank])?)?
            .to_dtype(dtype)
    }
}

impl Module for AttnRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
