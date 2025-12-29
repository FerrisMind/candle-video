//! VAE Decoder for LTX-Video embedded weights
//!
//! This module provides the VAE decoder that loads weights directly from the
//! embedded format in ltxv-2b-0.9.8-distilled.safetensors:
//! - vae.decoder.conv_in
//! - vae.decoder.up_blocks.{0,2,4,6} - res_blocks with time_embedder
//! - vae.decoder.up_blocks.{1,3,5} - upsampler conv only
//! - vae.decoder.{last_time_embedder, last_scale_shift_table, conv_out}

use candle_core::{D, DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear, ops::silu};

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Apply causal padding to temporal dimension (left side only)
pub fn causal_pad(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let first_frame = x.i((.., .., 0..1, .., ..))?;
    let pad_frames = first_frame.repeat((1, 1, pad, 1, 1))?;
    Tensor::cat(&[&pad_frames, x], 2)
}

/// Apply symmetric padding to spatial dimensions
pub fn spatial_pad(x: &Tensor, pad_h: usize, pad_w: usize) -> Result<Tensor> {
    if pad_h == 0 && pad_w == 0 {
        return Ok(x.clone());
    }
    let x = x.pad_with_zeros(4, pad_w, pad_w)?;
    x.pad_with_zeros(3, pad_h, pad_h)
}

/// Unpatchify a 5D tensor using depth-to-space transformation (for Upsampler)
///
/// Uses per-frame processing to avoid B*T folding issues that cause incorrect pixel ordering.
/// Handles temporal stride and causal slicing for upsampler blocks.
pub fn unpatchify(x: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(x.clone());
    }

    let (b, c_packed, t, h, w) = x.dims5()?;
    let s_t = patch_size_t;
    let s_h = patch_size_hw;
    let s_w = patch_size_hw;
    let scale_factor = s_t * s_h * s_w;
    let c = c_packed / scale_factor;
    let h_new = h * s_h;
    let w_new = w * s_w;

    // Process each frame separately to avoid B*T folding memory layout issues
    let mut frames = Vec::with_capacity(b * t);

    for bi in 0..b {
        for ti in 0..t {
            // Extract frame: [1, C_packed, 1, H, W] -> [1, C_packed, H, W]
            let frame = x.i((bi..bi + 1, .., ti..ti + 1, .., ..))?;
            let frame = frame.squeeze(2)?;

            // Reshape: [1, c*s_t, s_h, s_w, H, W]
            let frame = frame.reshape(&[1, c * s_t, s_h, s_w, h, w][..])?;

            // Permute to [1, c*s_t, H, s_h, W, s_w] to match diffusers LTXVideoUpsampler3d
            // (see `tp/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx.py`).
            let frame = frame.permute(&[0, 1, 4, 2, 5, 3][..])?;
            let frame = frame.contiguous()?;

            // Flatten: [1, c*s_t, H*s_h, W*s_w]
            let frame = frame.reshape((1, c * s_t, h_new, w_new))?;
            frames.push(frame);
        }
    }

    // Stack frames: [B*T, c*s_t, H_new, W_new]
    let x = Tensor::cat(&frames, 0)?;

    // Handle temporal unpacking
    if s_t > 1 {
        // Reshape to [B, T, c*s_t, H_new, W_new]
        let x = x.reshape(&[b, t, c * s_t, h_new, w_new][..])?;
        // Reshape to [B, T, c, s_t, H_new, W_new]
        let x = x.reshape(&[b, t, c, s_t, h_new, w_new][..])?;
        // Permute to [B, c, T, s_t, H_new, W_new]
        let x = x.permute(&[0, 2, 1, 3, 4, 5][..])?;
        // Flatten temporal: [B, c, T*s_t, H_new, W_new]
        let t_new = t * s_t;
        let x = x.contiguous()?.reshape((b, c, t_new, h_new, w_new))?;

        // Slice for causal boundary: [:, :, s_t-1:]
        let start_t = s_t - 1;
        if start_t > 0 && t_new > start_t {
            x.narrow(2, start_t, t_new - start_t)
        } else {
            Ok(x)
        }
    } else {
        // Reshape to [B, T, c, H_new, W_new] then permute to [B, c, T, H_new, W_new]
        let x = x.reshape(&[b, t, c, h_new, w_new][..])?;
        x.permute(&[0, 2, 1, 3, 4][..])
    }
}

/// Final decoder unpatchify with per-frame processing
///
/// Uses per-frame processing to avoid B*T folding issues.
/// This produces correct channel-to-pixel mapping matching Python.
pub fn unpatchify_decoder(x: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(x.clone());
    }

    let (b, c_packed, t, h, w) = x.dims5()?;
    let p_t = patch_size_t;
    let p = patch_size_hw;
    let c = c_packed / (p_t * p * p);
    let h_new = h * p;
    let w_new = w * p;

    // Process each frame separately to avoid B*T folding memory layout issues
    let mut frames = Vec::with_capacity(b * t);

    for bi in 0..b {
        for ti in 0..t {
            // Extract frame: [1, C_packed, 1, H, W] -> [1, C_packed, H, W]
            let frame = x.i((bi..bi + 1, .., ti..ti + 1, .., ..))?;
            let frame = frame.squeeze(2)?;

            // Reshape: [1, c*p_t, p, p, H, W]
            let frame = frame.reshape(&[1, c * p_t, p, p, h, w][..])?;

            // Permute to [1, c*p_t, H, p_w, W, p_h] - swapped for correct Python mapping
            // This gives output(y,x) = channel[x*p + y] matching Python
            let frame = frame.permute(&[0, 1, 4, 3, 5, 2][..])?;
            let frame = frame.contiguous()?;

            // Flatten: [1, c*p_t, H*p, W*p]
            let frame = frame.reshape((1, c * p_t, h_new, w_new))?;
            frames.push(frame);
        }
    }

    // Stack frames: [B*T, c*p_t, H_new, W_new]
    let x = Tensor::cat(&frames, 0)?;

    // Handle temporal unpacking (for p_t > 1)
    if p_t > 1 {
        let x = x.reshape(&[b, t, c * p_t, h_new, w_new][..])?;
        let x = x.reshape(&[b, t, c, p_t, h_new, w_new][..])?;
        let x = x.permute(&[0, 2, 1, 3, 4, 5][..])?;
        let t_new = t * p_t;
        x.contiguous()?.reshape((b, c, t_new, h_new, w_new))
    } else {
        // Reshape to [B, T, c, H_new, W_new] then permute to [B, c, T, H_new, W_new]
        let x = x.reshape(&[b, t, c, h_new, w_new][..])?;
        x.permute(&[0, 2, 1, 3, 4][..])
    }
}

/// Patchify a 5D tensor using space-to-depth transformation
#[allow(dead_code)]
pub fn patchify(x: &Tensor, patch_size_hw: usize, patch_size_t: usize) -> Result<Tensor> {
    if patch_size_hw == 1 && patch_size_t == 1 {
        return Ok(x.clone());
    }

    let (b, c, t, h, w) = x.dims5()?;

    if patch_size_t > 1 {
        let t_new = t / patch_size_t;
        let h_new = h / patch_size_hw;
        let w_new = w / patch_size_hw;
        let c_new = c * patch_size_t * patch_size_hw * patch_size_hw;

        let x = x.reshape((b, c, t_new, patch_size_t, h * w))?;
        let x = x.permute((0, 1, 3, 2, 4))?.contiguous()?;
        let x = x.reshape((b, c * patch_size_t, t_new, h, w))?;

        let x = x.reshape((b * t_new, c * patch_size_t, h_new, patch_size_hw, w))?;
        let x = x.reshape((
            b * t_new,
            c * patch_size_t,
            h_new,
            patch_size_hw,
            w_new,
            patch_size_hw,
        ))?;
        let x = x.permute((0, 1, 3, 5, 2, 4))?.contiguous()?;
        let x = x.reshape((b * t_new, c_new, h_new, w_new))?;

        x.reshape((b, c_new, t_new, h_new, w_new))
    } else {
        let h_new = h / patch_size_hw;
        let w_new = w / patch_size_hw;
        let c_new = c * patch_size_hw * patch_size_hw;

        let x = x.permute((0, 2, 1, 3, 4))?.contiguous()?;
        let x = x.reshape((b * t, c, h, w))?;
        let x = x.reshape((b * t, c, h_new, patch_size_hw, w_new, patch_size_hw))?;
        let x = x.permute((0, 1, 3, 5, 2, 4))?.contiguous()?;
        let x = x.reshape((b * t, c_new, h_new, w_new))?;

        let x = x.reshape((b, t, c_new, h_new, w_new))?;
        x.permute((0, 2, 1, 3, 4))
    }
}

// ===========================================================================
// RMSNorm3d
// ===========================================================================

/// Root Mean Square Normalization for video tensors (no learnable weights)
pub struct RMSNorm3d {
    eps: f64,
}

impl RMSNorm3d {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;

        let x = x.permute((0, 2, 3, 4, 1))?;
        let x = x.reshape((b * t * h * w, c))?;

        // Use affine to add eps while preserving dtype (BF16)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = variance.affine(1.0, self.eps)?.sqrt()?;
        let x = x.broadcast_div(&rms)?;

        let x = x.reshape((b, t, h, w, c))?;
        x.permute((0, 4, 1, 2, 3))
    }
}

// ===========================================================================
// CausalConv3d
// ===========================================================================

pub struct CausalConv3d {
    weight: Tensor,
    bias: Tensor,
    kernel_size: usize,
    is_causal: bool,
}

impl CausalConv3d {
    pub fn new(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        is_causal: bool,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_ch, in_ch, kernel_size, kernel_size, kernel_size),
            "conv.weight",
        )?;
        let bias = vb.get(out_ch, "conv.bias")?;
        Ok(Self {
            weight,
            bias,
            kernel_size,
            is_causal,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let k = self.kernel_size;
        let pad = (k - 1) / 2;

        // Temporal padding
        let x = if self.is_causal && k > 1 {
            causal_pad(x, k - 1)?
        } else if k > 1 {
            let (_b, _c, t, _h, _w) = x.dims5()?;
            let first_frame = x.i((.., .., 0..1, .., ..))?;
            let last_frame = x.i((.., .., (t - 1)..t, .., ..))?;

            let pad_before = vec![first_frame.clone(); pad];
            let pad_after = vec![last_frame.clone(); pad];

            let mut frames = pad_before;
            frames.push(x.clone());
            frames.extend(pad_after);

            Tensor::cat(&frames, 2)?
        } else {
            x.clone()
        };

        // Spatial padding
        let x = spatial_pad(&x, pad, pad)?;

        // 3D conv via 2D folding
        let (b, c_in, t, h, w) = x.dims5()?;
        let c_out = self.weight.dim(0)?;
        let weight_2d = self.weight.reshape((c_out, c_in * k, k, k))?;

        let t_out = t.saturating_sub(k) + 1;
        let mut outputs = Vec::with_capacity(t_out);
        for t_start in 0..t_out {
            let window = x.i((.., .., t_start..t_start + k, .., ..))?;
            let window_2d = window.reshape((b, c_in * k, h, w))?;
            let out_2d = window_2d.conv2d(&weight_2d, 0, 1, 1, 1)?;
            outputs.push(out_2d);
        }

        let out = Tensor::stack(&outputs, 2)?;
        let bias = self.bias.reshape((1, c_out, 1, 1, 1))?;
        out.broadcast_add(&bias)
    }
}

// ===========================================================================
// TimeEmbedder
// ===========================================================================

pub struct TimeEmbedder {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimeEmbedder {
    pub fn new(vb: VarBuilder, output_dim: usize) -> Result<Self> {
        let linear_1 = linear(256, output_dim, vb.pp("timestep_embedder.linear_1"))?;
        let linear_2 = linear(output_dim, output_dim, vb.pp("timestep_embedder.linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    fn sinusoidal_embedding(&self, timesteps: &Tensor) -> Result<Tensor> {
        let device = timesteps.device();
        let half_dim = 128;
        // Compute in F32 for numerical precision, then convert to target dtype
        let indices = Tensor::arange(0u32, half_dim as u32, device)?.to_dtype(DType::F32)?;
        let log_val = 10000.0_f64.ln();
        let exponent = (indices.affine(-log_val / (half_dim as f64), 0.0))?.exp()?;

        let ts = timesteps.to_dtype(DType::F32)?;
        let emb = ts.unsqueeze(1)?.broadcast_mul(&exponent.unsqueeze(0)?)?;

        Tensor::cat(&[emb.sin()?, emb.cos()?], D::Minus1)
    }

    pub fn forward(&self, timesteps: &Tensor, batch_size: usize) -> Result<Tensor> {
        let emb = self.sinusoidal_embedding(timesteps)?;
        let emb = if emb.dim(0)? == 1 && batch_size > 1 {
            emb.broadcast_as((batch_size, 256))?
        } else {
            emb
        };

        // Convert embedding to match linear layer dtype (BF16)
        let target_dtype = self.linear_1.weight().dtype();
        let emb = emb.to_dtype(target_dtype)?;

        let h = self.linear_1.forward(&emb)?;
        let h = silu(&h)?;
        self.linear_2.forward(&h)
    }
}

// ===========================================================================
// ResBlock
// ===========================================================================

pub struct ResBlock {
    norm1: RMSNorm3d,
    conv1: CausalConv3d,
    norm2: RMSNorm3d,
    conv2: CausalConv3d,
    scale_shift_table: Tensor,
    channels: usize,
}

impl ResBlock {
    pub fn new(vb: VarBuilder, channels: usize, is_causal: bool) -> Result<Self> {
        let norm1 = RMSNorm3d::new(1e-8);
        let conv1 = CausalConv3d::new(vb.pp("conv1"), channels, channels, 3, is_causal)?;
        let norm2 = RMSNorm3d::new(1e-8);
        let conv2 = CausalConv3d::new(vb.pp("conv2"), channels, channels, 3, is_causal)?;
        let scale_shift_table = vb.get((4, channels), "scale_shift_table")?;

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            scale_shift_table,
            channels,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.norm1.forward(x)?;

        let (shift_1, scale_1, shift_2, scale_2) = if let Some(temb) = temb {
            let temb_unflat = temb.reshape((temb.dim(0)?, 4, self.channels, 1, 1, 1))?;
            let table_exp = self
                .scale_shift_table
                .reshape((1, 4, self.channels, 1, 1, 1))?;
            let combined = (temb_unflat + table_exp)?;

            let s1 = combined.i((.., 0, .., .., .., ..))?.squeeze(1)?;
            let sc1 = combined.i((.., 1, .., .., .., ..))?.squeeze(1)?;
            let s2 = combined.i((.., 2, .., .., .., ..))?.squeeze(1)?;
            let sc2 = combined.i((.., 3, .., .., .., ..))?.squeeze(1)?;
            (Some(s1), Some(sc1), Some(s2), Some(sc2))
        } else {
            (None, None, None, None)
        };

        // First modulation
        let hidden = if let (Some(shift), Some(scale)) = (&shift_1, &scale_1) {
            let ones = Tensor::ones_like(scale)?;
            let scaled = hidden.broadcast_mul(&(ones + scale)?)?;
            scaled.broadcast_add(shift)?
        } else {
            hidden
        };

        let hidden = silu(&hidden)?;
        let hidden = self.conv1.forward(&hidden)?;

        let hidden = self.norm2.forward(&hidden)?;

        // Second modulation
        let hidden = if let (Some(shift), Some(scale)) = (&shift_2, &scale_2) {
            let ones = Tensor::ones_like(scale)?;
            let scaled = hidden.broadcast_mul(&(ones + scale)?)?;
            scaled.broadcast_add(shift)?
        } else {
            hidden
        };

        let hidden = silu(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;

        hidden + x
    }
}

// ===========================================================================
// ResBlockGroup
// ===========================================================================

pub struct ResBlockGroup {
    time_embedder: TimeEmbedder,
    res_blocks: Vec<ResBlock>,
    channels: usize,
}

impl ResBlockGroup {
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        num_blocks: usize,
        is_causal: bool,
    ) -> Result<Self> {
        let time_embedder = TimeEmbedder::new(vb.pp("time_embedder"), channels * 4)?;

        let mut res_blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let block = ResBlock::new(vb.pp(format!("res_blocks.{}", i)), channels, is_causal)?;
            res_blocks.push(block);
        }

        Ok(Self {
            time_embedder,
            res_blocks,
            channels,
        })
    }

    pub fn forward(&self, x: &Tensor, timestep: Option<&Tensor>) -> Result<Tensor> {
        let batch_size = x.dim(0)?;

        let temb = if let Some(ts) = timestep {
            let emb = self.time_embedder.forward(ts, batch_size)?;
            Some(emb.reshape((batch_size, self.channels * 4, 1, 1, 1))?)
        } else {
            None
        };

        let mut hidden = x.clone();
        for block in &self.res_blocks {
            hidden = block.forward(&hidden, temb.as_ref())?;
        }

        Ok(hidden)
    }
}

// ===========================================================================
// Upsampler
// ===========================================================================

pub struct Upsampler {
    conv: CausalConv3d,
}

impl Upsampler {
    pub fn new(vb: VarBuilder, in_ch: usize, out_ch: usize, is_causal: bool) -> Result<Self> {
        let conv = CausalConv3d::new(vb.pp("conv"), in_ch, out_ch, 3, is_causal)?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.conv.forward(x)?;
        unpatchify(&hidden, 2, 2)
    }
}

// ===========================================================================
// VaeDecoder
// ===========================================================================

/// VAE Decoder for LTX-Video
///
/// Loads weights from embedded format in safetensors.
/// Use `vb.pp("vae.decoder")` when loading.
pub struct VaeDecoder {
    conv_in: CausalConv3d,
    up_block_0: ResBlockGroup,
    up_block_1: Upsampler,
    up_block_2: ResBlockGroup,
    up_block_3: Upsampler,
    up_block_4: ResBlockGroup,
    up_block_5: Upsampler,
    up_block_6: ResBlockGroup,
    norm_out: RMSNorm3d,
    conv_out: CausalConv3d,
    timestep_scale_multiplier: Option<Tensor>,
    last_time_embedder: Option<TimeEmbedder>,
    last_scale_shift_table: Option<Tensor>,
    patch_size: usize,
}

impl VaeDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let is_causal = false;

        let conv_in = CausalConv3d::new(vb.pp("conv_in"), 128, 1024, 3, is_causal)?;

        let up_block_0 = ResBlockGroup::new(vb.pp("up_blocks.0"), 1024, 5, is_causal)?;
        let up_block_1 = Upsampler::new(vb.pp("up_blocks.1"), 1024, 4096, is_causal)?;

        let up_block_2 = ResBlockGroup::new(vb.pp("up_blocks.2"), 512, 5, is_causal)?;
        let up_block_3 = Upsampler::new(vb.pp("up_blocks.3"), 512, 2048, is_causal)?;

        let up_block_4 = ResBlockGroup::new(vb.pp("up_blocks.4"), 256, 5, is_causal)?;
        let up_block_5 = Upsampler::new(vb.pp("up_blocks.5"), 256, 1024, is_causal)?;

        let up_block_6 = ResBlockGroup::new(vb.pp("up_blocks.6"), 128, 5, is_causal)?;

        let norm_out = RMSNorm3d::new(1e-8);
        let conv_out = CausalConv3d::new(vb.pp("conv_out"), 128, 48, 3, is_causal)?;

        let timestep_scale_multiplier = vb.get((), "timestep_scale_multiplier").ok();
        let last_time_embedder = TimeEmbedder::new(vb.pp("last_time_embedder"), 256).ok();
        let last_scale_shift_table = vb.get((2, 128), "last_scale_shift_table").ok();

        Ok(Self {
            conv_in,
            up_block_0,
            up_block_1,
            up_block_2,
            up_block_3,
            up_block_4,
            up_block_5,
            up_block_6,
            norm_out,
            conv_out,
            timestep_scale_multiplier,
            last_time_embedder,
            last_scale_shift_table,
            patch_size: 4,
        })
    }

    pub fn decode(&self, latents: &Tensor, timestep: Option<f64>) -> Result<Tensor> {
        let device = latents.device();
        let batch_size = latents.dim(0)?;

        // Prepare timestep
        let ts: Option<Tensor> = if let Some(t) = timestep {
            Some(Tensor::new(&[t as f32], device)?)
        } else {
            None
        };

        // Scale timestep
        let scaled_ts = if let (Some(t), Some(mult)) = (&ts, &self.timestep_scale_multiplier) {
            let m = mult.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            Some(t.affine(m as f64, 0.0)?)
        } else {
            ts
        };
        let scaled_ts_ref = scaled_ts.as_ref();

        // Forward pass
        let mut hidden = self.conv_in.forward(latents)?;

        hidden = self.up_block_0.forward(&hidden, scaled_ts_ref)?;
        hidden = self.up_block_1.forward(&hidden)?;

        hidden = self.up_block_2.forward(&hidden, scaled_ts_ref)?;
        hidden = self.up_block_3.forward(&hidden)?;

        hidden = self.up_block_4.forward(&hidden, scaled_ts_ref)?;
        hidden = self.up_block_5.forward(&hidden)?;

        hidden = self.up_block_6.forward(&hidden, scaled_ts_ref)?;

        hidden = self.norm_out.forward(&hidden)?;

        // Final modulation
        if let (Some(embedder), Some(table), Some(ts)) = (
            &self.last_time_embedder,
            &self.last_scale_shift_table,
            &scaled_ts,
        ) {
            let temb = embedder.forward(ts, batch_size)?;
            let temb = temb.reshape((batch_size, 2, 128, 1, 1, 1))?;
            let table = table.reshape((1, 2, 128, 1, 1, 1))?;
            let combined = (temb + table)?;

            let shift = combined.i((.., 0, .., .., .., ..))?.squeeze(1)?;
            let scale = combined.i((.., 1, .., .., .., ..))?.squeeze(1)?;

            let ones = Tensor::ones_like(&scale)?;
            let scaled = hidden.broadcast_mul(&(ones + scale)?)?;
            hidden = scaled.broadcast_add(&shift)?;
        }

        hidden = silu(&hidden)?;
        hidden = self.conv_out.forward(&hidden)?;

        // Final unpatchify: 48 -> 3 with 4x4 spatial upscale (decoder-specific permute order)
        unpatchify_decoder(&hidden, self.patch_size, 1)
    }
}

// ===========================================================================
// SpaceToDepthDownsample (Encoder downsampling block)
// ===========================================================================

/// Latent log variance mode for VAE encoding
///
/// Controls how the encoder outputs variance information for the latent distribution.
/// Matches Python: latent_log_var in Encoder class
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum LatentLogVar {
    /// No log variance output, encoder returns only mean
    #[default]
    None,
    /// Per-channel log variance: output has 2x channels (mean + log_var)
    PerChannel,
    /// Uniform log variance: single log_var value shared across channels
    Uniform,
    /// Constant log variance: fixed value (approx -30 for minimal variance)
    Constant,
}

/// Downsampler using space-to-depth transformation
/// Inverse of DepthToSpaceUpsample used in the decoder
#[allow(dead_code)]
pub struct SpaceToDepthDownsample {
    conv: CausalConv3d,
    stride_t: usize,
    stride_h: usize,
    stride_w: usize,
}

impl SpaceToDepthDownsample {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize), // (t, h, w)
        is_causal: bool,
    ) -> Result<Self> {
        // After space-to-depth, channels are multiplied by stride_t * stride_h * stride_w
        let packed_channels = in_channels * stride.0 * stride.1 * stride.2;
        let conv = CausalConv3d::new(vb.pp("conv"), packed_channels, out_channels, 3, is_causal)?;

        Ok(Self {
            conv,
            stride_t: stride.0,
            stride_h: stride.1,
            stride_w: stride.2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Space-to-depth: reduce spatial/temporal dimensions, increase channels
        let packed = patchify(x, self.stride_h, self.stride_t)?;
        self.conv.forward(&packed)
    }
}

// ===========================================================================
// VaeEncoder
// ===========================================================================

/// VAE Encoder for LTX-Video (Image-to-Video conditioning)
///
/// Encodes input images/videos into latent space for conditioning.
/// Architecture mirrors the decoder with downsampling instead of upsampling.
///
/// Use `vb.pp("vae.encoder")` when loading weights.
pub struct VaeEncoder {
    conv_in: CausalConv3d,
    down_block_0: ResBlockGroup,
    down_block_1: SpaceToDepthDownsample,
    down_block_2: ResBlockGroup,
    down_block_3: SpaceToDepthDownsample,
    down_block_4: ResBlockGroup,
    down_block_5: SpaceToDepthDownsample,
    down_block_6: ResBlockGroup,
    norm_out: RMSNorm3d,
    conv_out: CausalConv3d,
    patch_size: usize,
}

impl VaeEncoder {
    /// Create a new VAE encoder
    ///
    /// # Arguments
    /// * `vb` - VarBuilder pointing to encoder weights (e.g., `vb.pp("vae.encoder")`)
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let is_causal = true; // Encoder uses causal convolutions for temporal coherence

        // Initial patchify: 3 -> 48 with 4x4 spatial patchify
        let patch_size = 4;
        let in_channels = 3 * patch_size * patch_size; // 48 after patchify

        let conv_in = CausalConv3d::new(vb.pp("conv_in"), in_channels, 128, 3, is_causal)?;

        // Downsampling blocks (mirror of decoder upsampling)
        let down_block_0 = ResBlockGroup::new(vb.pp("down_blocks.0"), 128, 5, is_causal)?;
        let down_block_1 =
            SpaceToDepthDownsample::new(vb.pp("down_blocks.1"), 128, 256, (2, 2, 2), is_causal)?;

        let down_block_2 = ResBlockGroup::new(vb.pp("down_blocks.2"), 256, 5, is_causal)?;
        let down_block_3 =
            SpaceToDepthDownsample::new(vb.pp("down_blocks.3"), 256, 512, (2, 2, 2), is_causal)?;

        let down_block_4 = ResBlockGroup::new(vb.pp("down_blocks.4"), 512, 5, is_causal)?;
        let down_block_5 =
            SpaceToDepthDownsample::new(vb.pp("down_blocks.5"), 512, 1024, (2, 2, 2), is_causal)?;

        let down_block_6 = ResBlockGroup::new(vb.pp("down_blocks.6"), 1024, 5, is_causal)?;

        let norm_out = RMSNorm3d::new(1e-8);
        // Output: 1024 -> 128 (latent channels)
        let conv_out = CausalConv3d::new(vb.pp("conv_out"), 1024, 128, 3, is_causal)?;

        Ok(Self {
            conv_in,
            down_block_0,
            down_block_1,
            down_block_2,
            down_block_3,
            down_block_4,
            down_block_5,
            down_block_6,
            norm_out,
            conv_out,
            patch_size,
        })
    }

    /// Encode input video/image to latent space
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, 3, T, H, W] in range [-1, 1]
    ///
    /// # Returns
    /// Latent tensor [B, 128, T', H', W'] where T'=T/8, H'=H/32, W'=W/32
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // Initial patchify: [B, 3, T, H, W] -> [B, 48, T, H/4, W/4]
        let hidden = patchify(x, self.patch_size, 1)?;

        // Forward through encoder
        let hidden = self.conv_in.forward(&hidden)?;

        let hidden = self.down_block_0.forward(&hidden, None)?;
        let hidden = self.down_block_1.forward(&hidden)?;

        let hidden = self.down_block_2.forward(&hidden, None)?;
        let hidden = self.down_block_3.forward(&hidden)?;

        let hidden = self.down_block_4.forward(&hidden, None)?;
        let hidden = self.down_block_5.forward(&hidden)?;

        let hidden = self.down_block_6.forward(&hidden, None)?;

        let hidden = self.norm_out.forward(&hidden)?;
        let hidden = silu(&hidden)?;

        self.conv_out.forward(&hidden)
    }
}

/// Encode and normalize latents for use in the diffusion process
///
/// # Arguments
/// * `encoder` - VAE Encoder
/// * `x` - Input tensor [B, 3, T, H, W]
/// * `latents_mean` - Per-channel mean for normalization
/// * `latents_std` - Per-channel standard deviation for normalization
/// * `scaling_factor` - Scaling factor for normalization
pub fn vae_encode(
    encoder: &VaeEncoder,
    x: &Tensor,
    latents_mean: Option<&Tensor>,
    latents_std: Option<&Tensor>,
    scaling_factor: f64,
) -> Result<Tensor> {
    let latents = encoder.encode(x)?;

    // Normalize if mean/std provided
    if let (Some(mean), Some(std)) = (latents_mean, latents_std) {
        let device = latents.device();
        let dtype = latents.dtype();

        // Reshape mean/std from [C] to [1, C, 1, 1, 1] for broadcasting
        let mean = mean
            .reshape((1, (), 1, 1, 1))?
            .to_device(device)?
            .to_dtype(dtype)?;
        let std = std
            .reshape((1, (), 1, 1, 1))?
            .to_device(device)?
            .to_dtype(dtype)?;
        let scale = Tensor::new(&[scaling_factor as f32], device)?.to_dtype(dtype)?;

        // (latents - mean) * scaling_factor / std
        latents
            .broadcast_sub(&mean)?
            .broadcast_mul(&scale)?
            .broadcast_div(&std)
    } else {
        Ok(latents)
    }
}
