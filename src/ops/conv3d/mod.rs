use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub mod cpu;
pub mod cuda;
pub mod metal;
pub mod padding;

pub use cpu::{
    Im2ColConfig, conv3d_cpu_forward, conv3d_pointwise, conv3d_spatial_only, conv3d_temporal_only,
};
pub use cuda::{
    conv3d_cuda_forward, conv3d_pointwise_cuda, conv3d_spatial_only_cuda,
    conv3d_temporal_only_cuda, is_cuda_device, is_cuda_tensor, is_supported_cuda_dtype,
    supported_cuda_dtypes,
};
pub use metal::{
    conv3d_metal_forward, conv3d_pointwise_metal, conv3d_spatial_only_metal,
    conv3d_temporal_only_metal, is_metal_device, is_metal_tensor, is_supported_metal_dtype,
    supported_metal_dtypes,
};
pub use padding::{
    apply_causal_temporal_padding, apply_full_padding, apply_spatial_padding,
    apply_symmetric_temporal_padding, apply_temporal_padding,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaddingMode {
    #[default]
    Zeros,

    Replicate,
}

#[derive(Clone, Copy, Debug)]
pub struct Conv3dConfig {
    pub kernel: (usize, usize, usize),

    pub stride: (usize, usize, usize),

    pub padding: (usize, usize, usize),

    pub dilation: (usize, usize, usize),

    pub groups: usize,

    pub is_causal: bool,

    pub padding_mode: PaddingMode,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            kernel: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
            is_causal: false,
            padding_mode: PaddingMode::Zeros,
        }
    }
}

impl Conv3dConfig {
    pub fn new(kernel: (usize, usize, usize)) -> Self {
        Self {
            kernel,
            ..Default::default()
        }
    }

    pub fn causal(kernel: (usize, usize, usize)) -> Self {
        Self {
            kernel,
            is_causal: true,
            padding_mode: PaddingMode::Replicate,
            ..Default::default()
        }
    }

    pub fn same_padding(kernel: (usize, usize, usize)) -> Self {
        Self {
            kernel,
            padding: (kernel.0 / 2, kernel.1 / 2, kernel.2 / 2),
            is_causal: false,
            ..Default::default()
        }
    }

    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_dilation(mut self, dilation: (usize, usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    pub fn with_causal(mut self, is_causal: bool) -> Self {
        self.is_causal = is_causal;
        self
    }

    pub fn with_padding_mode(mut self, padding_mode: PaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }
}

pub type CausalConv3dConfig = Conv3dConfig;

#[derive(Debug, Clone)]
pub struct Conv3d {
    weight: Tensor,

    bias: Option<Tensor>,

    config: Conv3dConfig,

    in_channels: usize,

    out_channels: usize,
}

pub type CausalConv3d = Conv3d;

impl Conv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        config: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kt, kh, kw) = config.kernel;
        let groups = config.groups;

        if groups == 0 {
            candle_core::bail!("Conv3d groups must be positive, got groups=0");
        }
        if !in_channels.is_multiple_of(groups) {
            candle_core::bail!(
                "Conv3d in_channels ({}) must be divisible by groups ({}). \
                 in_channels / groups = {} remainder {}",
                in_channels,
                groups,
                in_channels / groups,
                in_channels % groups
            );
        }
        if !out_channels.is_multiple_of(groups) {
            candle_core::bail!(
                "Conv3d out_channels ({}) must be divisible by groups ({}). \
                 out_channels / groups = {} remainder {}",
                out_channels,
                groups,
                out_channels / groups,
                out_channels % groups
            );
        }

        let weight = vb.get((out_channels, in_channels / groups, kt, kh, kw), "weight")?;

        let bias = vb.get(out_channels, "bias").ok();

        Ok(Self {
            weight,
            bias,
            config,
            in_channels,
            out_channels,
        })
    }

    pub fn new_with_kernel(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        config: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv3dConfig { kernel, ..config };
        Self::new(in_channels, out_channels, config, vb)
    }

    pub fn new_with_conv_submodule(
        in_channels: usize,
        out_channels: usize,
        config: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(in_channels, out_channels, config, vb.pp("conv"))
    }

    pub fn kernel_size(&self) -> (usize, usize, usize) {
        self.config.kernel
    }

    pub fn stride(&self) -> (usize, usize, usize) {
        self.config.stride
    }

    pub fn padding(&self) -> (usize, usize, usize) {
        self.config.padding
    }

    pub fn dilation(&self) -> (usize, usize, usize) {
        self.config.dilation
    }

    pub fn groups(&self) -> usize {
        self.config.groups
    }

    pub fn is_causal(&self) -> bool {
        self.config.is_causal
    }

    pub fn padding_mode(&self) -> PaddingMode {
        self.config.padding_mode
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn output_dims(
        &self,
        input_t: usize,
        input_h: usize,
        input_w: usize,
    ) -> Result<(usize, usize, usize)> {
        let (kt, kh, kw) = self.config.kernel;
        let (st, sh, sw) = self.config.stride;
        let (pt, ph, pw) = self.config.padding;
        let (dt, dh, dw) = self.config.dilation;

        let effective_pt = if self.config.is_causal {
            kt - 1
        } else {
            pt * 2
        };

        let calc_out = |dim_name: &str,
                        input: usize,
                        pad: usize,
                        dil: usize,
                        k: usize,
                        s: usize|
         -> Result<usize> {
            let dilated_k = dil * (k - 1) + 1;
            let padded = input + pad;
            if padded < dilated_k {
                candle_core::bail!(
                    "Output {} dimension would be negative or zero: input_size={}, total_padding={}, \
                     dilated_kernel_size={} (kernel={}, dilation={}), stride={}. \
                     Formula: (input + padding - dilated_kernel) / stride + 1 = ({} + {} - {}) / {} + 1",
                    dim_name,
                    input,
                    pad,
                    dilated_k,
                    k,
                    dil,
                    s,
                    input,
                    pad,
                    dilated_k,
                    s
                );
            }
            let out = (padded - dilated_k) / s + 1;
            if out == 0 {
                candle_core::bail!(
                    "Output {} dimension would be zero: input_size={}, total_padding={}, \
                     dilated_kernel_size={}, stride={}",
                    dim_name,
                    input,
                    pad,
                    dilated_k,
                    s
                );
            }
            Ok(out)
        };

        let t_out = calc_out("temporal", input_t, effective_pt, dt, kt, st)?;
        let h_out = calc_out("height", input_h, ph * 2, dh, kh, sh)?;
        let w_out = calc_out("width", input_w, pw * 2, dw, kw, sw)?;

        Ok((t_out, h_out, w_out))
    }

    fn validate_input(&self, x: &Tensor) -> Result<(usize, usize, usize, usize, usize)> {
        let dims = x.dims();
        if dims.len() != 5 {
            candle_core::bail!(
                "Conv3d expects 5D input tensor (batch, channels, time, height, width), \
                 got {}D tensor with shape {:?}. \
                 Please reshape your input to have 5 dimensions.",
                dims.len(),
                dims
            );
        }

        let (b, c, t, h, w) = x.dims5()?;

        if c != self.in_channels {
            candle_core::bail!(
                "Input channels mismatch: input tensor has {} channels, \
                 but Conv3d expects {} channels (in_channels={}, groups={}). \
                 Weight shape is {:?}.",
                c,
                self.in_channels,
                self.in_channels,
                self.config.groups,
                self.weight.dims()
            );
        }

        Ok((b, c, t, h, w))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, t, h, w) = self.validate_input(x)?;

        let (t_out, h_out, w_out) = self.output_dims(t, h, w)?;

        let x_padded = self.apply_padding(x)?;

        let y = self.conv3d_forward(&x_padded, b, t_out, h_out, w_out)?;

        if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, self.out_channels, 1, 1, 1))?;
            y.broadcast_add(&bias)
        } else {
            Ok(y)
        }
    }

    fn apply_padding(&self, x: &Tensor) -> Result<Tensor> {
        let (kt, kh, kw) = self.config.kernel;
        let (pt, ph, pw) = self.config.padding;

        if self.config.is_causal {
            padding::apply_full_padding(
                x,
                (kt, kh, kw),
                (0, ph, pw),
                true,
                self.config.padding_mode,
            )
        } else {
            let x = if pt > 0 {
                padding::apply_symmetric_temporal_padding(x, pt, self.config.padding_mode)?
            } else {
                x.clone()
            };

            padding::apply_spatial_padding(&x, kh, kw, ph, pw, self.config.padding_mode)
        }
    }

    #[allow(dead_code)]
    fn apply_causal_temporal_padding(&self, x: &Tensor, kt: usize) -> Result<Tensor> {
        padding::apply_causal_temporal_padding(x, kt, self.config.padding_mode)
    }

    #[allow(dead_code)]
    fn apply_symmetric_temporal_padding(&self, x: &Tensor, pt: usize) -> Result<Tensor> {
        padding::apply_symmetric_temporal_padding(x, pt, self.config.padding_mode)
    }

    #[allow(dead_code)]
    fn apply_spatial_padding(&self, x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
        let (_, kh, kw) = self.config.kernel;
        padding::apply_spatial_padding(x, kh, kw, ph, pw, self.config.padding_mode)
    }

    fn conv3d_forward(
        &self,
        x: &Tensor,
        _batch: usize,
        t_out: usize,
        h_out: usize,
        w_out: usize,
    ) -> Result<Tensor> {
        let (kt, kh, kw) = self.config.kernel;
        let (st, sh, sw) = self.config.stride;
        let (dt, dh, dw) = self.config.dilation;
        let groups = self.config.groups;

        let is_cuda = cuda::is_cuda_tensor(x);
        let is_metal = metal::is_metal_tensor(x);

        if groups == 1 {
            if kt == 1 && kh == 1 && kw == 1 && st == 1 && sh == 1 && sw == 1 {
                return if is_cuda {
                    cuda::conv3d_pointwise_cuda(x, &self.weight, None)
                } else if is_metal {
                    metal::conv3d_pointwise_metal(x, &self.weight, None)
                } else {
                    cpu::conv3d_pointwise(x, &self.weight, None)
                };
            }

            if kt > 1 && kh == 1 && kw == 1 {
                return if is_cuda {
                    cuda::conv3d_temporal_only_cuda(
                        x,
                        &self.weight,
                        None,
                        kt,
                        st,
                        dt,
                        t_out,
                        sh,
                        sw,
                        h_out,
                        w_out,
                        groups,
                    )
                } else if is_metal {
                    metal::conv3d_temporal_only_metal(
                        x,
                        &self.weight,
                        None,
                        kt,
                        st,
                        dt,
                        t_out,
                        sh,
                        sw,
                        h_out,
                        w_out,
                        groups,
                    )
                } else {
                    cpu::conv3d_temporal_only(
                        x,
                        &self.weight,
                        None,
                        kt,
                        st,
                        dt,
                        t_out,
                        sh,
                        sw,
                        h_out,
                        w_out,
                    )
                };
            }

            if kt == 1 && (kh > 1 || kw > 1) {
                return if is_cuda {
                    cuda::conv3d_spatial_only_cuda(
                        x,
                        &self.weight,
                        None,
                        kh,
                        kw,
                        sh,
                        sw,
                        dh,
                        dw,
                        h_out,
                        w_out,
                        st,
                        t_out,
                        groups,
                    )
                } else if is_metal {
                    metal::conv3d_spatial_only_metal(
                        x,
                        &self.weight,
                        None,
                        kh,
                        kw,
                        sh,
                        sw,
                        dh,
                        dw,
                        h_out,
                        w_out,
                        st,
                        t_out,
                        groups,
                    )
                } else {
                    cpu::conv3d_spatial_only(
                        x,
                        &self.weight,
                        None,
                        kh,
                        kw,
                        sh,
                        sw,
                        dh,
                        dw,
                        h_out,
                        w_out,
                        st,
                        t_out,
                    )
                };
            }
        }

        let config =
            cpu::Im2ColConfig::new(self.config.kernel, self.config.stride, self.config.dilation);

        if is_cuda {
            cuda::conv3d_cuda_forward(x, &self.weight, None, &config, groups, t_out, h_out, w_out)
        } else if is_metal {
            metal::conv3d_metal_forward(x, &self.weight, None, &config, groups, t_out, h_out, w_out)
        } else {
            cpu::conv3d_cpu_forward(x, &self.weight, None, &config, groups, t_out, h_out, w_out)
        }
    }

    #[allow(dead_code)]
    fn conv3d_pointwise(
        &self,
        x: &Tensor,
        _batch: usize,
        _t_out: usize,
        _h_out: usize,
        _w_out: usize,
    ) -> Result<Tensor> {
        cpu::conv3d_pointwise(x, &self.weight, self.bias.as_ref())
    }

    #[allow(dead_code)]
    fn im2col_3d(&self, x: &Tensor, t_out: usize, h_out: usize, w_out: usize) -> Result<Tensor> {
        let config =
            cpu::Im2ColConfig::new(self.config.kernel, self.config.stride, self.config.dilation);
        cpu::im2col_3d(x, &config, t_out, h_out, w_out)
    }
}

impl Conv3d {
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if !self.config.is_causal {
            return Ok((self.forward(x)?, None));
        }

        let (kt, kh, kw) = self.config.kernel;
        let (b, c, _t, h, w) = self.validate_input(x)?;
        let (ph, pw) = (self.config.padding.1, self.config.padding.2);

        if kt <= 1 {
            return Ok((self.forward(x)?, None));
        }

        let cache_frames = kt - 1;

        let x_with_cache = match cache {
            Some(cache_tensor) => {
                let cache_dims = cache_tensor.dims();
                if cache_dims.len() != 5 {
                    candle_core::bail!(
                        "Cache tensor must be 5D (batch, channels, frames, height, width), \
                         got {}D tensor with shape {:?}. \
                         Expected shape: ({}, {}, {}, {}, {})",
                        cache_dims.len(),
                        cache_dims,
                        b,
                        c,
                        cache_frames,
                        h,
                        w
                    );
                }
                let (cb, cc, ct, ch, cw) = cache_tensor.dims5()?;

                if cb != b {
                    candle_core::bail!(
                        "Cache batch size mismatch: cache has batch={}, but input has batch={}. \
                         Cache shape: {:?}, Input shape: {:?}",
                        cb,
                        b,
                        cache_dims,
                        x.dims()
                    );
                }
                if cc != c {
                    candle_core::bail!(
                        "Cache channels mismatch: cache has {} channels, but input has {} channels. \
                         Cache shape: {:?}, Input shape: {:?}",
                        cc,
                        c,
                        cache_dims,
                        x.dims()
                    );
                }
                if ch != h {
                    candle_core::bail!(
                        "Cache height mismatch: cache has height={}, but input has height={}. \
                         Cache shape: {:?}, Input shape: {:?}",
                        ch,
                        h,
                        cache_dims,
                        x.dims()
                    );
                }
                if cw != w {
                    candle_core::bail!(
                        "Cache width mismatch: cache has width={}, but input has width={}. \
                         Cache shape: {:?}, Input shape: {:?}",
                        cw,
                        w,
                        cache_dims,
                        x.dims()
                    );
                }

                let _ = ct != cache_frames;

                let cache_on_device = cache_tensor.to_device(x.device())?;
                Tensor::cat(&[&cache_on_device, x], 2)?
            }
            None => {
                let first_frame = x.i((.., .., 0..1, .., ..))?;
                let padding = first_frame.broadcast_as((b, c, cache_frames, h, w))?;
                Tensor::cat(&[&padding.contiguous()?, x], 2)?
            }
        };

        let t_with_cache = x_with_cache.dims5()?.2;
        let (t_out, h_out, w_out) = {
            let (st, sh, sw) = self.config.stride;
            let (dt, dh, dw) = self.config.dilation;

            let calc_out = |dim_name: &str,
                            input: usize,
                            pad: usize,
                            dil: usize,
                            k: usize,
                            s: usize|
             -> Result<usize> {
                let dilated_k = dil * (k - 1) + 1;
                let padded = input + pad;
                if padded < dilated_k {
                    candle_core::bail!(
                        "Output {} dimension would be negative: input_size={}, padding={}, \
                         dilated_kernel_size={} (kernel={}, dilation={})",
                        dim_name,
                        input,
                        pad,
                        dilated_k,
                        k,
                        dil
                    );
                }
                Ok((padded - dilated_k) / s + 1)
            };

            let t_out = calc_out("temporal", t_with_cache, 0, dt, kt, st)?;
            let h_out = calc_out("height", h, ph * 2, dh, kh, sh)?;
            let w_out = calc_out("width", w, pw * 2, dw, kw, sw)?;

            (t_out, h_out, w_out)
        };

        let x_padded = if ph > 0 || pw > 0 {
            padding::apply_spatial_padding(&x_with_cache, kh, kw, ph, pw, self.config.padding_mode)?
        } else {
            x_with_cache.clone()
        };

        let y = self.conv3d_forward(&x_padded, b, t_out, h_out, w_out)?;

        let output = if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, self.out_channels, 1, 1, 1))?;
            y.broadcast_add(&bias)?
        } else {
            y
        };

        let new_cache = if cache_frames > 0 {
            let combined_t = x_with_cache.dims5()?.2;
            if combined_t >= cache_frames {
                let cache_start = combined_t - cache_frames;
                Some(
                    x_with_cache
                        .i((.., .., cache_start.., .., ..))?
                        .contiguous()?,
                )
            } else {
                let first_frame = x_with_cache.i((.., .., 0..1, .., ..))?;
                let pad_count = cache_frames - combined_t;
                let padding = first_frame.broadcast_as((b, c, pad_count, h, w))?;
                Some(Tensor::cat(&[&padding.contiguous()?, &x_with_cache], 2)?.contiguous()?)
            }
        } else {
            None
        };

        Ok((output, new_cache))
    }
}

impl Module for Conv3d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Conv3d::forward(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn create_test_conv3d(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        config: Conv3dConfig,
    ) -> Result<Conv3d> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let (kt, kh, kw) = kernel;
        let groups = config.groups;

        let weight_shape = (out_channels, in_channels / groups, kt, kh, kw);
        let _ = vs.get_with_hints(
            weight_shape,
            "weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.1,
            },
        )?;

        let _ = vs.get_with_hints(out_channels, "bias", candle_nn::Init::Const(0.0))?;

        Conv3d::new(in_channels, out_channels, config, vs)
    }

    #[test]
    fn test_conv3d_config_default() {
        let config = Conv3dConfig::default();
        assert!(!config.is_causal);
        assert_eq!(config.kernel, (1, 1, 1));
        assert_eq!(config.stride, (1, 1, 1));
        assert_eq!(config.padding, (0, 0, 0));
        assert_eq!(config.groups, 1);
        assert_eq!(config.padding_mode, PaddingMode::Zeros);
    }

    #[test]
    fn test_conv3d_config_causal() {
        let config = Conv3dConfig::causal((3, 3, 3));
        assert!(config.is_causal);
        assert_eq!(config.kernel, (3, 3, 3));
        assert_eq!(config.padding_mode, PaddingMode::Replicate);
    }

    #[test]
    fn test_conv3d_config_same_padding() {
        let config = Conv3dConfig::same_padding((3, 3, 3));
        assert_eq!(config.padding, (1, 1, 1));
        assert!(!config.is_causal);
    }

    #[test]
    fn test_conv3d_config_builder() {
        let config = Conv3dConfig::new((3, 3, 3))
            .with_stride((2, 2, 2))
            .with_padding((1, 1, 1))
            .with_dilation((1, 1, 1))
            .with_groups(2)
            .with_causal(true)
            .with_padding_mode(PaddingMode::Replicate);

        assert_eq!(config.kernel, (3, 3, 3));
        assert_eq!(config.stride, (2, 2, 2));
        assert_eq!(config.padding, (1, 1, 1));
        assert_eq!(config.groups, 2);
        assert!(config.is_causal);
        assert_eq!(config.padding_mode, PaddingMode::Replicate);
    }

    #[test]
    fn test_conv3d_output_dims() -> Result<()> {
        let config = Conv3dConfig::new((3, 3, 3)).with_padding((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let (t_out, h_out, w_out) = conv.output_dims(8, 16, 16)?;
        assert_eq!((t_out, h_out, w_out), (8, 16, 16));

        Ok(())
    }

    #[test]
    fn test_conv3d_output_dims_stride() -> Result<()> {
        let config = Conv3dConfig::new((3, 3, 3))
            .with_stride((2, 2, 2))
            .with_padding((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let (t_out, h_out, w_out) = conv.output_dims(8, 16, 16)?;
        assert_eq!((t_out, h_out, w_out), (4, 8, 8));

        Ok(())
    }

    #[test]
    fn test_conv3d_output_dims_causal() -> Result<()> {
        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let (t_out, h_out, w_out) = conv.output_dims(8, 16, 16)?;
        assert_eq!((t_out, h_out, w_out), (8, 16, 16));

        Ok(())
    }

    #[test]
    fn test_conv3d_forward_shape() -> Result<()> {
        let config = Conv3dConfig::new((3, 3, 3)).with_padding((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?;
        let y = conv.forward(&x)?;

        assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_conv3d_forward_pointwise() -> Result<()> {
        let config = Conv3dConfig::new((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (1, 1, 1), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?;
        let y = conv.forward(&x)?;

        assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_conv3d_forward_causal() -> Result<()> {
        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?;
        let y = conv.forward(&x)?;

        assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_conv3d_invalid_input_dims() -> Result<()> {
        let config = Conv3dConfig::new((3, 3, 3));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16), &device)?;
        let result = conv.forward(&x);

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_conv3d_invalid_channels() -> Result<()> {
        let config = Conv3dConfig::new((3, 3, 3));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 8, 8, 16, 16), &device)?;
        let result = conv.forward(&x);

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_conv3d_new_with_conv_submodule() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = Conv3dConfig::new((3, 3, 3)).with_padding((1, 1, 1));
        let (kt, kh, kw) = config.kernel;

        let _ = vs.pp("conv").get_with_hints(
            (8, 4, kt, kh, kw),
            "weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.1,
            },
        )?;
        let _ = vs
            .pp("conv")
            .get_with_hints(8, "bias", candle_nn::Init::Const(0.0))?;

        let conv = Conv3d::new_with_conv_submodule(4, 8, config, vs)?;

        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        let y = conv.forward(&x)?;
        assert_eq!(y.dims(), &[1, 8, 4, 8, 8]);

        Ok(())
    }

    #[test]
    fn test_conv3d_alias() {
        let config = CausalConv3dConfig::causal((3, 3, 3));
        assert!(config.is_causal);
    }

    #[test]
    fn test_forward_with_cache_non_causal_returns_none() -> Result<()> {
        let config = Conv3dConfig::new((3, 3, 3)).with_padding((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        let (output, cache) = conv.forward_with_cache(&x, None)?;

        assert_eq!(output.dims(), &[1, 8, 4, 8, 8]);
        assert!(cache.is_none());
        Ok(())
    }

    #[test]
    fn test_forward_with_cache_kt1_returns_none() -> Result<()> {
        let config = Conv3dConfig::causal((1, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (1, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        let (output, cache) = conv.forward_with_cache(&x, None)?;

        assert_eq!(output.dims(), &[1, 8, 4, 8, 8]);
        assert!(cache.is_none());
        Ok(())
    }

    #[test]
    fn test_forward_with_cache_first_call_no_cache() -> Result<()> {
        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        let (output, cache) = conv.forward_with_cache(&x, None)?;

        assert_eq!(output.dims(), &[1, 8, 4, 8, 8]);

        assert!(cache.is_some());
        let cache = cache.unwrap();
        assert_eq!(cache.dims(), &[1, 4, 2, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_forward_with_cache_subsequent_call() -> Result<()> {
        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;

        let x1 = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        let (_, cache1) = conv.forward_with_cache(&x1, None)?;

        let x2 = Tensor::randn(0f32, 1.0, (1, 4, 2, 8, 8), &device)?;
        let (output2, cache2) = conv.forward_with_cache(&x2, cache1.as_ref())?;

        assert_eq!(output2.dims(), &[1, 8, 2, 8, 8]);

        assert!(cache2.is_some());
        let cache2 = cache2.unwrap();
        assert_eq!(cache2.dims(), &[1, 4, 2, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_forward_with_cache_single_frame() -> Result<()> {
        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;

        let x1 = Tensor::randn(0f32, 1.0, (1, 4, 1, 8, 8), &device)?;
        let (output1, cache1) = conv.forward_with_cache(&x1, None)?;

        assert_eq!(output1.dims(), &[1, 8, 1, 8, 8]);
        assert!(cache1.is_some());

        let x2 = Tensor::randn(0f32, 1.0, (1, 4, 1, 8, 8), &device)?;
        let (output2, cache2) = conv.forward_with_cache(&x2, cache1.as_ref())?;

        assert_eq!(output2.dims(), &[1, 8, 1, 8, 8]);
        assert!(cache2.is_some());
        Ok(())
    }

    #[test]
    fn test_forward_with_cache_invalid_cache_shape() -> Result<()> {
        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;

        let bad_cache = Tensor::randn(0f32, 1.0, (2, 4, 2, 8, 8), &device)?;
        let result = conv.forward_with_cache(&x, Some(&bad_cache));
        assert!(result.is_err());

        let bad_cache = Tensor::randn(0f32, 1.0, (1, 8, 2, 8, 8), &device)?;
        let result = conv.forward_with_cache(&x, Some(&bad_cache));
        assert!(result.is_err());

        let bad_cache = Tensor::randn(0f32, 1.0, (1, 4, 2, 16, 8), &device)?;
        let result = conv.forward_with_cache(&x, Some(&bad_cache));
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_forward_with_cache_cache_shape() -> Result<()> {
        let device = Device::Cpu;

        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        let (_, cache) = conv.forward_with_cache(&x, None)?;
        assert_eq!(cache.unwrap().dims(), &[1, 4, 2, 8, 8]);

        let config = Conv3dConfig::causal((5, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (5, 3, 3), config)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 8, 8, 8), &device)?;
        let (_, cache) = conv.forward_with_cache(&x, None)?;
        assert_eq!(cache.unwrap().dims(), &[1, 4, 4, 8, 8]);

        Ok(())
    }
}
