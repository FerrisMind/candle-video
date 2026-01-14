pub use crate::ops::conv3d::{Conv3d, Conv3dConfig, PaddingMode};

pub type CausalConv3d = Conv3d;
pub type CausalConv3dConfig = Conv3dConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Module, VarBuilder, VarMap};

    fn create_test_conv3d(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        config: Conv3dConfig,
    ) -> candle_core::Result<Conv3d> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

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
    fn test_conv3d_alias() {
        let config = CausalConv3dConfig::causal((3, 3, 3));
        assert!(config.is_causal);
    }

    #[test]
    fn test_conv3d_forward_basic() -> candle_core::Result<()> {
        let config = Conv3dConfig::new((3, 3, 3)).with_padding((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?;
        let y = conv.forward(&x)?;

        assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_conv3d_forward_causal() -> candle_core::Result<()> {
        let config = Conv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?;
        let y = conv.forward(&x)?;

        assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_conv3d_forward_pointwise() -> candle_core::Result<()> {
        let config = Conv3dConfig::new((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (1, 1, 1), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?;
        let y = conv.forward(&x)?;

        assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_causal_conv3d_alias() -> candle_core::Result<()> {
        let config = CausalConv3dConfig::causal((3, 3, 3)).with_padding((0, 1, 1));

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let _ = vs.get_with_hints(
            (8, 4, 3, 3, 3),
            "weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.1,
            },
        )?;
        let _ = vs.get_with_hints(8, "bias", candle_nn::Init::Const(0.0))?;

        let conv: CausalConv3d = Conv3d::new(4, 8, config, vs)?;

        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        let y = conv.forward(&x)?;

        assert_eq!(y.dims(), &[1, 8, 4, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_conv3d_module_trait() -> candle_core::Result<()> {
        let config = Conv3dConfig::new((3, 3, 3)).with_padding((1, 1, 1));
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config)?;

        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;

        let y = x.apply(&conv)?;
        assert_eq!(y.dims(), &[1, 8, 4, 8, 8]);
        Ok(())
    }
}
