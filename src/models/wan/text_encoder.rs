//! UMT5-XXL text encoder for Wan video models (safetensors or GGUF).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_nn::var_builder::SimpleBackend;
use candle_transformers::models::t5;

use super::quantized_umt5_encoder::QuantizedUmt5Encoder;
use super::umt5::Umt5EncoderModel as Umt5EncoderStack;

use crate::engine::backends::TextEncoderBackend;
use crate::models::ltx_video::loader::{LoaderError, WeightLoader};
use crate::models::ltx_video::t2v_pipeline::TextEncoder as PipelineTextEncoder;

use super::configs::WanTextEncoderConfig;
use super::loader::{WanLayout, discover_safetensors};

struct ScaledFp8Safetensors {
    tensors: candle_core::safetensors::MmapedSafetensors,
    scales: HashMap<String, String>,
}

impl ScaledFp8Safetensors {
    fn open(path: &Path) -> std::result::Result<Self, LoaderError> {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(LoaderError::Candle)?;
        let entries = tensors.tensors();
        let names: HashSet<&str> = entries.iter().map(|(name, _)| name.as_str()).collect();
        let mut scales = HashMap::new();

        for (name, view) in &entries {
            if !is_fp8_dtype(view.dtype()) || name == "scaled_fp8" {
                continue;
            }
            let stem = name.strip_suffix(".weight").ok_or_else(|| {
                LoaderError::Candle(candle_core::Error::Msg(format!(
                    "scaled FP8 tensor `{name}` is not a linear `.weight` tensor"
                )))
            })?;
            let scale_name = format!("{stem}.scale_weight");
            if !names.contains(scale_name.as_str()) {
                return Err(LoaderError::Candle(candle_core::Error::Msg(format!(
                    "missing FP8 scale `{scale_name}` for tensor `{name}`"
                ))));
            }
            let scale_view = tensors.get(&scale_name).map_err(LoaderError::Candle)?;
            if !scale_view.shape().is_empty() || !is_f32_dtype(scale_view.dtype()) {
                return Err(LoaderError::Candle(candle_core::Error::Msg(format!(
                    "FP8 scale `{scale_name}` must be scalar F32, got {:?} {:?}",
                    scale_view.dtype(),
                    scale_view.shape()
                ))));
            }
            scales.insert(name.clone(), scale_name);
        }

        for (name, _) in entries
            .iter()
            .filter(|(name, _)| name.ends_with(".scale_weight"))
        {
            let weight_name = format!("{}.weight", name.trim_end_matches(".scale_weight"));
            if !scales.contains_key(&weight_name) {
                return Err(LoaderError::Candle(candle_core::Error::Msg(format!(
                    "unexpected FP8 scale `{name}` without scaled tensor `{weight_name}`"
                ))));
            }
        }

        if scales.is_empty() {
            return Err(LoaderError::Candle(candle_core::Error::Msg(format!(
                "{} contains no scaled FP8 tensors",
                path.display()
            ))));
        }

        Ok(Self { tensors, scales })
    }

    fn load(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let tensor = self.tensors.load(name, device)?.to_dtype(dtype)?;
        match self.scales.get(name) {
            Some(scale_name) => {
                let scale = self.tensors.load(scale_name, device)?.to_dtype(dtype)?;
                tensor.broadcast_mul(&scale)
            }
            None => Ok(tensor),
        }
    }
}

fn is_fp8_dtype(dtype: impl std::fmt::Debug) -> bool {
    matches!(format!("{dtype:?}").as_str(), "F8_E4M3" | "F8E4M3")
}

fn is_f32_dtype(dtype: impl std::fmt::Debug) -> bool {
    matches!(format!("{dtype:?}").as_str(), "F32" | "F32E")
}

impl SimpleBackend for ScaledFp8Safetensors {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let tensor = self.load(name, dtype, device)?;
        if tensor.shape() != &shape {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for scaled FP8 UMT5 tensor `{name}`"),
                expected: shape,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        self.load(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.get(name).is_ok()
    }
}

fn is_scaled_fp8_safetensors(path: &Path) -> std::result::Result<bool, LoaderError> {
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
        .map_err(LoaderError::Candle)?;
    Ok(tensors.get("scaled_fp8").is_ok()
        || tensors
            .tensors()
            .iter()
            .any(|(name, _)| name.ends_with(".scale_weight")))
}

fn scaled_fp8_var_builder<'a>(
    path: &Path,
    device: &'a Device,
    dtype: DType,
) -> std::result::Result<VarBuilder<'a>, LoaderError> {
    let backend = ScaledFp8Safetensors::open(path)?;
    Ok(VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

/// UMT5 encoder configuration for Wan 2.1 T2V 1.3B.
#[derive(Debug, Clone)]
pub struct Umt5EncoderConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub layer_norm_epsilon: f64,
    pub dropout_rate: f64,
}

impl From<&WanTextEncoderConfig> for Umt5EncoderConfig {
    fn from(cfg: &WanTextEncoderConfig) -> Self {
        Self {
            d_model: cfg.d_model,
            d_ff: cfg.d_ff,
            d_kv: cfg.d_kv,
            num_heads: cfg.num_heads,
            num_layers: cfg.num_layers,
            vocab_size: cfg.vocab_size,
            layer_norm_epsilon: cfg.layer_norm_epsilon,
            dropout_rate: cfg.dropout_rate,
        }
    }
}

impl Umt5EncoderConfig {
    pub fn wan21_t2v_13b() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            vocab_size: 256_384,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.1,
        }
    }

    pub fn to_candle_t5_config(&self) -> t5::Config {
        t5::Config {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            d_kv: self.d_kv,
            d_ff: self.d_ff,
            num_layers: self.num_layers,
            num_decoder_layers: None,
            num_heads: self.num_heads,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: self.dropout_rate,
            layer_norm_epsilon: self.layer_norm_epsilon,
            initializer_factor: 1.0,
            feed_forward_proj: t5::ActivationWithOptionalGating {
                gated: true,
                activation: candle_nn::Activation::NewGelu,
            },
            tie_word_embeddings: false,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: Some(0),
        }
    }
}

enum Umt5Backend {
    Safetensors { model: Umt5EncoderStack },
    Gguf(QuantizedUmt5Encoder),
}

/// Loaded UMT5 encoder for Wan prompt conditioning.
pub struct Umt5TextEncoder {
    config: Umt5EncoderConfig,
    backend: Umt5Backend,
    device: Device,
    dtype: DType,
}

impl Umt5TextEncoder {
    pub fn load(
        text_encoder_dir: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> std::result::Result<Self, LoaderError> {
        let dir = text_encoder_dir.as_ref();
        let config_path = dir.join("config.json");
        let wan_cfg: WanTextEncoderConfig =
            crate::models::ltx_video::loader::load_model_config(&config_path)?;
        let config = Umt5EncoderConfig::from(&wan_cfg);

        let shards = discover_safetensors(dir)?;
        let candle_cfg = config.to_candle_t5_config();
        let model = if shards.len() == 1 && is_scaled_fp8_safetensors(&shards[0])? {
            let vb = scaled_fp8_var_builder(&shards[0], device, dtype)?;
            Umt5EncoderStack::load(vb, &candle_cfg).map_err(LoaderError::Candle)?
        } else {
            let loader = WeightLoader::new(device.clone(), dtype);
            let vb = if shards.len() == 1 {
                loader
                    .load_single(&shards[0])
                    .map_err(LoaderError::Candle)?
            } else {
                loader
                    .load_sharded(shards.as_slice())
                    .map_err(LoaderError::Candle)?
            };
            Umt5EncoderStack::load(vb, &candle_cfg).map_err(LoaderError::Candle)?
        };

        Ok(Self {
            config,
            backend: Umt5Backend::Safetensors { model },
            device: device.clone(),
            dtype,
        })
    }

    /// Load from consolidated `text_encoder_gguf/*.gguf` (Q5_K_M, Q8_0, …).
    pub fn load_gguf(
        gguf_path: impl AsRef<Path>,
        device: &Device,
    ) -> std::result::Result<Self, LoaderError> {
        let config = Umt5EncoderConfig::wan21_t2v_13b();
        let model =
            QuantizedUmt5Encoder::load(gguf_path.as_ref(), device).map_err(LoaderError::Candle)?;
        Ok(Self {
            config,
            backend: Umt5Backend::Gguf(model),
            device: device.clone(),
            dtype: DType::F32,
        })
    }

    /// Load text encoder for either Diffusers or consolidated Wan layout.
    pub fn load_for_layout(
        layout: &WanLayout,
        text_encoder_device: &Device,
        transformer_dtype: DType,
    ) -> std::result::Result<Self, LoaderError> {
        match layout {
            WanLayout::Diffusers(paths) => {
                let te_dtype = if text_encoder_device.is_cuda() {
                    transformer_dtype
                } else if transformer_dtype == DType::F16 {
                    DType::F16
                } else {
                    DType::F32
                };
                Self::load(
                    paths.root.join("text_encoder"),
                    text_encoder_device,
                    te_dtype,
                )
            }
            WanLayout::Consolidated(paths) => {
                Self::load_gguf(&paths.text_encoder_gguf, text_encoder_device)
            }
        }
    }

    pub fn from_var_builder(
        config: Umt5EncoderConfig,
        vb: VarBuilder,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let candle_cfg = config.to_candle_t5_config();
        let model = Umt5EncoderStack::load(vb, &candle_cfg)?;
        Ok(Self {
            config,
            backend: Umt5Backend::Safetensors { model },
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &Umt5EncoderConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.backend, Umt5Backend::Gguf(_))
    }

    pub fn forward_hidden_states(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match &mut self.backend {
            Umt5Backend::Safetensors { model } => model.forward(input_ids),
            Umt5Backend::Gguf(m) => m.forward(input_ids, None),
        }
    }

    /// Diffusers-style prompt embeddings: run encoder on trimmed ids (no pad tokens in
    /// attention), then zero-pad to `max_sequence_length`.
    pub fn encode_padded_prompt_embeds(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        max_sequence_length: usize,
    ) -> Result<Tensor> {
        match &mut self.backend {
            Umt5Backend::Gguf(m) => {
                m.encode_padded_prompt_embeds(input_ids, attention_mask, max_sequence_length)
            }
            Umt5Backend::Safetensors { .. } => {
                self.encode_padded_prompt_embeds_fp(input_ids, attention_mask, max_sequence_length)
            }
        }
    }

    fn encode_padded_prompt_embeds_fp(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        max_sequence_length: usize,
    ) -> Result<Tensor> {
        let batch = input_ids.dim(0)?;
        let mut rows = Vec::with_capacity(batch);

        for b in 0..batch {
            let mask_row = attention_mask.i((b, ..))?;
            let seq_len = mask_row
                .to_vec1::<f32>()?
                .into_iter()
                .filter(|&v| v > 0.0)
                .count();
            let seq_len = seq_len.min(max_sequence_length);

            let trimmed_ids = input_ids.i((b, ..seq_len))?.reshape((1, seq_len))?;
            let hidden = self.forward_hidden_states(&trimmed_ids)?;
            let dim = hidden.dim(2)?;
            let trimmed = hidden.i((0, .., ..))?;

            let pad_len = max_sequence_length.saturating_sub(seq_len);
            let row = if pad_len > 0 {
                let pad = Tensor::zeros((pad_len, dim), hidden.dtype(), hidden.device())?;
                Tensor::cat(&[&trimmed, &pad], 0)?
            } else {
                trimmed
            };
            rows.push(row);
        }

        Tensor::stack(&rows, 0)
    }
}

impl TextEncoderBackend for Umt5TextEncoder {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn forward(&mut self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_hidden_states(input_ids)
    }
}

impl PipelineTextEncoder for Umt5TextEncoder {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_hidden_states(input_ids)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn scaled_fp8_backend_applies_scalar_weight_scale() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("scaled.safetensors");
        let device = Device::Cpu;
        let weight = Tensor::new(&[1.0f32, 2.0], &device)
            .expect("weight")
            .to_dtype(DType::F8E4M3)
            .expect("fp8");
        let scale = Tensor::new(2.0f32, &device).expect("scale");
        candle_core::safetensors::save(
            &HashMap::from([
                ("encoder.block.0.q.weight".to_string(), weight),
                ("encoder.block.0.q.scale_weight".to_string(), scale),
            ]),
            &path,
        )
        .expect("save");

        let vb = scaled_fp8_var_builder(&path, &device, DType::F32).expect("var builder");
        let values = vb
            .get((2,), "encoder.block.0.q.weight")
            .expect("scaled weight")
            .to_vec1::<f32>()
            .expect("values");

        assert_eq!(values, vec![2.0, 4.0]);
    }

    #[test]
    fn scaled_fp8_backend_rejects_missing_scale() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("missing-scale.safetensors");
        let device = Device::Cpu;
        let weight = Tensor::new(&[1.0f32], &device)
            .expect("weight")
            .to_dtype(DType::F8E4M3)
            .expect("fp8");
        candle_core::safetensors::save(
            &HashMap::from([("encoder.block.0.q.weight".to_string(), weight)]),
            &path,
        )
        .expect("save");

        let error = match scaled_fp8_var_builder(&path, &device, DType::F32) {
            Ok(_) => panic!("missing scale must fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("missing FP8 scale"));
    }
}
