//! UMT5-XXL text encoder for Wan video models (safetensors or GGUF).

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5;

use super::quantized_umt5_encoder::QuantizedUmt5Encoder;
use super::umt5::Umt5EncoderModel as Umt5EncoderStack;

use crate::engine::backends::TextEncoderBackend;
use crate::models::ltx_video::loader::{LoaderError, WeightLoader};
use crate::models::ltx_video::t2v_pipeline::TextEncoder as PipelineTextEncoder;

use super::configs::WanTextEncoderConfig;
use super::loader::{WanLayout, discover_safetensors};

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

        let candle_cfg = config.to_candle_t5_config();
        let model = Umt5EncoderStack::load(vb, &candle_cfg).map_err(LoaderError::Candle)?;

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
