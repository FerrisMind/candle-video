//! Quantized UMT5-XXL encoder from GGUF (city96 / Wan consolidated bundles).

use candle_core::{Device, IndexOp, Result, Tensor};

use crate::models::ltx_video::quantized_t5_encoder::{QuantizedT5EncoderModel, T5EncoderConfig};

/// GGUF-backed UMT5 encoder for Wan prompt conditioning.
pub struct QuantizedUmt5Encoder {
    inner: QuantizedT5EncoderModel,
}

impl QuantizedUmt5Encoder {
    pub fn load(gguf_path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let inner = QuantizedT5EncoderModel::load_with_config(
            gguf_path,
            device,
            T5EncoderConfig::umt5_xxl(),
        )?;
        Ok(Self { inner })
    }

    pub fn d_model(&self) -> usize {
        self.inner.d_model()
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.inner.forward(input_ids, attention_mask)
    }

    /// Diffusers-style padded prompt embeddings.
    pub fn encode_padded_prompt_embeds(
        &self,
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
                .count()
                .min(max_sequence_length);

            let trimmed_ids = input_ids.i((b, ..seq_len))?.reshape((1, seq_len))?;
            let trimmed_mask = mask_row.i(..seq_len)?.reshape((1, seq_len))?;
            let hidden = self.forward(&trimmed_ids, Some(&trimmed_mask))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ltx_video::quantized_t5_encoder::RelativeBiasMode;

    #[test]
    fn umt5_config_uses_per_layer_bias_mode() {
        let cfg = T5EncoderConfig::umt5_xxl();
        assert_eq!(cfg.vocab_size, 256_384);
        assert_eq!(cfg.relative_bias_mode, RelativeBiasMode::Umt5EveryLayer);
    }
}
