//! Wan UMT5 tokenizer (T5TokenizerFast / SentencePiece).

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use tokenizers::Tokenizer as HfTokenizer;

use crate::models::ltx_video::loader::LoaderError;
use crate::models::ltx_video::t2v_pipeline::Tokenizer as PipelineTokenizer;

use super::prompt_clean::prompt_clean;

/// Default max sequence length for Wan 2.1 T2V (Diffusers default in `_get_t5_prompt_embeds`).
pub const WAN_DEFAULT_MAX_SEQ_LEN: usize = 226;

/// Tokenizer for Wan UMT5 checkpoints.
pub struct WanTokenizer {
    inner: HfTokenizer,
    device: Device,
    model_max_length: usize,
}

impl WanTokenizer {
    pub fn from_pretrained(
        tokenizer_dir: impl AsRef<Path>,
    ) -> std::result::Result<Self, LoaderError> {
        let dir = tokenizer_dir.as_ref();
        let tokenizer_json = dir.join("tokenizer.json");
        let path = if tokenizer_json.exists() {
            tokenizer_json
        } else {
            dir.join("spiece.model")
        };
        let inner = HfTokenizer::from_file(&path)
            .map_err(|e| LoaderError::FileRead {
                path: path.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            })?;
        Ok(Self {
            inner,
            device: Device::Cpu,
            model_max_length: WAN_DEFAULT_MAX_SEQ_LEN,
        })
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn with_model_max_length(mut self, max_len: usize) -> Self {
        self.model_max_length = max_len;
        self
    }

    pub fn encode_cleaned(
        &self,
        prompts: &[String],
        max_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut all_ids = Vec::with_capacity(prompts.len() * max_length);
        let mut all_mask = Vec::with_capacity(prompts.len() * max_length);

        for prompt in prompts {
            let encoding = self
                .inner
                .encode(prompt.as_str(), true)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let mut ids: Vec<u32> = encoding.get_ids().to_vec();
            let mut mask: Vec<u32> = vec![1; ids.len()];

            if ids.len() > max_length {
                ids.truncate(max_length);
                mask.truncate(max_length);
                if let Some(last) = ids.last_mut() {
                    *last = 1; // eos
                }
            } else {
                let pad = max_length - ids.len();
                ids.extend(std::iter::repeat_n(0u32, pad));
                mask.extend(std::iter::repeat_n(0u32, pad));
            }

            all_ids.extend(ids);
            all_mask.extend(mask);
        }

        let batch = prompts.len();
        let ids_t =
            Tensor::new(all_ids, &self.device)?.reshape((batch, max_length))?;
        let mask_t = Tensor::new(all_mask, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((batch, max_length))?;
        Ok((ids_t, mask_t))
    }

    pub fn encode_prompts(
        &self,
        prompts: &[String],
        max_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cleaned: Vec<String> = prompts.iter().map(|p| prompt_clean(p)).collect();
        self.encode_cleaned(&cleaned, max_length)
    }
}

impl PipelineTokenizer for WanTokenizer {
    fn encode_batch(&self, prompts: &[String], max_length: usize) -> Result<(Tensor, Tensor)> {
        self.encode_prompts(prompts, max_length)
    }

    fn model_max_length(&self) -> usize {
        self.model_max_length
    }
}
