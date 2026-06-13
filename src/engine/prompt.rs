//! Prompt encoding abstractions shared across model backends.

use candle_core::{Result, Tensor};

/// Encoded prompt tensors produced by a text encoder.
#[derive(Debug, Clone)]
pub struct PromptEmbeds {
    pub hidden_states: Tensor,
    pub attention_mask: Tensor,
}

/// Backend-agnostic prompt encoder contract.
pub trait PromptEncoder {
    fn encode(&mut self, prompt: &str, max_len: usize) -> Result<PromptEmbeds>;
    fn encode_batch(&mut self, prompts: &[String], max_len: usize) -> Result<PromptEmbeds>;
}
