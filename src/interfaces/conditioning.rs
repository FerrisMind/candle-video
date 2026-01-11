use candle_core::{Device, Result, Tensor};

#[derive(Debug)]
pub struct Conditioning {
    pub prompt_embeds: Tensor,
    pub prompt_attention_mask: Tensor,
    pub negative_prompt_embeds: Option<Tensor>,
    pub negative_prompt_attention_mask: Option<Tensor>,
}

pub trait TextConditioner {
    fn encode_prompt(
        &mut self,
        prompt: &str,
        negative: Option<&str>,
        device: &Device,
    ) -> Result<Conditioning>;
}
