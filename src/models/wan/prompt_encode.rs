//! Wan prompt encoding (tokenizer + UMT5), matching Diffusers `encode_prompt`.

use candle_core::{Device, Result, Tensor};

use super::text_encoder::Umt5TextEncoder;
use super::tokenizer::WanTokenizer;

/// Encode positive and optional negative prompts for Wan CFG.
pub fn encode_prompt(
    tokenizer: &WanTokenizer,
    text_encoder: &mut Umt5TextEncoder,
    prompt: &[String],
    negative_prompt: Option<&[String]>,
    do_classifier_free_guidance: bool,
    max_sequence_length: usize,
    device: &Device,
) -> Result<(Tensor, Option<Tensor>)> {
    let (p_ids, p_mask) = tokenizer.encode_prompts(prompt, max_sequence_length)?;
    let p_ids = p_ids.to_device(device)?;
    let p_mask = p_mask.to_device(device)?;

    let prompt_embeds =
        text_encoder.encode_padded_prompt_embeds(&p_ids, &p_mask, max_sequence_length)?;

    if do_classifier_free_guidance {
        let neg_text: Vec<String> = match negative_prompt {
            Some(n) if !n.is_empty() => n.to_vec(),
            _ => vec![String::new(); prompt.len()],
        };
        let (n_ids, n_mask) = tokenizer.encode_prompts(&neg_text, max_sequence_length)?;
        let n_ids = n_ids.to_device(device)?;
        let n_mask = n_mask.to_device(device)?;
        let negative_embeds =
            text_encoder.encode_padded_prompt_embeds(&n_ids, &n_mask, max_sequence_length)?;
        return Ok((prompt_embeds, Some(negative_embeds)));
    }

    Ok((prompt_embeds, None))
}
