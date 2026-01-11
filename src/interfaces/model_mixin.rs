use std::path::Path;

use candle_core::{Error, Result};

pub trait ModelMixin {
    fn from_pretrained<P: AsRef<Path>>(_path: P) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Msg("from_pretrained is not implemented".to_string()))
    }

    fn save_pretrained<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        Err(Error::Msg("save_pretrained is not implemented".to_string()))
    }

    fn enable_gradient_checkpointing(&mut self, _enabled: bool) {}

    fn set_use_memory_efficient_attention_xformers(&mut self, _enabled: bool) {}
}

pub fn apply_model_mixin<P: ModelMixin>(model: &mut P) {
    if let Ok(path) = std::env::var("CANDLE_VIDEO_MODEL_LOAD_PATH") {
        let _ = P::from_pretrained(path);
    }
    if let Ok(path) = std::env::var("CANDLE_VIDEO_MODEL_SAVE_PATH") {
        let _ = model.save_pretrained(path);
    }
    if std::env::var("CANDLE_VIDEO_MODEL_XFORMERS").is_ok() {
        model.set_use_memory_efficient_attention_xformers(true);
    }
}
