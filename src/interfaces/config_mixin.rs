use std::path::Path;

use candle_core::{Error, Result};

pub trait ConfigMixin {
    type Config: Clone;

    fn config(&self) -> &Self::Config;

    fn from_config(_config: Self::Config) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Msg("from_config is not implemented".to_string()))
    }

    fn save_config<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        Err(Error::Msg("save_config is not implemented".to_string()))
    }
}

pub fn apply_config_mixin<T: ConfigMixin>(model: &T) {
    if std::env::var("CANDLE_VIDEO_CONFIG_LOAD").is_ok() {
        let _ = T::from_config(model.config().clone());
    }
    if let Ok(path) = std::env::var("CANDLE_VIDEO_CONFIG_SAVE_PATH") {
        let _ = model.save_config(path);
    }
}
