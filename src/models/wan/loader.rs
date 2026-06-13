//! Diffusers-style weight and config discovery for Wan checkpoints.

use std::path::{Path, PathBuf};

use crate::models::ltx_video::loader::{LoaderError, load_model_config};

use super::configs::{
    WanFullConfig, WanModelIndex, WanSchedulerConfig, WanTextEncoderConfig, WanTransformerConfig,
    WanVaeConfig, infer_variant, validate_model_index,
};

/// Relative paths inside a Diffusers Wan model directory.
#[derive(Debug, Clone)]
pub struct WanComponentPaths {
    pub root: PathBuf,
    pub model_index: PathBuf,
    pub transformer_config: PathBuf,
    pub vae_config: PathBuf,
    pub scheduler_config: PathBuf,
    pub text_encoder_config: PathBuf,
    pub tokenizer_dir: PathBuf,
}

impl WanComponentPaths {
    pub fn from_root(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        Self {
            model_index: root.join("model_index.json"),
            transformer_config: root.join("transformer/config.json"),
            vae_config: root.join("vae/config.json"),
            scheduler_config: root.join("scheduler/scheduler_config.json"),
            text_encoder_config: root.join("text_encoder/config.json"),
            tokenizer_dir: root.join("tokenizer"),
            root,
        }
    }

    pub fn validate_exists(&self) -> Result<(), LoaderError> {
        let required = [
            ("model_index.json", &self.model_index),
            ("transformer/config.json", &self.transformer_config),
            ("vae/config.json", &self.vae_config),
            ("scheduler/scheduler_config.json", &self.scheduler_config),
            ("text_encoder/config.json", &self.text_encoder_config),
        ];
        for (label, path) in required {
            if !path.exists() {
                return Err(LoaderError::FileRead {
                    path: format!("missing Wan component: {label} at {}", path.display()),
                    source: std::io::Error::new(std::io::ErrorKind::NotFound, label),
                });
            }
        }
        Ok(())
    }
}

fn shards_from_index(component_dir: &Path, index_path: &Path) -> Result<Vec<PathBuf>, LoaderError> {
    let index: serde_json::Value = load_model_config(index_path)?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| LoaderError::JsonParse {
            path: index_path.display().to_string(),
            source: serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "missing weight_map",
            )),
        })?;
    let mut files: Vec<PathBuf> = weight_map
        .values()
        .filter_map(|v| v.as_str())
        .map(|f| component_dir.join(f))
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

/// Discover safetensors shard files for a component subfolder.
pub fn discover_safetensors(component_dir: &Path) -> Result<Vec<PathBuf>, LoaderError> {
    for index_name in [
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ] {
        let index_path = component_dir.join(index_name);
        if index_path.exists() {
            return shards_from_index(component_dir, &index_path);
        }
    }

    for single_name in [
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
    ] {
        let single = component_dir.join(single_name);
        if single.exists() {
            return Ok(vec![single]);
        }
    }

    Err(LoaderError::NoSafetensorsFound {
        path: component_dir.display().to_string(),
    })
}

/// Load and validate all Wan configs from a Diffusers model directory.
pub fn load_wan_config(root: impl AsRef<Path>) -> Result<WanFullConfig, LoaderError> {
    let paths = WanComponentPaths::from_root(root);
    paths.validate_exists()?;

    let model_index: WanModelIndex = load_model_config(&paths.model_index)?;
    validate_model_index(&model_index).map_err(|e| LoaderError::JsonParse {
        path: paths.model_index.display().to_string(),
        source: serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e,
        )),
    })?;

    let transformer: WanTransformerConfig = load_model_config(&paths.transformer_config)?;
    let vae: WanVaeConfig = load_model_config(&paths.vae_config)?;
    let scheduler: WanSchedulerConfig = load_model_config(&paths.scheduler_config)?;
    let text_encoder: WanTextEncoderConfig = load_model_config(&paths.text_encoder_config)?;
    let variant = infer_variant(&transformer);

    Ok(WanFullConfig {
        variant,
        model_index,
        transformer,
        vae,
        scheduler,
        text_encoder,
    })
}

/// Summary of weight files found for each Wan component.
#[derive(Debug, Clone, Default)]
pub struct WanWeightInventory {
    pub transformer_shards: Vec<PathBuf>,
    pub vae_shards: Vec<PathBuf>,
    pub text_encoder_shards: Vec<PathBuf>,
}

impl WanWeightInventory {
    pub fn discover(root: impl AsRef<Path>) -> Result<Self, LoaderError> {
        let root = root.as_ref();
        Ok(Self {
            transformer_shards: discover_safetensors(&root.join("transformer")).unwrap_or_default(),
            vae_shards: discover_safetensors(&root.join("vae")).unwrap_or_default(),
            text_encoder_shards: discover_safetensors(&root.join("text_encoder"))
                .unwrap_or_default(),
        })
    }

    pub fn has_all_components(&self) -> bool {
        !self.transformer_shards.is_empty()
            && !self.vae_shards.is_empty()
            && !self.text_encoder_shards.is_empty()
    }
}
