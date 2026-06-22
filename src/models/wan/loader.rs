//! Diffusers-style weight and config discovery for Wan checkpoints.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::models::ltx_video::loader::{LoaderError, load_model_config};

use super::configs::{
    WanFullConfig, WanModelIndex, WanSchedulerConfig, WanTextEncoderConfig, WanTransformerConfig,
    WanVaeConfig, infer_variant, validate_model_index,
};

/// Layout of a Wan checkpoint on disk.
#[derive(Debug, Clone)]
pub enum WanLayout {
    /// HuggingFace Diffusers folder (`model_index.json` + component subdirs).
    Diffusers(WanComponentPaths),
    /// Oxide-style bundle: single transformer safetensors + `vae/` + `text_encoder_gguf/`.
    Consolidated(WanConsolidatedPaths),
}

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

/// Paths for consolidated Wan 2.1 T2V 1.3B bundles (GGUF text encoder).
#[derive(Debug, Clone)]
pub struct WanConsolidatedPaths {
    pub root: PathBuf,
    pub transformer_weights: PathBuf,
    pub vae_weights: PathBuf,
    pub text_encoder_gguf: PathBuf,
    pub tokenizer_json: PathBuf,
}

impl WanConsolidatedPaths {
    pub fn from_root(root: impl AsRef<Path>) -> Result<Self, LoaderError> {
        let root = root.as_ref().to_path_buf();
        let transformer_weights =
            find_transformer_safetensors(&root).ok_or_else(|| LoaderError::FileRead {
                path: root.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no transformer .safetensors in consolidated Wan bundle",
                ),
            })?;
        let vae_weights = root.join("vae/wan_2.1_vae.safetensors");
        if !vae_weights.exists() {
            return Err(LoaderError::FileRead {
                path: vae_weights.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "vae weights"),
            });
        }
        let gguf_dir = root.join("text_encoder_gguf");
        let text_encoder_gguf =
            find_gguf_in_dir(&gguf_dir).ok_or_else(|| LoaderError::FileRead {
                path: gguf_dir.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "UMT5 GGUF"),
            })?;
        let tokenizer_json = gguf_dir.join("tokenizer.json");
        if !tokenizer_json.exists() {
            return Err(LoaderError::FileRead {
                path: tokenizer_json.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "tokenizer.json"),
            });
        }
        Ok(Self {
            root,
            transformer_weights,
            vae_weights,
            text_encoder_gguf,
            tokenizer_json,
        })
    }
}

/// Detect whether `root` is Diffusers or consolidated layout.
pub fn detect_wan_layout(root: impl AsRef<Path>) -> Result<WanLayout, LoaderError> {
    let root = root.as_ref();
    let diffusers = WanComponentPaths::from_root(root);
    if diffusers.model_index.exists() {
        diffusers.validate_exists()?;
        return Ok(WanLayout::Diffusers(diffusers));
    }
    Ok(WanLayout::Consolidated(WanConsolidatedPaths::from_root(
        root,
    )?))
}

fn find_transformer_safetensors(root: &Path) -> Option<PathBuf> {
    let preferred = [
        "wan2.1_t2v_1.3B_fp16.safetensors",
        "wan2.1_t2v_1.3b_fp16.safetensors",
    ];
    for name in preferred {
        let p = root.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension().map(|e| e == "safetensors").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("t2v") || n.contains("wan2.1"))
                    .unwrap_or(false)
        })
}

fn find_gguf_in_dir(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    // Same preference order as LTX T5 GGUF discovery (best quant first).
    for name in [
        "umt5-xxl-encoder-Q8_0.gguf",
        "umt5-xxl-encoder-Q6_K.gguf",
        "umt5-xxl-encoder-Q5_K_M.gguf",
        "umt5-xxl-encoder-Q4_K_M.gguf",
        "umt5-xxl-encoder-Q4_0.gguf",
        "umt5-xxl-encoder-Q3_K_M.gguf",
        "umt5-xxl-encoder-Q2_K.gguf",
    ] {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map(|e| e == "gguf").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("umt5") || n.contains("encoder"))
                    .unwrap_or(false)
        })
        .collect();
    if files.is_empty() {
        files = fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
            .collect();
    }
    files.sort();
    files.into_iter().next()
}

fn remapped_transformer_cache(native: &Path) -> PathBuf {
    native.with_extension("diffusers.safetensors")
}

fn remapped_vae_cache(native: &Path) -> PathBuf {
    native.with_file_name(format!(
        "{}.diffusers.safetensors",
        native.file_stem().and_then(|s| s.to_str()).unwrap_or("vae")
    ))
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

    for single_name in ["diffusion_pytorch_model.safetensors", "model.safetensors"] {
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
    match detect_wan_layout(root.as_ref())? {
        WanLayout::Consolidated(_) => Ok(WanFullConfig::wan21_t2v_13b()),
        WanLayout::Diffusers(paths) => load_wan_config_diffusers(&paths),
    }
}

fn load_wan_config_diffusers(paths: &WanComponentPaths) -> Result<WanFullConfig, LoaderError> {
    let model_index: WanModelIndex = load_model_config(&paths.model_index)?;
    validate_model_index(&model_index).map_err(|e| LoaderError::JsonParse {
        path: paths.model_index.display().to_string(),
        source: serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
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

/// Build a VarBuilder for Wan transformer weights (Diffusers or native single-file).
pub fn load_transformer_var_builder<'a>(
    layout: &'a WanLayout,
    device: &'a Device,
    dtype: DType,
) -> Result<VarBuilder<'a>, LoaderError> {
    match layout {
        WanLayout::Diffusers(paths) => {
            let shards = discover_safetensors(&paths.root.join("transformer"))?;
            let paths: Vec<&Path> = shards.iter().map(|p| p.as_path()).collect();
            unsafe {
                VarBuilder::from_mmaped_safetensors(&paths, dtype, device)
                    .map_err(LoaderError::Candle)
            }
        }
        WanLayout::Consolidated(paths) => {
            let mmap_path = remapped_transformer_cache(&paths.transformer_weights);
            if !mmap_path.exists() {
                return Err(LoaderError::FileRead {
                    path: mmap_path.display().to_string(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "missing .diffusers.safetensors cache — run scripts/wan/remap_consolidated.py",
                    ),
                });
            }
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[mmap_path.as_path()], dtype, device)
                    .map_err(LoaderError::Candle)
            }
        }
    }
}

/// Build a VarBuilder for Wan VAE weights (Diffusers or native single-file).
pub fn load_vae_var_builder<'a>(
    layout: &'a WanLayout,
    device: &'a Device,
    dtype: DType,
) -> Result<VarBuilder<'a>, LoaderError> {
    match layout {
        WanLayout::Diffusers(paths) => {
            let shards = discover_safetensors(&paths.root.join("vae"))?;
            let paths: Vec<&Path> = shards.iter().map(|p| p.as_path()).collect();
            unsafe {
                VarBuilder::from_mmaped_safetensors(&paths, dtype, device)
                    .map_err(LoaderError::Candle)
            }
        }
        WanLayout::Consolidated(paths) => {
            let mmap_path = remapped_vae_cache(&paths.vae_weights);
            if !mmap_path.exists() {
                return Err(LoaderError::FileRead {
                    path: mmap_path.display().to_string(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "missing .diffusers.safetensors cache — run scripts/wan/remap_consolidated.py",
                    ),
                });
            }
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[mmap_path.as_path()], dtype, device)
                    .map_err(LoaderError::Candle)
            }
        }
    }
}

/// Tokenizer path for either layout (directory containing `tokenizer.json` or `spiece.model`).
pub fn wan_tokenizer_path(layout: &WanLayout) -> PathBuf {
    match layout {
        WanLayout::Diffusers(p) => p.tokenizer_dir.clone(),
        WanLayout::Consolidated(p) => p.tokenizer_json.parent().unwrap_or(&p.root).to_path_buf(),
    }
}

/// Scheduler config path (Diffusers only; consolidated uses baked-in config).
pub fn wan_scheduler_config_path(layout: &WanLayout) -> Option<PathBuf> {
    match layout {
        WanLayout::Diffusers(p) => Some(p.scheduler_config.clone()),
        WanLayout::Consolidated(_) => None,
    }
}

/// Summary of weight files found for each Wan component.
#[derive(Debug, Clone, Default)]
pub struct WanWeightInventory {
    pub transformer_shards: Vec<PathBuf>,
    pub vae_shards: Vec<PathBuf>,
    pub text_encoder_shards: Vec<PathBuf>,
    pub text_encoder_gguf: Option<PathBuf>,
}

impl WanWeightInventory {
    pub fn discover(root: impl AsRef<Path>) -> Result<Self, LoaderError> {
        let root = root.as_ref();
        match detect_wan_layout(root)? {
            WanLayout::Diffusers(paths) => Ok(Self {
                transformer_shards: discover_safetensors(&paths.root.join("transformer"))
                    .unwrap_or_default(),
                vae_shards: discover_safetensors(&paths.root.join("vae")).unwrap_or_default(),
                text_encoder_shards: discover_safetensors(&paths.root.join("text_encoder"))
                    .unwrap_or_default(),
                text_encoder_gguf: None,
            }),
            WanLayout::Consolidated(paths) => Ok(Self {
                transformer_shards: vec![paths.transformer_weights],
                vae_shards: vec![paths.vae_weights],
                text_encoder_shards: vec![],
                text_encoder_gguf: Some(paths.text_encoder_gguf),
            }),
        }
    }

    pub fn has_all_components(&self) -> bool {
        !self.transformer_shards.is_empty()
            && !self.vae_shards.is_empty()
            && (!self.text_encoder_shards.is_empty() || self.text_encoder_gguf.is_some())
    }
}

/// ponytail: smoke check that remapped native keys are non-empty after load.
pub fn _assert_remapped_nonempty(tensors: &HashMap<String, Tensor>) {
    debug_assert!(!tensors.is_empty());
}
