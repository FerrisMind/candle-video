//! Diffusers-style weight and config discovery for Wan checkpoints.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::models::ltx_video::loader::{LoaderError, load_model_config};

use super::configs::{
    WanFullConfig, WanModelIndex, WanSchedulerConfig, WanTextEncoderConfig, WanTransformerConfig,
    WanVaeConfig, infer_variant, validate_model_index, validate_wan21_t2v_13b,
};
use super::weight_format::{remap_transformer_key, remap_vae_key};

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
    let mut files = Vec::with_capacity(weight_map.len());
    for (tensor_name, value) in weight_map.iter() {
        let file_name = value.as_str().ok_or_else(|| LoaderError::JsonParse {
            path: index_path.display().to_string(),
            source: serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("weight_map entry `{tensor_name}` is not a string"),
            )),
        })?;
        files.push(component_dir.join(file_name));
    }
    files.sort();
    files.dedup();
    let missing: Vec<String> = files
        .iter()
        .filter(|path| !path.exists())
        .map(|path| path.display().to_string())
        .collect();
    if !missing.is_empty() {
        return Err(LoaderError::MissingShards { missing });
    }
    validate_index_headers(index_path, &files, weight_map)?;
    Ok(files)
}

fn validate_index_headers(
    index_path: &Path,
    files: &[PathBuf],
    weight_map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), LoaderError> {
    let mut expected = HashMap::with_capacity(weight_map.len());
    for (tensor, file) in weight_map {
        let file = file.as_str().ok_or_else(|| LoaderError::JsonParse {
            path: index_path.display().to_string(),
            source: serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("weight_map entry `{tensor}` must contain a string shard name"),
            )),
        })?;
        expected.insert(tensor.clone(), file.to_string());
    }
    let mut seen: HashMap<String, String> = HashMap::new();
    for path in files {
        let actual_file = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| LoaderError::FileRead {
                path: path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "safetensors shard has a non-UTF8 filename",
                ),
            })?
            .to_string();
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(LoaderError::Candle)?;
        for (tensor, _) in tensors.tensors() {
            if let Some(previous_file) = seen.insert(tensor.clone(), actual_file.clone()) {
                return Err(LoaderError::DuplicateMappedTensor {
                    mapped: tensor,
                    first: previous_file,
                    second: actual_file.clone(),
                });
            }
            match expected.get(&tensor) {
                None => {
                    return Err(LoaderError::UnexpectedTensors {
                        names: vec![tensor],
                    });
                }
                Some(expected_file) if expected_file != &actual_file => {
                    return Err(LoaderError::ShardTensorMismatch {
                        tensor,
                        expected: expected_file.clone(),
                        actual: actual_file.clone(),
                    });
                }
                Some(_) => {}
            }
        }
    }
    let missing: Vec<String> = expected
        .keys()
        .filter(|tensor| !seen.contains_key(*tensor))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(LoaderError::MissingTensors { missing });
    }
    let _ = index_path;
    Ok(())
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

    let mut safetensors: Vec<PathBuf> = fs::read_dir(component_dir)
        .map_err(|source| LoaderError::FileRead {
            path: component_dir.display().to_string(),
            source,
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect();
    safetensors.sort();
    if !safetensors.is_empty() {
        return Ok(safetensors);
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
    let variant = infer_variant(&transformer).map_err(|error| LoaderError::JsonParse {
        path: paths.transformer_config.display().to_string(),
        source: serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
    })?;
    let config = WanFullConfig {
        variant,
        model_index,
        transformer,
        vae,
        scheduler,
        text_encoder,
    };
    validate_wan21_t2v_13b(&config).map_err(|error| LoaderError::JsonParse {
        path: paths.root.display().to_string(),
        source: serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
    })?;
    Ok(config)
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
            mapped_safetensors_var_builder(&paths, device, dtype, str::to_string)
        }
        WanLayout::Consolidated(paths) => mapped_safetensors_var_builder(
            &[paths.transformer_weights.as_path()],
            device,
            dtype,
            remap_transformer_key,
        ),
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
            mapped_safetensors_var_builder(&paths, device, dtype, str::to_string)
        }
        WanLayout::Consolidated(paths) => {
            mapped_safetensors_var_builder(&[paths.vae_weights.as_path()], device, dtype, |name| {
                remap_vae_key(name).unwrap_or_else(|| name.to_string())
            })
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
                transformer_shards: discover_safetensors(&paths.root.join("transformer"))?,
                vae_shards: discover_safetensors(&paths.root.join("vae"))?,
                text_encoder_shards: discover_safetensors(&paths.root.join("text_encoder"))?,
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

/// Write a machine-readable inventory for a Wan model directory.
///
/// The manifest intentionally records only SafeTensors headers and file
/// metadata, so generating it does not copy or materialise multi-gigabyte
/// weights. SHA-256 is left null unless a caller adds it externally.
pub fn write_wan_manifest(
    root: impl AsRef<Path>,
    output: Option<&Path>,
) -> Result<PathBuf, LoaderError> {
    let root = root.as_ref();
    let layout = detect_wan_layout(root)?;
    let mut files = Vec::new();
    let mut add_file = |component: &str, path: &Path| -> Result<(), LoaderError> {
        if !path.exists() {
            return Ok(());
        }
        let metadata = fs::metadata(path).map_err(|source| LoaderError::FileRead {
            path: path.display().to_string(),
            source,
        })?;
        let (tensor_count, dtypes) = if path.extension().is_some_and(|ext| ext == "safetensors") {
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
                .map_err(LoaderError::Candle)?;
            let mut dtypes = HashMap::<String, usize>::new();
            for (_, view) in tensors.tensors() {
                *dtypes.entry(format!("{:?}", view.dtype())).or_default() += 1;
            }
            (
                Some(dtypes.values().sum::<usize>()),
                serde_json::json!(dtypes),
            )
        } else {
            (None, serde_json::json!({}))
        };
        files.push(serde_json::json!({
            "component": component,
            "path": path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/"),
            "bytes": metadata.len(),
            "tensor_count": tensor_count,
            "dtypes": dtypes,
            "sha256": serde_json::Value::Null,
        }));
        Ok(())
    };

    match &layout {
        WanLayout::Diffusers(paths) => {
            add_file("model_index", &paths.model_index)?;
            add_file("transformer_config", &paths.transformer_config)?;
            add_file("vae_config", &paths.vae_config)?;
            add_file("scheduler_config", &paths.scheduler_config)?;
            add_file("text_encoder_config", &paths.text_encoder_config)?;
            for path in discover_safetensors(&paths.root.join("transformer"))? {
                add_file("transformer", &path)?;
            }
            for path in discover_safetensors(&paths.root.join("vae"))? {
                add_file("vae", &path)?;
            }
            for path in discover_safetensors(&paths.root.join("text_encoder"))? {
                add_file("text_encoder", &path)?;
            }
            let tokenizer = paths.tokenizer_dir.join("tokenizer.json");
            add_file("tokenizer", &tokenizer)?;
        }
        WanLayout::Consolidated(paths) => {
            add_file("transformer", &paths.transformer_weights)?;
            add_file("vae", &paths.vae_weights)?;
            add_file("text_encoder_gguf", &paths.text_encoder_gguf)?;
            add_file("tokenizer", &paths.tokenizer_json)?;
        }
    }

    let config = load_wan_config(root)?;
    let manifest = serde_json::json!({
        "format": "candle-video-wan-manifest-v1",
        "model": config.variant.display_name(),
        "diffusers_version": config.model_index.diffusers_version,
        "layout": match layout {
            WanLayout::Diffusers(_) => "diffusers",
            WanLayout::Consolidated(_) => "consolidated",
        },
        "components": {
            "transformer": {
                "tensor_count": files.iter().filter(|f| f["component"] == "transformer").map(|f| f["tensor_count"].as_u64().unwrap_or(0)).sum::<u64>(),
                "dtype": "checkpoint-defined"
            },
            "vae": {
                "tensor_count": files.iter().filter(|f| f["component"] == "vae").map(|f| f["tensor_count"].as_u64().unwrap_or(0)).sum::<u64>(),
                "dtype": "checkpoint-defined"
            },
            "text_encoder": {
                "tensor_count": files.iter().filter(|f| f["component"] == "text_encoder").map(|f| f["tensor_count"].as_u64().unwrap_or(0)).sum::<u64>(),
                "dtype": "checkpoint-defined"
            }
        },
        "files": files,
    });
    let output = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join("candle-video-manifest.json"));
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| LoaderError::FileRead {
            path: parent.display().to_string(),
            source,
        })?;
    }
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|source| LoaderError::JsonParse {
        path: output.display().to_string(),
        source,
    })?;
    fs::write(&output, bytes).map_err(|source| LoaderError::FileRead {
        path: output.display().to_string(),
        source,
    })?;
    Ok(output)
}

/// Validate the complete Wan transformer tensor inventory before constructing modules.
pub fn validate_transformer_weights(
    layout: &WanLayout,
    config: &WanTransformerConfig,
) -> Result<(), LoaderError> {
    let paths = match layout {
        WanLayout::Diffusers(paths) => discover_safetensors(&paths.root.join("transformer"))?,
        WanLayout::Consolidated(paths) => vec![paths.transformer_weights.clone()],
    };
    let headers = match layout {
        WanLayout::Diffusers(_) => collect_mapped_names(&paths, str::to_string)?,
        WanLayout::Consolidated(_) => collect_mapped_names(&paths, remap_transformer_key)?,
    };
    let mapped: HashSet<String> = headers.keys().cloned().collect();
    let expected = expected_transformer_tensor_names(config);
    let missing: Vec<String> = expected.difference(&mapped).cloned().collect();
    if !missing.is_empty() {
        return Err(LoaderError::MissingTensors { missing });
    }
    let mut unexpected: Vec<String> = mapped.difference(&expected).cloned().collect();
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(LoaderError::UnexpectedTensors { names: unexpected });
    }
    let expected_shapes = expected_transformer_tensor_shapes(config);
    for (name, expected_shape) in expected_shapes {
        let meta = headers
            .get(&name)
            .ok_or_else(|| LoaderError::MissingTensors {
                missing: vec![name.clone()],
            })?;
        if meta.shape != expected_shape {
            return Err(LoaderError::TensorShapeMismatch {
                tensor: name,
                expected: expected_shape,
                actual: meta.shape.clone(),
            });
        }
        if !matches!(meta.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(LoaderError::UnsupportedTensorDtype {
                tensor: name,
                dtype: meta.dtype.clone(),
            });
        }
    }
    Ok(())
}

/// Validate the complete Wan VAE tensor inventory before constructing the
/// decoder. The real Wan2.1 checkpoint contains 194 tensors; this preflight
/// derives the expected topology from the config and checks names, shapes, and
/// storage dtypes without materialising any tensor data.
pub fn validate_vae_weights(layout: &WanLayout, config: &WanVaeConfig) -> Result<(), LoaderError> {
    let paths = match layout {
        WanLayout::Diffusers(paths) => discover_safetensors(&paths.root.join("vae"))?,
        WanLayout::Consolidated(paths) => vec![paths.vae_weights.clone()],
    };
    let headers = match layout {
        WanLayout::Diffusers(_) => collect_mapped_names(&paths, str::to_string)?,
        WanLayout::Consolidated(_) => collect_mapped_names(&paths, |name| {
            remap_vae_key(name).unwrap_or_else(|| name.to_string())
        })?,
    };
    let expected = expected_vae_tensor_shapes(config)?;
    let mapped: HashSet<String> = headers.keys().cloned().collect();
    let expected_names: HashSet<String> = expected.keys().cloned().collect();
    let missing: Vec<String> = expected_names.difference(&mapped).cloned().collect();
    if !missing.is_empty() {
        return Err(LoaderError::MissingTensors { missing });
    }
    let mut unexpected: Vec<String> = mapped.difference(&expected_names).cloned().collect();
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(LoaderError::UnexpectedTensors { names: unexpected });
    }
    for (name, expected_shape) in expected {
        let meta = headers
            .get(&name)
            .ok_or_else(|| LoaderError::MissingTensors {
                missing: vec![name.clone()],
            })?;
        if meta.shape != expected_shape {
            return Err(LoaderError::TensorShapeMismatch {
                tensor: name,
                expected: expected_shape,
                actual: meta.shape.clone(),
            });
        }
        if !matches!(meta.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(LoaderError::UnsupportedTensorDtype {
                tensor: name,
                dtype: meta.dtype.clone(),
            });
        }
    }
    Ok(())
}

fn insert_conv3d(
    shapes: &mut HashMap<String, Vec<usize>>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: [usize; 3],
) {
    shapes.insert(
        format!("{prefix}.weight"),
        vec![output, input, kernel[0], kernel[1], kernel[2]],
    );
    shapes.insert(format!("{prefix}.bias"), vec![output]);
}

fn insert_norm(shapes: &mut HashMap<String, Vec<usize>>, prefix: &str, channels: usize) {
    shapes.insert(format!("{prefix}.gamma"), vec![channels, 1, 1, 1]);
}

fn insert_resnet(
    shapes: &mut HashMap<String, Vec<usize>>,
    prefix: &str,
    input: usize,
    output: usize,
) {
    insert_norm(shapes, &format!("{prefix}.norm1"), input);
    insert_conv3d(shapes, &format!("{prefix}.conv1"), input, output, [3, 3, 3]);
    insert_norm(shapes, &format!("{prefix}.norm2"), output);
    insert_conv3d(
        shapes,
        &format!("{prefix}.conv2"),
        output,
        output,
        [3, 3, 3],
    );
    if input != output {
        insert_conv3d(
            shapes,
            &format!("{prefix}.conv_shortcut"),
            input,
            output,
            [1, 1, 1],
        );
    }
}

fn insert_mid(shapes: &mut HashMap<String, Vec<usize>>, prefix: &str, channels: usize) {
    insert_resnet(shapes, &format!("{prefix}.resnets.0"), channels, channels);
    insert_resnet(shapes, &format!("{prefix}.resnets.1"), channels, channels);
    shapes.insert(
        format!("{prefix}.attentions.0.norm.gamma"),
        vec![channels, 1, 1],
    );
    shapes.insert(
        format!("{prefix}.attentions.0.to_qkv.weight"),
        vec![channels * 3, channels, 1, 1],
    );
    shapes.insert(
        format!("{prefix}.attentions.0.to_qkv.bias"),
        vec![channels * 3],
    );
    shapes.insert(
        format!("{prefix}.attentions.0.proj.weight"),
        vec![channels, channels, 1, 1],
    );
    shapes.insert(format!("{prefix}.attentions.0.proj.bias"), vec![channels]);
}

fn expected_vae_tensor_shapes(
    config: &WanVaeConfig,
) -> Result<HashMap<String, Vec<usize>>, LoaderError> {
    if config.base_dim == 0 || config.dim_mult.len() != 4 || config.z_dim == 0 {
        return Err(LoaderError::Candle(candle_core::Error::Msg(
            "Wan VAE inventory requires four non-empty dim_mult levels".into(),
        )));
    }
    if config.temporal_downsample.len() != config.dim_mult.len() - 1 {
        return Err(LoaderError::Candle(candle_core::Error::Msg(format!(
            "Wan VAE temporal_downsample length {} does not match dim_mult {}",
            config.temporal_downsample.len(),
            config.dim_mult.len() - 1
        ))));
    }

    let mut shapes = HashMap::new();
    let channels: Vec<usize> = config
        .dim_mult
        .iter()
        .map(|mult| config.base_dim * *mult)
        .collect();
    insert_conv3d(
        &mut shapes,
        "post_quant_conv",
        config.z_dim,
        config.z_dim,
        [1, 1, 1],
    );
    insert_conv3d(
        &mut shapes,
        "quant_conv",
        config.z_dim * 2,
        config.z_dim * 2,
        [1, 1, 1],
    );

    insert_conv3d(&mut shapes, "encoder.conv_in", 3, channels[0], [3, 3, 3]);
    let encoder_resnets = [
        (0usize, channels[0], channels[0]),
        (1, channels[0], channels[0]),
        (3, channels[0], channels[1]),
        (4, channels[1], channels[1]),
        (6, channels[1], channels[2]),
        (7, channels[2], channels[2]),
        (9, channels[2], channels[3]),
        (10, channels[3], channels[3]),
    ];
    for (index, input, output) in encoder_resnets {
        insert_resnet(
            &mut shapes,
            &format!("encoder.down_blocks.{index}"),
            input,
            output,
        );
    }
    for (index, channel, temporal) in [
        (2usize, channels[0], false),
        (5, channels[1], true),
        (8, channels[2], true),
    ] {
        shapes.insert(
            format!("encoder.down_blocks.{index}.resample.1.weight"),
            vec![channel, channel, 3, 3],
        );
        shapes.insert(
            format!("encoder.down_blocks.{index}.resample.1.bias"),
            vec![channel],
        );
        if temporal {
            shapes.insert(
                format!("encoder.down_blocks.{index}.time_conv.weight"),
                vec![channel, channel, 3, 1, 1],
            );
            shapes.insert(
                format!("encoder.down_blocks.{index}.time_conv.bias"),
                vec![channel],
            );
        }
    }
    insert_mid(&mut shapes, "encoder.mid_block", channels[3]);
    insert_norm(&mut shapes, "encoder.norm_out", channels[3]);
    insert_conv3d(
        &mut shapes,
        "encoder.conv_out",
        channels[3],
        config.z_dim * 2,
        [3, 3, 3],
    );

    insert_conv3d(
        &mut shapes,
        "decoder.conv_in",
        config.z_dim,
        channels[3],
        [3, 3, 3],
    );
    insert_mid(&mut shapes, "decoder.mid_block", channels[3]);
    let decoder_dims = [
        channels[3],
        channels[3],
        channels[2],
        channels[1],
        channels[0],
    ];
    let temporal_upsample = config
        .temporal_downsample
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    for index in 0..4 {
        let input = if index == 0 {
            decoder_dims[index]
        } else {
            decoder_dims[index] / 2
        };
        let output = decoder_dims[index + 1];
        for residual in 0..=config.num_res_blocks {
            let residual_input = if residual == 0 { input } else { output };
            insert_resnet(
                &mut shapes,
                &format!("decoder.up_blocks.{index}.resnets.{residual}"),
                residual_input,
                output,
            );
        }
        if index < 3 {
            let up = format!("decoder.up_blocks.{index}.upsamplers.0");
            shapes.insert(
                format!("{up}.resample.1.weight"),
                vec![output / 2, output, 3, 3],
            );
            shapes.insert(format!("{up}.resample.1.bias"), vec![output / 2]);
            if temporal_upsample.get(index).copied().unwrap_or(false) {
                shapes.insert(
                    format!("{up}.time_conv.weight"),
                    vec![output * 2, output, 3, 1, 1],
                );
                shapes.insert(format!("{up}.time_conv.bias"), vec![output * 2]);
            }
        }
    }
    insert_norm(&mut shapes, "decoder.norm_out", channels[0]);
    insert_conv3d(&mut shapes, "decoder.conv_out", channels[0], 3, [3, 3, 3]);

    Ok(shapes)
}

fn expected_transformer_tensor_names(config: &WanTransformerConfig) -> HashSet<String> {
    let inner_dim = config.num_attention_heads * config.attention_head_dim;
    let mut names = HashSet::new();
    for name in [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "condition_embedder.time_embedder.linear_1.weight",
        "condition_embedder.time_embedder.linear_1.bias",
        "condition_embedder.time_embedder.linear_2.weight",
        "condition_embedder.time_embedder.linear_2.bias",
        "condition_embedder.time_proj.weight",
        "condition_embedder.time_proj.bias",
        "condition_embedder.text_embedder.linear_1.weight",
        "condition_embedder.text_embedder.linear_1.bias",
        "condition_embedder.text_embedder.linear_2.weight",
        "condition_embedder.text_embedder.linear_2.bias",
        "scale_shift_table",
        "proj_out.weight",
        "proj_out.bias",
    ] {
        names.insert(name.to_string());
    }
    for block in 0..config.num_layers {
        for attention in ["attn1", "attn2"] {
            for norm in ["norm_q", "norm_k"] {
                names.insert(format!("blocks.{block}.{attention}.{norm}.weight"));
            }
            for projection in ["to_q", "to_k", "to_v"] {
                names.insert(format!("blocks.{block}.{attention}.{projection}.weight"));
                names.insert(format!("blocks.{block}.{attention}.{projection}.bias"));
            }
            names.insert(format!("blocks.{block}.{attention}.to_out.0.weight"));
            names.insert(format!("blocks.{block}.{attention}.to_out.0.bias"));
        }
        names.insert(format!("blocks.{block}.ffn.net.0.proj.weight"));
        names.insert(format!("blocks.{block}.ffn.net.0.proj.bias"));
        names.insert(format!("blocks.{block}.ffn.net.2.weight"));
        names.insert(format!("blocks.{block}.ffn.net.2.bias"));
        names.insert(format!("blocks.{block}.norm2.weight"));
        names.insert(format!("blocks.{block}.norm2.bias"));
        names.insert(format!("blocks.{block}.scale_shift_table"));
    }
    debug_assert_eq!(names.len(), 15 + config.num_layers * 27);
    let _ = inner_dim;
    names
}

fn expected_transformer_tensor_shapes(
    config: &WanTransformerConfig,
) -> HashMap<String, Vec<usize>> {
    let inner = config.num_attention_heads * config.attention_head_dim;
    let mut shapes = HashMap::new();
    let globals = [
        (
            "patch_embedding.weight",
            vec![inner, config.in_channels, 1, 2, 2],
        ),
        ("patch_embedding.bias", vec![inner]),
        (
            "condition_embedder.time_embedder.linear_1.weight",
            vec![inner, config.freq_dim],
        ),
        (
            "condition_embedder.time_embedder.linear_1.bias",
            vec![inner],
        ),
        (
            "condition_embedder.time_embedder.linear_2.weight",
            vec![inner, inner],
        ),
        (
            "condition_embedder.time_embedder.linear_2.bias",
            vec![inner],
        ),
        (
            "condition_embedder.time_proj.weight",
            vec![inner * 6, inner],
        ),
        ("condition_embedder.time_proj.bias", vec![inner * 6]),
        (
            "condition_embedder.text_embedder.linear_1.weight",
            vec![inner, config.text_dim],
        ),
        (
            "condition_embedder.text_embedder.linear_1.bias",
            vec![inner],
        ),
        (
            "condition_embedder.text_embedder.linear_2.weight",
            vec![inner, inner],
        ),
        (
            "condition_embedder.text_embedder.linear_2.bias",
            vec![inner],
        ),
        ("scale_shift_table", vec![1, 2, inner]),
        (
            "proj_out.weight",
            vec![
                config.out_channels * config.patch_size.iter().product::<usize>(),
                inner,
            ],
        ),
        (
            "proj_out.bias",
            vec![config.out_channels * config.patch_size.iter().product::<usize>()],
        ),
    ];
    for (name, shape) in globals {
        shapes.insert(name.to_string(), shape);
    }
    for block in 0..config.num_layers {
        for attention in ["attn1", "attn2"] {
            for norm in ["norm_q", "norm_k"] {
                shapes.insert(
                    format!("blocks.{block}.{attention}.{norm}.weight"),
                    vec![inner],
                );
            }
            for projection in ["to_q", "to_k", "to_v"] {
                shapes.insert(
                    format!("blocks.{block}.{attention}.{projection}.weight"),
                    vec![inner, inner],
                );
                shapes.insert(
                    format!("blocks.{block}.{attention}.{projection}.bias"),
                    vec![inner],
                );
            }
            shapes.insert(
                format!("blocks.{block}.{attention}.to_out.0.weight"),
                vec![inner, inner],
            );
            shapes.insert(
                format!("blocks.{block}.{attention}.to_out.0.bias"),
                vec![inner],
            );
        }
        shapes.insert(
            format!("blocks.{block}.ffn.net.0.proj.weight"),
            vec![config.ffn_dim, inner],
        );
        shapes.insert(
            format!("blocks.{block}.ffn.net.0.proj.bias"),
            vec![config.ffn_dim],
        );
        shapes.insert(
            format!("blocks.{block}.ffn.net.2.weight"),
            vec![inner, config.ffn_dim],
        );
        shapes.insert(format!("blocks.{block}.ffn.net.2.bias"), vec![inner]);
        shapes.insert(format!("blocks.{block}.norm2.weight"), vec![inner]);
        shapes.insert(format!("blocks.{block}.norm2.bias"), vec![inner]);
        shapes.insert(
            format!("blocks.{block}.scale_shift_table"),
            vec![1, 6, inner],
        );
    }
    shapes
}

#[derive(Debug, Clone)]
struct TensorHeader {
    shape: Vec<usize>,
    dtype: String,
}

fn collect_mapped_names<F>(
    paths: &[PathBuf],
    remap: F,
) -> Result<HashMap<String, TensorHeader>, LoaderError>
where
    F: Fn(&str) -> String,
{
    let mut mapped_to_native: HashMap<String, (String, PathBuf, TensorHeader)> = HashMap::new();
    for path in paths {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(LoaderError::Candle)?;
        for (native_name, view) in tensors.tensors() {
            let mapped_name = remap(&native_name);
            let header = TensorHeader {
                shape: view.shape().to_vec(),
                dtype: format!("{:?}", view.dtype()),
            };
            if let Some(first) = mapped_to_native.insert(
                mapped_name.clone(),
                (native_name.clone(), path.clone(), header.clone()),
            ) {
                return Err(LoaderError::DuplicateMappedTensor {
                    mapped: mapped_name,
                    first: format!("{}:{}", first.1.display(), first.0),
                    second: format!("{}:{native_name}", path.display()),
                });
            }
        }
    }
    Ok(mapped_to_native
        .into_iter()
        .map(|(mapped, (_, _, header))| (mapped, header))
        .collect())
}

fn mapped_safetensors_var_builder<'a, F>(
    paths: &[&Path],
    device: &'a Device,
    dtype: DType,
    remap: F,
) -> Result<VarBuilder<'a>, LoaderError>
where
    F: Fn(&str) -> String,
{
    let mut mapped_to_native = HashMap::new();

    for path in paths {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(LoaderError::Candle)?;
        for (native_name, _) in tensors.tensors() {
            let mapped_name = remap(&native_name);
            if let Some(first) = mapped_to_native.insert(
                mapped_name.clone(),
                (native_name.clone(), (*path).to_path_buf()),
            ) {
                return Err(LoaderError::DuplicateMappedTensor {
                    mapped: mapped_name,
                    first: format!("{}:{}", first.1.display(), first.0),
                    second: format!("{}:{native_name}", path.display()),
                });
            }
        }
    }

    let name_map: HashMap<String, String> = mapped_to_native
        .into_iter()
        .map(|(mapped, (native, _))| (mapped, native))
        .collect();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
        .map_err(LoaderError::Candle)?;
    Ok(vb.rename_f(move |requested| {
        name_map
            .get(requested)
            .cloned()
            .unwrap_or_else(|| requested.to_string())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn save_tensors(path: &Path, tensors: HashMap<String, Tensor>) {
        candle_core::safetensors::save(&tensors, path).expect("save fixture");
    }

    #[test]
    fn native_transformer_is_mapped_without_a_remapped_cache() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("transformer.safetensors");
        let device = Device::Cpu;
        save_tensors(
            &path,
            HashMap::from([(
                "model.diffusion_model.blocks.0.self_attn.q.weight".to_string(),
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).expect("tensor"),
            )]),
        );

        let vb = mapped_safetensors_var_builder(
            &[path.as_path()],
            &device,
            DType::F32,
            remap_transformer_key,
        )
        .expect("mapped var builder");
        let tensor = vb
            .get((2, 2), "blocks.0.attn1.to_q.weight")
            .expect("mapped tensor");
        assert_eq!(
            tensor.to_vec2::<f32>().expect("values"),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
    }

    #[test]
    fn native_vae_is_mapped_without_a_remapped_cache() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("vae.safetensors");
        let device = Device::Cpu;
        save_tensors(
            &path,
            HashMap::from([(
                "conv2.weight".to_string(),
                Tensor::ones((1, 1, 1, 1, 1), DType::F32, &device).expect("tensor"),
            )]),
        );

        let vb = mapped_safetensors_var_builder(&[path.as_path()], &device, DType::F32, |name| {
            remap_vae_key(name).unwrap_or_else(|| name.to_string())
        })
        .expect("mapped var builder");
        assert!(vb.contains_tensor("post_quant_conv.weight"));
        vb.get((1, 1, 1, 1, 1), "post_quant_conv.weight")
            .expect("mapped tensor");
    }

    #[test]
    fn mapped_name_collisions_are_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("duplicate.safetensors");
        let device = Device::Cpu;
        save_tensors(
            &path,
            HashMap::from([
                (
                    "blocks.0.self_attn.q.weight".to_string(),
                    Tensor::zeros((1,), DType::F32, &device).expect("tensor"),
                ),
                (
                    "model.diffusion_model.blocks.0.self_attn.q.weight".to_string(),
                    Tensor::ones((1,), DType::F32, &device).expect("tensor"),
                ),
            ]),
        );

        let error = match mapped_safetensors_var_builder(
            &[path.as_path()],
            &device,
            DType::F32,
            remap_transformer_key,
        ) {
            Ok(_) => panic!("mapped collision must fail"),
            Err(error) => error,
        };
        assert!(
            matches!(error, LoaderError::DuplicateMappedTensor { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn arbitrary_single_safetensors_filename_is_discovered() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("umt5_xxl_fp8_e4m3fn_scaled.safetensors");
        save_tensors(
            &path,
            HashMap::from([(
                "shared.weight".to_string(),
                Tensor::zeros((1,), DType::F32, &Device::Cpu).expect("tensor"),
            )]),
        );

        let discovered = discover_safetensors(temp.path()).expect("discover");

        assert_eq!(discovered, vec![path]);
    }

    #[test]
    fn wan21_transformer_manifest_has_825_tensors() {
        let config = WanFullConfig::wan21_t2v_13b();
        assert_eq!(
            expected_transformer_tensor_names(&config.transformer).len(),
            825
        );
    }

    #[test]
    fn wan21_vae_manifest_has_194_weight_tensors() {
        let config = WanFullConfig::wan21_t2v_13b();
        let shapes = expected_vae_tensor_shapes(&config.vae).expect("manifest");
        assert_eq!(shapes.len(), 194);
    }

    #[test]
    fn indexed_loader_rejects_missing_shard_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path()
                .join("diffusion_pytorch_model.safetensors.index.json"),
            r#"{"weight_map":{"tensor":"missing-00001-of-00001.safetensors"}}"#,
        )
        .expect("index");

        let error = match discover_safetensors(temp.path()) {
            Ok(_) => panic!("missing shard must fail"),
            Err(error) => error,
        };

        assert!(matches!(error, LoaderError::MissingShards { .. }));
    }

    #[test]
    fn indexed_loader_rejects_tensor_in_wrong_shard() {
        let temp = tempfile::tempdir().expect("tempdir");
        let device = Device::Cpu;
        let first = temp.path().join("part-00001.safetensors");
        let second = temp.path().join("part-00002.safetensors");
        save_tensors(
            &first,
            HashMap::from([(
                "tensor".to_string(),
                Tensor::zeros((1,), DType::F32, &device).expect("tensor"),
            )]),
        );
        save_tensors(
            &second,
            HashMap::from([(
                "other".to_string(),
                Tensor::zeros((1,), DType::F32, &device).expect("tensor"),
            )]),
        );
        std::fs::write(
            temp.path().join("diffusion_pytorch_model.safetensors.index.json"),
            r#"{"weight_map":{"tensor":"part-00002.safetensors","other":"part-00001.safetensors"}}"#,
        )
        .expect("index");

        let error = match discover_safetensors(temp.path()) {
            Ok(_) => panic!("wrong shard must fail"),
            Err(error) => error,
        };

        assert!(matches!(error, LoaderError::ShardTensorMismatch { .. }));
    }
}
