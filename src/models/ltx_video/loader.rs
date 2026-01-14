use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("Failed to read file: {path}")]
    FileRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse JSON config: {path}")]
    JsonParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Missing shard files: {missing:?}")]
    MissingShards { missing: Vec<String> },

    #[error("No safetensors files found in directory: {path}")]
    NoSafetensorsFound { path: String },

    #[error("Missing required tensors: {missing:?}")]
    MissingTensors { missing: Vec<String> },

    #[error("Invalid safetensors file: {path}")]
    InvalidSafetensors {
        path: String,
        #[source]
        source: safetensors::SafeTensorError,
    },

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

#[derive(Debug, Clone)]
enum MappingRule {
    Exact {
        from: String,
        to: String,
    },

    Prefix {
        from_prefix: String,
        to_prefix: String,
    },

    Suffix {
        from_suffix: String,
        to_suffix: String,
    },
}

impl MappingRule {
    fn apply(&self, name: &str) -> Option<String> {
        match self {
            MappingRule::Exact { from, to } => {
                if name == from {
                    Some(to.clone())
                } else {
                    None
                }
            }
            MappingRule::Prefix {
                from_prefix,
                to_prefix,
            } => {
                if name.starts_with(from_prefix) {
                    Some(format!("{}{}", to_prefix, &name[from_prefix.len()..]))
                } else {
                    None
                }
            }
            MappingRule::Suffix {
                from_suffix,
                to_suffix,
            } => {
                if name.ends_with(from_suffix) {
                    let base = &name[..name.len() - from_suffix.len()];
                    Some(format!("{}{}", base, to_suffix))
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsIndex {
    pub weight_map: HashMap<String, String>,

    #[serde(default)]
    pub metadata: Option<IndexMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    #[serde(default)]
    pub format: Option<String>,

    #[serde(default)]
    pub total_size: Option<u64>,

    #[serde(default)]
    pub model_type: Option<String>,
}

impl SafetensorsIndex {
    pub fn load(path: impl AsRef<Path>) -> std::result::Result<Self, LoaderError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| LoaderError::FileRead {
            path: path.display().to_string(),
            source: e,
        })?;

        serde_json::from_str(&content).map_err(|e| LoaderError::JsonParse {
            path: path.display().to_string(),
            source: e,
        })
    }

    pub fn shard_files(&self) -> Vec<String> {
        let files: HashSet<_> = self.weight_map.values().collect();
        let mut result: Vec<_> = files.into_iter().cloned().collect();
        result.sort();
        result
    }

    pub fn get_file_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(|s| s.as_str())
    }

    pub fn is_sharded(&self) -> bool {
        self.shard_files().len() > 1
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.weight_map.keys().map(|s| s.as_str()).collect()
    }
}

pub struct WeightLoader {
    device: Device,

    dtype: DType,

    mapping_rules: Vec<MappingRule>,

    strict_mode: bool,
}

impl WeightLoader {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            mapping_rules: Vec::new(),
            strict_mode: false,
        }
    }

    pub fn add_mapping(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.mapping_rules.push(MappingRule::Exact {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    pub fn add_prefix_mapping(
        mut self,
        from_prefix: impl Into<String>,
        to_prefix: impl Into<String>,
    ) -> Self {
        self.mapping_rules.push(MappingRule::Prefix {
            from_prefix: from_prefix.into(),
            to_prefix: to_prefix.into(),
        });
        self
    }

    pub fn add_suffix_mapping(
        mut self,
        from_suffix: impl Into<String>,
        to_suffix: impl Into<String>,
    ) -> Self {
        self.mapping_rules.push(MappingRule::Suffix {
            from_suffix: from_suffix.into(),
            to_suffix: to_suffix.into(),
        });
        self
    }

    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    pub fn is_strict_mode(&self) -> bool {
        self.strict_mode
    }

    pub fn has_mapping(&self, name: &str) -> bool {
        self.mapping_rules
            .iter()
            .any(|rule| rule.apply(name).is_some())
    }

    pub fn map_name(&self, name: &str) -> String {
        let mut current = name.to_string();

        for rule in &self.mapping_rules {
            if let Some(mapped) = rule.apply(&current) {
                current = mapped;
            }
        }

        current
    }

    pub fn load_single(&self, path: impl AsRef<Path>) -> Result<VarBuilder<'_>> {
        let path = path.as_ref();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], self.dtype, &self.device)? };
        Ok(vb)
    }

    pub fn load_sharded(&self, paths: &[PathBuf]) -> Result<VarBuilder<'_>> {
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, self.dtype, &self.device)? };
        Ok(vb)
    }

    pub fn load_from_directory(
        &self,
        dir: impl AsRef<Path>,
    ) -> std::result::Result<VarBuilder<'_>, LoaderError> {
        let dir = dir.as_ref();

        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index = SafetensorsIndex::load(&index_path)?;
            let shard_files = index.shard_files();

            let mut missing = Vec::new();
            let mut paths = Vec::new();

            for shard in &shard_files {
                let shard_path = dir.join(shard);
                if !shard_path.exists() {
                    missing.push(shard.clone());
                } else {
                    paths.push(shard_path);
                }
            }

            if !missing.is_empty() {
                return Err(LoaderError::MissingShards { missing });
            }

            return self.load_sharded(&paths).map_err(LoaderError::from);
        }

        let single_path = dir.join("model.safetensors");
        if single_path.exists() {
            return self.load_single(&single_path).map_err(LoaderError::from);
        }

        let files = find_sharded_files(dir, "").map_err(|e| LoaderError::FileRead {
            path: dir.display().to_string(),
            source: std::io::Error::other(e.to_string()),
        })?;

        if files.is_empty() {
            return Err(LoaderError::NoSafetensorsFound {
                path: dir.display().to_string(),
            });
        }

        if files.len() == 1 {
            self.load_single(&files[0]).map_err(LoaderError::from)
        } else {
            self.load_sharded(&files).map_err(LoaderError::from)
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn get_tensor<S: Into<candle_core::Shape>>(
        &self,
        vb: &VarBuilder,
        shape: S,
        name: &str,
    ) -> Result<Tensor> {
        let mapped_name = self.map_name(name);
        vb.get(shape, &mapped_name)
    }

    pub fn load_all_tensors(&self, path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
        use candle_core::safetensors::load;
        let tensors = load(path, &self.device)?;
        Ok(tensors)
    }
}

pub fn find_sharded_files(dir: impl AsRef<Path>, prefix: &str) -> Result<Vec<PathBuf>> {
    use std::fs;
    let dir = dir.as_ref();
    let mut files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && name.starts_with(prefix)
            && name.ends_with(".safetensors")
        {
            files.push(path);
        }
    }

    files.sort();
    Ok(files)
}

pub fn load_model_config<T: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> std::result::Result<T, LoaderError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| LoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    serde_json::from_str(&content).map_err(|e| LoaderError::JsonParse {
        path: path.display().to_string(),
        source: e,
    })
}

pub fn validate_tensor_names(expected: &[String], actual: &[&str]) -> Vec<String> {
    let actual_set: HashSet<_> = actual.iter().cloned().collect();

    expected
        .iter()
        .filter(|name| !actual_set.contains(name.as_str()))
        .cloned()
        .collect()
}

pub fn list_tensor_names(path: impl AsRef<Path>) -> std::result::Result<Vec<String>, LoaderError> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| LoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
        LoaderError::InvalidSafetensors {
            path: path.display().to_string(),
            source: e,
        }
    })?;

    Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
}

pub fn get_tensor_info(
    path: impl AsRef<Path>,
) -> std::result::Result<HashMap<String, TensorInfo>, LoaderError> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| LoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
        LoaderError::InvalidSafetensors {
            path: path.display().to_string(),
            source: e,
        }
    })?;

    let mut info = HashMap::new();
    for name in tensors.names() {
        if let Ok(view) = tensors.tensor(name) {
            info.insert(
                name.to_string(),
                TensorInfo {
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            );
        }
    }

    Ok(info)
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,

    pub shape: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_loader_creation() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32);
        assert_eq!(loader.dtype, DType::F32);
    }

    #[test]
    fn test_name_mapping_exact() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32)
            .add_mapping("model.diffusion_model", "diffusion_model");

        assert_eq!(loader.map_name("model.diffusion_model"), "diffusion_model");

        assert_eq!(loader.map_name("other.name"), "other.name");
    }

    #[test]
    fn test_name_mapping_prefix() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32).add_prefix_mapping("model.", "");

        assert_eq!(
            loader.map_name("model.transformer.weight"),
            "transformer.weight"
        );

        assert_eq!(loader.map_name("other.weight"), "other.weight");
    }

    #[test]
    fn test_name_mapping_suffix() {
        let loader =
            WeightLoader::new(Device::Cpu, DType::F32).add_suffix_mapping(".gamma", ".weight");

        assert_eq!(loader.map_name("layer_norm.gamma"), "layer_norm.weight");
    }

    #[test]
    fn test_name_mapping_chain() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32)
            .add_prefix_mapping("model.", "")
            .add_suffix_mapping(".gamma", ".weight");

        assert_eq!(
            loader.map_name("model.layer_norm.gamma"),
            "layer_norm.weight"
        );
    }

    #[test]
    fn test_validate_tensor_names() {
        let expected = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let actual = vec!["a", "b"];

        let missing = validate_tensor_names(&expected, &actual);
        assert_eq!(missing, vec!["c".to_string()]);
    }

    #[test]
    fn test_safetensors_index_shard_files() {
        let mut weight_map = HashMap::new();
        weight_map.insert("a".to_string(), "shard1.safetensors".to_string());
        weight_map.insert("b".to_string(), "shard1.safetensors".to_string());
        weight_map.insert("c".to_string(), "shard2.safetensors".to_string());

        let index = SafetensorsIndex {
            weight_map,
            metadata: None,
        };

        let shards = index.shard_files();
        assert_eq!(shards.len(), 2);
        assert!(shards.contains(&"shard1.safetensors".to_string()));
        assert!(shards.contains(&"shard2.safetensors".to_string()));
    }
}
