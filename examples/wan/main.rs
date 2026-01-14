// Wan2.1+ Text-to-Video Generation Example
//
// This example demonstrates text-to-video generation using the Wan2.1 model.
// It follows the same pattern as the LTX-Video example.
//
// Usage:
//   cargo run --example wan --release --features flash-attn,cudnn -- \
//       --local-weights ./models/Wan2.1-T2V-1.3B \
//       --prompt "A cat walking on the grass"

use candle_core::{DType, Device, Tensor};
use clap::Parser;
use std::path::{Path, PathBuf};
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during Wan example execution.
#[derive(Debug, Error)]
pub enum WanExampleError {
    #[error("Model file not found: {path}")]
    ModelNotFound { path: String },

    #[error("Tokenizer file not found: {path}. Please download tokenizer.json for UMT5.")]
    TokenizerNotFound { path: String },

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),
}

// =============================================================================
// Weight Path Resolution
// =============================================================================

/// Resolved paths for all Wan model components.
#[derive(Debug, Clone)]
pub struct WanModelPaths {
    /// Path to transformer weights (wan2.1_t2v_1.3B_fp16.safetensors)
    pub transformer: PathBuf,
    /// Path to VAE weights directory
    pub vae_dir: PathBuf,
    /// Path to text encoder GGUF file
    pub text_encoder: PathBuf,
    /// Path to tokenizer.json
    pub tokenizer: PathBuf,
}

impl WanModelPaths {
    /// Resolve all model paths from a base directory.
    ///
    /// Expected directory structure:
    /// ```text
    /// {base_dir}/
    ///   wan2.1_t2v_1.3B_fp16.safetensors
    ///   vae/
    ///     wan_2.1_vae.safetensors (or diffusion_pytorch_model.safetensors)
    ///   text_encoder_gguf/
    ///     umt5-xxl-encoder-Q5_K_M.gguf (or similar)
    ///     tokenizer.json
    /// ```
    pub fn resolve(base_dir: impl AsRef<Path>) -> Result<Self, WanExampleError> {
        let base = base_dir.as_ref();

        // Resolve transformer path
        let transformer = Self::resolve_transformer_path(base)?;

        // Resolve VAE directory
        let vae_dir = Self::resolve_vae_dir(base)?;

        // Resolve text encoder path
        let text_encoder = Self::resolve_text_encoder_path(base)?;

        // Resolve tokenizer path
        let tokenizer = Self::resolve_tokenizer_path(base)?;

        Ok(Self {
            transformer,
            vae_dir,
            text_encoder,
            tokenizer,
        })
    }

    /// Resolve transformer weights path.
    fn resolve_transformer_path(base: &Path) -> Result<PathBuf, WanExampleError> {
        // Try common file names
        let candidates = [
            "wan2.1_t2v_1.3B_fp16.safetensors",
            "wan2.1_t2v_14B_fp16.safetensors",
            "transformer.safetensors",
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
        ];

        for candidate in &candidates {
            let path = base.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        // Also check transformer subdirectory
        let transformer_dir = base.join("transformer");
        if transformer_dir.exists() {
            for candidate in &["diffusion_pytorch_model.safetensors", "model.safetensors"] {
                let path = transformer_dir.join(candidate);
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        Err(WanExampleError::ModelNotFound {
            path: format!(
                "{}/wan2.1_t2v_1.3B_fp16.safetensors (or similar)",
                base.display()
            ),
        })
    }

    /// Resolve VAE directory path.
    fn resolve_vae_dir(base: &Path) -> Result<PathBuf, WanExampleError> {
        let vae_dir = base.join("vae");
        if !vae_dir.exists() {
            return Err(WanExampleError::ModelNotFound {
                path: format!("{}/vae/", base.display()),
            });
        }

        // Verify at least one VAE weights file exists
        let vae_candidates = [
            "wan_2.1_vae.safetensors",
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
            "vae.safetensors",
        ];

        for candidate in &vae_candidates {
            if vae_dir.join(candidate).exists() {
                return Ok(vae_dir);
            }
        }

        Err(WanExampleError::ModelNotFound {
            path: format!("{}/vae/*.safetensors", base.display()),
        })
    }

    /// Resolve text encoder GGUF path.
    fn resolve_text_encoder_path(base: &Path) -> Result<PathBuf, WanExampleError> {
        let te_dir = base.join("text_encoder_gguf");

        if te_dir.exists() {
            // Look for GGUF files
            let gguf_candidates = [
                "umt5-xxl-encoder-Q5_K_M.gguf",
                "umt5-xxl-encoder-Q8_0.gguf",
                "umt5-xxl.gguf",
                "model.gguf",
            ];

            for candidate in &gguf_candidates {
                let path = te_dir.join(candidate);
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        // Also check base directory for GGUF files
        let base_gguf_candidates = ["umt5-xxl-encoder-Q5_K_M.gguf", "umt5-xxl-encoder-Q8_0.gguf"];

        for candidate in &base_gguf_candidates {
            let path = base.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(WanExampleError::ModelNotFound {
            path: format!(
                "{}/text_encoder_gguf/umt5-xxl-encoder-*.gguf",
                base.display()
            ),
        })
    }

    /// Resolve tokenizer.json path.
    fn resolve_tokenizer_path(base: &Path) -> Result<PathBuf, WanExampleError> {
        // Check common locations
        let candidates = [
            base.join("text_encoder_gguf").join("tokenizer.json"),
            base.join("text_encoder").join("tokenizer.json"),
            base.join("tokenizer").join("tokenizer.json"),
            base.join("tokenizer.json"),
        ];

        for path in &candidates {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        Err(WanExampleError::TokenizerNotFound {
            path: format!("{}/text_encoder_gguf/tokenizer.json", base.display()),
        })
    }

    /// Get the VAE weights file path.
    pub fn vae_weights(&self) -> Result<PathBuf, WanExampleError> {
        let candidates = [
            "wan_2.1_vae.safetensors",
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
            "vae.safetensors",
        ];

        for candidate in &candidates {
            let path = self.vae_dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(WanExampleError::ModelNotFound {
            path: format!("{}/*.safetensors", self.vae_dir.display()),
        })
    }
}

// =============================================================================
// Model Loader
// =============================================================================

/// Loader for Wan model components.
pub struct WanModelLoader {
    device: Device,
    dtype: DType,
}

impl WanModelLoader {
    /// Create a new model loader.
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { device, dtype }
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Load transformer from resolved paths.
    pub fn load_transformer(
        &self,
        paths: &WanModelPaths,
    ) -> Result<candle_video::models::wan::WanTransformer3DModel, WanExampleError> {
        use candle_video::models::wan::{WanTransformer3DConfig, load_transformer};

        println!("  Loading transformer from {:?}...", paths.transformer);

        let config = WanTransformer3DConfig::wan_t2v_1_3b();
        load_transformer(&paths.transformer, config, &self.device, self.dtype)
            .map_err(|e| WanExampleError::ModelLoading(e.to_string()))
    }

    /// Load VAE from resolved paths.
    pub fn load_vae(
        &self,
        paths: &WanModelPaths,
    ) -> Result<candle_video::models::wan::AutoencoderKLWan, WanExampleError> {
        use candle_video::models::wan::{AutoencoderKLWanConfig, load_vae};

        let vae_weights = paths.vae_weights()?;
        println!("  Loading VAE from {:?}...", vae_weights);

        let config = AutoencoderKLWanConfig::wan_2_1();
        load_vae(&vae_weights, config, &self.device, self.dtype)
            .map_err(|e| WanExampleError::ModelLoading(e.to_string()))
    }

    /// Load quantized text encoder from resolved paths.
    pub fn load_text_encoder(
        &self,
        paths: &WanModelPaths,
    ) -> Result<candle_video::models::wan::QuantizedUMT5Encoder, WanExampleError> {
        use candle_video::models::wan::load_quantized_text_encoder;

        println!("  Loading text encoder from {:?}...", paths.text_encoder);

        // Quantized models run on CPU for efficiency
        let te_device = Device::Cpu;
        load_quantized_text_encoder(&paths.text_encoder, &te_device)
            .map_err(|e| WanExampleError::ModelLoading(e.to_string()))
    }

    /// Load tokenizer from resolved paths.
    pub fn load_tokenizer(
        &self,
        paths: &WanModelPaths,
    ) -> Result<candle_video::models::wan::UMT5Tokenizer, WanExampleError> {
        use candle_video::models::wan::load_umt5_tokenizer;

        println!("  Loading tokenizer from {:?}...", paths.tokenizer);

        // Max length 512 for Wan models
        load_umt5_tokenizer(&paths.tokenizer, Some(512)).map_err(WanExampleError::Tokenizer)
    }
}

// =============================================================================
// Tokenizer Adapter
// =============================================================================

/// Adapter for UMT5Tokenizer that provides convenient text encoding.
///
/// This wraps the UMT5Tokenizer and provides methods for tokenizing text
/// with padding/truncation to a maximum length, returning tensors suitable
/// for the text encoder.
pub struct TokenizerAdapter {
    tokenizer: candle_video::models::wan::UMT5Tokenizer,
    device: Device,
    max_length: usize,
}

impl TokenizerAdapter {
    /// Create a new TokenizerAdapter.
    ///
    /// # Arguments
    /// * `tokenizer` - The UMT5Tokenizer to wrap
    /// * `device` - Device for output tensors
    /// * `max_length` - Maximum sequence length (512 for Wan)
    pub fn new(
        tokenizer: candle_video::models::wan::UMT5Tokenizer,
        device: Device,
        max_length: usize,
    ) -> Self {
        Self {
            tokenizer,
            device,
            max_length,
        }
    }

    /// Get the maximum sequence length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Tokenize text and return (input_ids, attention_mask) tensors.
    ///
    /// # Arguments
    /// * `text` - Text to tokenize
    ///
    /// # Returns
    /// Tuple of (input_ids, attention_mask) tensors, both with shape [1, max_length]
    pub fn encode(&self, text: &str) -> Result<(Tensor, Tensor), WanExampleError> {
        use candle_video::models::wan::Tokenizer as TokenizerTrait;

        let texts = vec![text.to_string()];
        let (input_ids, attention_mask) = self
            .tokenizer
            .encode(&texts, self.max_length)
            .map_err(|e| WanExampleError::Tokenizer(e.to_string()))?;

        // Convert to tensors
        let input_ids_flat: Vec<i64> = input_ids[0].iter().map(|&v| v as i64).collect();
        let attention_mask_flat: Vec<i64> = attention_mask[0].iter().map(|&v| v as i64).collect();

        let input_ids_tensor = Tensor::from_vec(input_ids_flat, (1, self.max_length), &self.device)
            .map_err(WanExampleError::Candle)?;
        let attention_mask_tensor =
            Tensor::from_vec(attention_mask_flat, (1, self.max_length), &self.device)
                .map_err(WanExampleError::Candle)?;

        Ok((input_ids_tensor, attention_mask_tensor))
    }

    /// Tokenize a batch of texts.
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to tokenize
    ///
    /// # Returns
    /// Tuple of (input_ids, attention_mask) tensors, both with shape [batch, max_length]
    pub fn encode_batch(&self, texts: &[String]) -> Result<(Tensor, Tensor), WanExampleError> {
        use candle_video::models::wan::Tokenizer as TokenizerTrait;

        let (input_ids, attention_mask) = self
            .tokenizer
            .encode(texts, self.max_length)
            .map_err(|e| WanExampleError::Tokenizer(e.to_string()))?;

        let batch_size = texts.len();

        // Flatten and convert to tensors
        let input_ids_flat: Vec<i64> = input_ids.into_iter().flatten().map(|v| v as i64).collect();
        let attention_mask_flat: Vec<i64> = attention_mask
            .into_iter()
            .flatten()
            .map(|v| v as i64)
            .collect();

        let input_ids_tensor =
            Tensor::from_vec(input_ids_flat, (batch_size, self.max_length), &self.device)
                .map_err(WanExampleError::Candle)?;
        let attention_mask_tensor = Tensor::from_vec(
            attention_mask_flat,
            (batch_size, self.max_length),
            &self.device,
        )
        .map_err(WanExampleError::Candle)?;

        Ok((input_ids_tensor, attention_mask_tensor))
    }
}

// =============================================================================
// Prompt Encoding
// =============================================================================

/// Result of encoding prompts for the pipeline.
#[derive(Debug)]
pub struct EncodedPrompts {
    /// Positive prompt embeddings [1, seq_len, hidden_dim]
    pub prompt_embeds: Tensor,
    /// Positive prompt attention mask [1, seq_len]
    pub prompt_attention_mask: Tensor,
    /// Negative prompt embeddings (only if guidance_scale > 1.0)
    pub negative_prompt_embeds: Option<Tensor>,
    /// Negative prompt attention mask (only if guidance_scale > 1.0)
    pub negative_prompt_attention_mask: Option<Tensor>,
}

/// Encode prompts for the Wan pipeline.
///
/// This function handles:
/// - Tokenizing the positive prompt
/// - Conditionally tokenizing and encoding the negative prompt when guidance_scale > 1.0
/// - Converting embeddings to the transformer's dtype
///
/// # Arguments
/// * `text_encoder` - The text encoder (UMT5)
/// * `tokenizer` - The tokenizer adapter
/// * `prompt` - Positive text prompt
/// * `negative_prompt` - Negative text prompt (used when guidance_scale > 1.0)
/// * `guidance_scale` - CFG guidance scale
/// * `transformer_dtype` - Target dtype for embeddings
/// * `device` - Target device for embeddings (CUDA for transformer)
///
/// # Returns
/// EncodedPrompts containing embeddings and attention masks
pub fn encode_prompts<E: candle_video::models::wan::TextEncoder>(
    text_encoder: &mut E,
    tokenizer: &TokenizerAdapter,
    prompt: &str,
    negative_prompt: &str,
    guidance_scale: f32,
    transformer_dtype: DType,
    device: &Device,
) -> Result<EncodedPrompts, WanExampleError> {
    // Encode positive prompt
    let (input_ids, attention_mask) = tokenizer.encode(prompt)?;
    let prompt_embeds = text_encoder
        .encode(&input_ids, &attention_mask)
        .map_err(WanExampleError::Candle)?;

    // Convert to transformer dtype and move to target device
    let prompt_embeds = prompt_embeds
        .to_dtype(transformer_dtype)
        .map_err(WanExampleError::Candle)?
        .to_device(device)
        .map_err(WanExampleError::Candle)?;
    let prompt_attention_mask = attention_mask
        .to_dtype(DType::F32)
        .map_err(WanExampleError::Candle)?
        .to_device(device)
        .map_err(WanExampleError::Candle)?;

    // Conditionally encode negative prompt for CFG
    let (negative_prompt_embeds, negative_prompt_attention_mask) = if guidance_scale > 1.0 {
        let (neg_input_ids, neg_attention_mask) = tokenizer.encode(negative_prompt)?;
        let neg_embeds = text_encoder
            .encode(&neg_input_ids, &neg_attention_mask)
            .map_err(WanExampleError::Candle)?;

        // Convert to transformer dtype and move to target device
        let neg_embeds = neg_embeds
            .to_dtype(transformer_dtype)
            .map_err(WanExampleError::Candle)?
            .to_device(device)
            .map_err(WanExampleError::Candle)?;
        let neg_attention_mask = neg_attention_mask
            .to_dtype(DType::F32)
            .map_err(WanExampleError::Candle)?
            .to_device(device)
            .map_err(WanExampleError::Candle)?;

        (Some(neg_embeds), Some(neg_attention_mask))
    } else {
        (None, None)
    };

    Ok(EncodedPrompts {
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    })
}

// =============================================================================
// Latent Initialization
// =============================================================================

/// Constants for Wan VAE compression factors.
pub const WAN_VAE_SCALE_FACTOR_SPATIAL: usize = 8;
pub const WAN_VAE_SCALE_FACTOR_TEMPORAL: usize = 4;
pub const WAN_LATENT_CHANNELS: usize = 16;

/// Calculate latent dimensions from video dimensions.
///
/// # Arguments
/// * `height` - Video height in pixels
/// * `width` - Video width in pixels
/// * `num_frames` - Number of video frames
///
/// # Returns
/// Tuple of (latent_frames, latent_height, latent_width)
///
/// # Formula
/// - latent_frames = (num_frames - 1) / 4 + 1
/// - latent_height = height / 8
/// - latent_width = width / 8
pub fn calculate_latent_dims(
    height: usize,
    width: usize,
    num_frames: usize,
) -> (usize, usize, usize) {
    let latent_frames = (num_frames - 1) / WAN_VAE_SCALE_FACTOR_TEMPORAL + 1;
    let latent_height = height / WAN_VAE_SCALE_FACTOR_SPATIAL;
    let latent_width = width / WAN_VAE_SCALE_FACTOR_SPATIAL;
    (latent_frames, latent_height, latent_width)
}

/// Prepare initial latents for video generation.
///
/// Generates random latents with shape [1, 16, F', H', W'] where:
/// - F' = (num_frames - 1) / 4 + 1
/// - H' = height / 8
/// - W' = width / 8
///
/// # Arguments
/// * `height` - Video height in pixels (should be divisible by 16)
/// * `width` - Video width in pixels (should be divisible by 16)
/// * `num_frames` - Number of video frames
/// * `device` - Device to create tensor on
/// * `seed` - Optional seed for deterministic RNG
///
/// # Returns
/// Latent tensor with shape [1, 16, latent_frames, latent_height, latent_width]
pub fn prepare_latents(
    height: usize,
    width: usize,
    num_frames: usize,
    device: &Device,
    seed: Option<u64>,
) -> Result<Tensor, WanExampleError> {
    use candle_video::utils::deterministic_rng::Pcg32;

    let (latent_frames, latent_height, latent_width) =
        calculate_latent_dims(height, width, num_frames);

    let shape = (
        1,
        WAN_LATENT_CHANNELS,
        latent_frames,
        latent_height,
        latent_width,
    );

    // Use deterministic RNG if seed provided, otherwise generate random seed
    let actual_seed = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        now.as_secs() ^ (now.subsec_nanos() as u64)
    });

    let mut rng = Pcg32::new(actual_seed, 1442695040888963407);
    let latents = rng.randn(shape, device).map_err(WanExampleError::Candle)?;

    Ok(latents)
}

/// Get the seed value that will be used (for printing/logging).
///
/// Returns the provided seed or generates a new one from system time.
pub fn get_or_generate_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        now.as_secs() ^ (now.subsec_nanos() as u64)
    })
}

// =============================================================================
// Scheduler Setup
// =============================================================================

use candle_video::interfaces::flow_match_scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
};
use candle_video::interfaces::scheduler_mixin::SchedulerMixin;

/// Create and configure the FlowMatchEulerDiscreteScheduler for Wan.
///
/// # Arguments
/// * `flow_shift` - Flow shift value (5.0 for 720p, 3.0 for 480p)
/// * `num_steps` - Number of inference steps
/// * `device` - Device for scheduler tensors
///
/// # Returns
/// Configured scheduler with timesteps set
///
/// # Requirements
/// - Creates FlowMatchEulerDiscreteScheduler with flow_shift
/// - Sets timesteps for specified number of steps
pub fn create_scheduler(
    flow_shift: f32,
    num_steps: usize,
    device: &Device,
) -> Result<FlowMatchEulerDiscreteScheduler, WanExampleError> {
    // Create scheduler config with flow shift
    let config = FlowMatchEulerDiscreteSchedulerConfig {
        shift: flow_shift,
        use_dynamic_shifting: false,
        ..Default::default()
    };

    // Create scheduler
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(config)
        .map_err(|e| WanExampleError::ModelLoading(format!("Failed to create scheduler: {}", e)))?;

    // Set timesteps
    SchedulerMixin::set_timesteps(&mut scheduler, num_steps, device)
        .map_err(|e| WanExampleError::ModelLoading(format!("Failed to set timesteps: {}", e)))?;

    Ok(scheduler)
}

/// Get timesteps from the scheduler as a vector of f64.
pub fn get_timesteps(scheduler: &FlowMatchEulerDiscreteScheduler) -> Vec<f64> {
    SchedulerMixin::timesteps(scheduler).to_vec()
}

// =============================================================================
// Denoising Loop
// =============================================================================

/// Result of the denoising loop.
#[derive(Debug)]
pub struct DenoisingResult {
    /// Final denoised latents
    pub latents: Tensor,
    /// Number of steps completed
    pub steps_completed: usize,
}

/// Run the denoising loop for video generation.
///
/// This function performs the iterative denoising process:
/// 1. For each timestep, run the transformer forward pass
/// 2. Apply classifier-free guidance (CFG) when guidance_scale > 1.0
/// 3. Use the scheduler to update latents
/// 4. Print progress every 10 steps
///
/// Memory optimization strategy (based on official Wan2.1 implementation):
/// - Explicit tensor drops to release GPU memory ASAP
/// - Synchronize after each forward pass to ensure memory is freed
/// - Compute CFG incrementally to avoid holding both predictions
///
/// # Arguments
/// * `transformer` - The Wan transformer model
/// * `scheduler` - The flow match scheduler
/// * `latents` - Initial noisy latents
/// * `encoded_prompts` - Encoded text prompts
/// * `guidance_scale` - CFG guidance scale
/// * `device` - Device for computations
///
/// # Returns
/// DenoisingResult containing the final denoised latents
pub fn run_denoising_loop(
    transformer: &candle_video::models::wan::WanTransformer3DModel,
    scheduler: &mut FlowMatchEulerDiscreteScheduler,
    mut latents: Tensor,
    encoded_prompts: &EncodedPrompts,
    guidance_scale: f32,
    device: &Device,
) -> Result<DenoisingResult, WanExampleError> {
    let timesteps = get_timesteps(scheduler);
    let num_steps = timesteps.len();
    let do_cfg = guidance_scale > 1.0;
    let transformer_dtype = transformer.dtype();

    println!("\nRunning denoising loop ({} steps)...", num_steps);

    for (i, &t) in timesteps.iter().enumerate() {
        // Create timestep tensor (must match transformer dtype)
        let timestep = Tensor::from_vec(vec![t as f32], (1,), device)
            .map_err(WanExampleError::Candle)?
            .to_dtype(transformer_dtype)
            .map_err(WanExampleError::Candle)?;

        // Convert latents to transformer dtype
        let latent_input = latents
            .to_dtype(transformer_dtype)
            .map_err(WanExampleError::Candle)?;

        // Compute noise prediction with CFG
        // Memory optimization: compute unconditional first, then conditional,
        // and apply CFG formula incrementally to minimize peak memory
        let noise_pred = if do_cfg {
            let neg_embeds = encoded_prompts
                .negative_prompt_embeds
                .as_ref()
                .ok_or_else(|| {
                    WanExampleError::ModelLoading(
                        "Negative prompt embeddings required for CFG".to_string(),
                    )
                })?;

            // Step 1: Unconditional forward pass (negative prompt)
            // We compute this first and keep it as the base for CFG
            let noise_pred_uncond = transformer
                .forward(&latent_input, &timestep, neg_embeds, None, false)
                .map_err(WanExampleError::Candle)?;

            let noise_pred_uncond = match noise_pred_uncond {
                Ok(output) => output.sample,
                Err(tensor) => tensor,
            };

            // Sync GPU to release intermediate tensors from uncond pass
            if let Device::Cuda(_) = device {
                device.synchronize().map_err(WanExampleError::Candle)?;
            }

            // Step 2: Conditional forward pass (positive prompt)
            let noise_pred_cond = transformer
                .forward(
                    &latent_input,
                    &timestep,
                    &encoded_prompts.prompt_embeds,
                    None,
                    false,
                )
                .map_err(WanExampleError::Candle)?;

            let noise_pred_cond = match noise_pred_cond {
                Ok(output) => output.sample,
                Err(tensor) => tensor,
            };

            // Sync GPU to release intermediate tensors from cond pass
            if let Device::Cuda(_) = device {
                device.synchronize().map_err(WanExampleError::Candle)?;
            }

            // Step 3: Apply CFG formula: uncond + scale * (cond - uncond)
            // Compute difference first
            let diff = noise_pred_cond
                .sub(&noise_pred_uncond)
                .map_err(WanExampleError::Candle)?;

            // Drop cond tensor explicitly - we only need diff now
            drop(noise_pred_cond);

            // Scale the difference
            let scaled = diff
                .affine(guidance_scale as f64, 0.0)
                .map_err(WanExampleError::Candle)?;

            // Drop diff tensor explicitly
            drop(diff);

            // Add to uncond to get final prediction
            let result = noise_pred_uncond
                .add(&scaled)
                .map_err(WanExampleError::Candle)?;

            // Drop intermediate tensors
            drop(scaled);
            drop(noise_pred_uncond);

            result
        } else {
            // No CFG - single forward pass
            let noise_pred = transformer
                .forward(
                    &latent_input,
                    &timestep,
                    &encoded_prompts.prompt_embeds,
                    None,
                    false,
                )
                .map_err(WanExampleError::Candle)?;

            match noise_pred {
                Ok(output) => output.sample,
                Err(tensor) => tensor,
            }
        };

        // Drop latent_input - we're done with it
        drop(latent_input);
        drop(timestep);

        // Scheduler step to update latents
        let step_output = SchedulerMixin::step(scheduler, &noise_pred, t, &latents)
            .map_err(|e| WanExampleError::ModelLoading(format!("Scheduler step failed: {}", e)))?;

        // Drop old latents and noise_pred before assigning new latents
        drop(noise_pred);
        let old_latents = std::mem::replace(&mut latents, step_output.prev_sample);
        drop(old_latents);

        // Sync GPU at end of step to ensure all memory is released
        if let Device::Cuda(_) = device {
            device.synchronize().map_err(WanExampleError::Candle)?;
        }

        // Print progress every 10 steps
        if i % 10 == 0 || i == num_steps - 1 {
            println!("  Step {}/{}", i + 1, num_steps);
        }
    }

    println!("  ✓ Denoising complete");

    Ok(DenoisingResult {
        latents,
        steps_completed: num_steps,
    })
}

// =============================================================================
// VAE Decoding
// =============================================================================

/// Result of VAE decoding.
#[derive(Debug)]
pub struct DecodeResult {
    /// Decoded video tensor with shape [1, 3, F, H, W]
    /// Values are in range [-1, 1] (clamped by VAE)
    pub video: Tensor,
}

/// Decode latents to video using the Wan VAE.
///
/// This function performs:
/// 1. Denormalize latents using VAE's mean/std statistics
/// 2. Decode with VAE to produce video tensor
/// 3. Return video tensor with shape [1, 3, F, H, W]
///
/// # Arguments
/// * `vae` - The Wan VAE model
/// * `latents` - Denoised latents with shape [1, 16, F', H', W']
///
/// # Returns
/// DecodeResult containing the decoded video tensor
///
/// # Requirements
/// - Denormalize latents
/// - Decode with VAE
/// - Return video tensor [1, 3, F, H, W]
pub fn decode_latents(
    vae: &candle_video::models::wan::AutoencoderKLWan,
    latents: &Tensor,
    vae_dtype: DType,
) -> Result<DecodeResult, WanExampleError> {
    println!("\nDecoding latents with VAE...");

    // Convert latents to VAE dtype (scheduler outputs F32, VAE expects BF16)
    let latents = latents
        .to_dtype(vae_dtype)
        .map_err(WanExampleError::Candle)?;

    // Step 1: Denormalize latents using VAE's mean/std statistics
    // This reverses the normalization applied during encoding
    let denormalized = vae
        .denormalize_latents(&latents)
        .map_err(WanExampleError::Candle)?;

    println!("  ✓ Latents denormalized: {:?}", denormalized.dims());

    // Step 2: Decode with VAE
    // The VAE decode method handles:
    // - Post-quantization convolution
    // - Frame-by-frame decoding with causal convolutions
    // - Output clamping to [-1, 1]
    let video = vae.decode(&denormalized).map_err(WanExampleError::Candle)?;

    println!("  ✓ VAE decode complete: {:?}", video.dims());

    Ok(DecodeResult { video })
}

// =============================================================================
// CLI Arguments
// =============================================================================

/// CLI arguments for Wan2.1+ Text-to-Video Generation.
///
/// All arguments have sensible defaults for 480p video generation.
/// For 720p, consider using --flow-shift 5.0 (default).
/// For 480p, consider using --flow-shift 3.0.
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(author, version, about = "Wan2.1+ Text-to-Video Generation", long_about = None)]
pub struct Args {
    /// Text prompt for video generation
    #[arg(long, default_value = "A cat walking on the grass")]
    pub prompt: String,

    /// Negative prompt for CFG guidance
    #[arg(long, default_value = "")]
    pub negative_prompt: String,

    /// Width of the generated video (must be divisible by 16)
    #[arg(long, default_value_t = 832)]
    pub width: usize,

    /// Height of the generated video (must be divisible by 16)
    #[arg(long, default_value_t = 480)]
    pub height: usize,

    /// Number of frames to generate
    #[arg(long, default_value_t = 81)]
    pub num_frames: usize,

    /// Number of denoising steps
    #[arg(long, default_value_t = 50)]
    pub steps: usize,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 5.0)]
    pub guidance_scale: f32,

    /// Path to local model weights directory
    #[arg(long)]
    pub local_weights: Option<String>,

    /// Output directory for generated video
    #[arg(long, default_value = "output")]
    pub output_dir: String,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Run on CPU instead of GPU
    #[arg(long)]
    pub cpu: bool,

    /// Save individual PNG frames
    #[arg(long)]
    pub frames: bool,

    /// Save as GIF animation
    #[arg(long)]
    pub gif: bool,

    /// Flow shift for scheduler (5.0 for 720p, 3.0 for 480p)
    #[arg(long, default_value_t = 5.0)]
    pub flow_shift: f32,
}

// =============================================================================
// Input Validation
// =============================================================================

/// Validate video dimensions.
///
/// Checks that height and width are divisible by 16, which is required
/// for proper VAE encoding/decoding.
///
/// # Arguments
/// * `height` - Video height in pixels
/// * `width` - Video width in pixels
///
/// # Returns
/// Ok(()) if dimensions are valid, or WanExampleError::InvalidDimensions if not.
///
/// # Requirements
/// - Height must be divisible by 16
/// - Width must be divisible by 16
pub fn validate_dimensions(height: usize, width: usize) -> Result<(), WanExampleError> {
    let mut errors = Vec::new();

    if !height.is_multiple_of(16) {
        errors.push(format!(
            "height {} is not divisible by 16 (nearest valid: {})",
            height,
            (height / 16) * 16
        ));
    }

    if !width.is_multiple_of(16) {
        errors.push(format!(
            "width {} is not divisible by 16 (nearest valid: {})",
            width,
            (width / 16) * 16
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(WanExampleError::InvalidDimensions(errors.join("; ")))
    }
}

// =============================================================================
// Output Handling
// =============================================================================

/// A single RGB frame extracted from the video tensor.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Raw RGB pixel data (row-major, RGB order)
    pub data: Vec<u8>,
    /// Frame width in pixels
    pub width: usize,
    /// Frame height in pixels
    pub height: usize,
}

/// Result of converting video tensor to frames.
#[derive(Debug)]
pub struct VideoFrames {
    /// List of RGB frames
    pub frames: Vec<VideoFrame>,
    /// Video width in pixels
    pub width: usize,
    /// Video height in pixels
    pub height: usize,
}

/// Convert video tensor to list of RGB frames.
///
/// This function converts a video tensor from [B, C, F, H, W] format to a list
/// of RGB frames suitable for saving as PNG or GIF.
///
/// # Arguments
/// * `video` - Video tensor with shape [B, C, F, H, W] where:
///   - B = batch size (typically 1)
///   - C = channels (3 for RGB)
///   - F = number of frames
///   - H = height
///   - W = width
///   - Values are in range [-1, 1] from VAE output
///
/// # Returns
/// VideoFrames containing list of RGB frames with pixel values in [0, 255]
///
/// # Requirements
/// - Convert [B, C, F, H, W] to list of RGB frames
/// - Clamp values to [0, 255]
/// - Convert to u8
pub fn video_tensor_to_frames(video: &Tensor) -> Result<VideoFrames, WanExampleError> {
    use candle_core::IndexOp;

    let dims = video.dims();
    if dims.len() != 5 {
        return Err(WanExampleError::InvalidDimensions(format!(
            "Expected 5D tensor [B, C, F, H, W], got {}D",
            dims.len()
        )));
    }

    let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

    if c != 3 {
        return Err(WanExampleError::InvalidDimensions(format!(
            "Expected 3 channels (RGB), got {}",
            c
        )));
    }

    let mut frames = Vec::with_capacity(b * f);

    for batch_idx in 0..b {
        for frame_idx in 0..f {
            // Extract frame: [C, H, W]
            let frame = video
                .i((batch_idx, .., frame_idx, .., ..))
                .map_err(WanExampleError::Candle)?;

            // Permute from [C, H, W] to [H, W, C] for row-major RGB layout
            let frame = frame.permute((1, 2, 0)).map_err(WanExampleError::Candle)?;

            // Convert from [-1, 1] to [0, 255]
            // Formula: pixel = (value + 1) * 127.5
            let frame = frame
                .affine(127.5, 127.5)
                .map_err(WanExampleError::Candle)?;

            // Clamp to [0, 255]
            let frame = frame.clamp(0.0, 255.0).map_err(WanExampleError::Candle)?;

            // Convert to u8
            let frame = frame.to_dtype(DType::U8).map_err(WanExampleError::Candle)?;

            // Flatten and extract data
            let data: Vec<u8> = frame
                .flatten_all()
                .map_err(WanExampleError::Candle)?
                .to_vec1()
                .map_err(WanExampleError::Candle)?;

            frames.push(VideoFrame {
                data,
                width: w,
                height: h,
            });
        }
    }

    Ok(VideoFrames {
        frames,
        width: w,
        height: h,
    })
}

/// Save individual frames as PNG files.
///
/// Creates the output directory if it doesn't exist and saves each frame
/// as a numbered PNG file (frame_0000.png, frame_0001.png, etc.).
///
/// # Arguments
/// * `frames` - VideoFrames containing the frames to save
/// * `output_dir` - Directory to save frames to
///
/// # Returns
/// Ok(()) on success, or error if saving fails
///
/// # Requirements
/// - Create output directory if needed
/// - Save each frame as PNG
pub fn save_frames_as_png(frames: &VideoFrames, output_dir: &str) -> Result<(), WanExampleError> {
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    println!("\nSaving {} frames as PNG...", frames.frames.len());

    for (i, frame) in frames.frames.iter().enumerate() {
        let filename = format!("{}/frame_{:04}.png", output_dir, i);
        image::save_buffer(
            &filename,
            &frame.data,
            frame.width as u32,
            frame.height as u32,
            image::ColorType::Rgb8,
        )
        .map_err(|e| WanExampleError::Io(std::io::Error::other(e)))?;

        // Print progress every 10 frames
        if i % 10 == 0 || i == frames.frames.len() - 1 {
            println!("  Saved frame {}/{}", i + 1, frames.frames.len());
        }
    }

    println!(
        "  ✓ Saved {} frames to {}/",
        frames.frames.len(),
        output_dir
    );
    Ok(())
}

/// Save frames as GIF animation.
///
/// Creates a GIF animation from the video frames with configurable frame delay.
/// Uses parallel processing for frame quantization to improve performance.
///
/// # Arguments
/// * `frames` - VideoFrames containing the frames to save
/// * `output_dir` - Directory to save GIF to
/// * `filename` - Name of the GIF file (default: "video.gif")
///
/// # Returns
/// Ok(()) on success, or error if saving fails
///
/// # Requirements
/// - Create GIF encoder
/// - Add frames with delay
/// - Save to output directory
pub fn save_frames_as_gif(
    frames: &VideoFrames,
    output_dir: &str,
    filename: &str,
) -> Result<(), WanExampleError> {
    use gif::{Encoder, Repeat};
    use rayon::prelude::*;
    use std::fs::File;

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    let gif_path = format!("{}/{}", output_dir, filename);
    println!("\nCreating GIF animation...");

    let mut image_file = File::create(&gif_path)?;
    let mut encoder = Encoder::new(
        &mut image_file,
        frames.width as u16,
        frames.height as u16,
        &[],
    )
    .map_err(|e| WanExampleError::Io(std::io::Error::other(e)))?;

    encoder
        .set_repeat(Repeat::Infinite)
        .map_err(|e| WanExampleError::Io(std::io::Error::other(e)))?;

    // Parallel quantization for better performance
    let gif_frames: Vec<_> = frames
        .frames
        .par_iter()
        .map(|frame| {
            let mut gif_frame = gif::Frame::from_rgb_speed(
                frame.width as u16,
                frame.height as u16,
                &frame.data,
                30,
            );
            // ~25 FPS (delay is in centiseconds, so 4 = 40ms = 25fps)
            gif_frame.delay = 4;
            gif_frame
        })
        .collect();

    // Sequential write (GIF format requires sequential frame writing)
    for (i, gif_frame) in gif_frames.into_iter().enumerate() {
        encoder
            .write_frame(&gif_frame)
            .map_err(|e| WanExampleError::Io(std::io::Error::other(e)))?;

        // Print progress every 10 frames
        if i % 10 == 0 || i == frames.frames.len() - 1 {
            println!("  Encoded frame {}/{}", i + 1, frames.frames.len());
        }
    }

    println!("  ✓ Saved GIF to {}", gif_path);
    Ok(())
}

/// Save video output based on CLI flags.
///
/// Handles the output saving logic based on --frames and --gif flags:
/// - If --frames is set: save only PNG frames
/// - If --gif is set or by default: save as GIF animation
///
/// # Arguments
/// * `video` - Decoded video tensor [B, C, F, H, W]
/// * `output_dir` - Output directory path
/// * `save_frames` - Whether to save individual PNG frames
/// * `save_gif` - Whether to save as GIF (default behavior if neither flag set)
///
/// # Returns
/// Ok(()) on success, or error if saving fails
pub fn save_video_output(
    video: &Tensor,
    output_dir: &str,
    save_frames: bool,
    _save_gif: bool,
) -> Result<(), WanExampleError> {
    // Convert video tensor to frames
    let frames = video_tensor_to_frames(video)?;
    println!(
        "  ✓ Converted {} frames ({}x{})",
        frames.frames.len(),
        frames.width,
        frames.height
    );

    // Exclusive mode: --frames saves ONLY frames
    if save_frames {
        save_frames_as_png(&frames, output_dir)?;
        return Ok(());
    }

    // Default or --gif: Save GIF
    save_frames_as_gif(&frames, output_dir, "video.gif")?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Wan2.1+ Text-to-Video Generation");
    println!("================================");
    println!("Prompt: {}", args.prompt);
    println!(
        "Size: {}x{} [{} frames]",
        args.width, args.height, args.num_frames
    );
    println!(
        "Steps: {}, Guidance: {:.2}",
        args.steps, args.guidance_scale
    );

    // Validate dimensions before any model loading
    validate_dimensions(args.height, args.width)?;

    // Set up device
    let device = if args.cpu {
        println!("Device: CPU (--cpu flag set)");
        Device::Cpu
    } else {
        match Device::new_cuda(0) {
            Ok(d) => {
                println!("Device: CUDA");
                d
            }
            Err(_) => {
                println!("Device: CPU (CUDA not available, falling back)");
                Device::Cpu
            }
        }
    };

    let dtype = DType::BF16;

    // Resolve model paths
    let model_paths = if let Some(ref local_path) = args.local_weights {
        println!("\nResolving model paths from: {}", local_path);
        match WanModelPaths::resolve(local_path) {
            Ok(paths) => {
                println!("  Transformer: {:?}", paths.transformer);
                println!("  VAE dir: {:?}", paths.vae_dir);
                println!("  Text encoder: {:?}", paths.text_encoder);
                println!("  Tokenizer: {:?}", paths.tokenizer);
                Some(paths)
            }
            Err(e) => {
                eprintln!("Error resolving model paths: {}", e);
                return Err(e.into());
            }
        }
    } else {
        println!("\nNo --local-weights provided. Please specify model directory.");
        println!("Example: --local-weights ./models/Wan2.1-T2V-1.3B");
        return Ok(());
    };

    // Load models if paths are resolved
    if let Some(paths) = model_paths {
        println!("\nLoading models...");
        let loader = WanModelLoader::new(device.clone(), dtype);

        // Load transformer first (largest model, ~2.6GB for 1.3B variant)
        let transformer = loader.load_transformer(&paths)?;
        println!("  ✓ Transformer loaded");

        // Note: VAE will be loaded AFTER denoising to save VRAM
        // This follows the official Wan2.1 offload_model pattern

        // Load text encoder (runs on CPU, so no GPU memory impact)
        let mut text_encoder = loader.load_text_encoder(&paths)?;
        println!("  ✓ Text encoder loaded (CPU)");

        // Load tokenizer and create adapter
        // Note: TokenizerAdapter uses CPU device because the quantized text encoder runs on CPU
        let tokenizer = loader.load_tokenizer(&paths)?;
        let tokenizer_adapter = TokenizerAdapter::new(tokenizer, Device::Cpu, 512);
        println!(
            "  ✓ Tokenizer loaded (max_length={})",
            tokenizer_adapter.max_length()
        );

        println!("\nAll models loaded successfully!");

        // Encode prompts
        println!("\nEncoding prompts...");
        let encoded = encode_prompts(
            &mut text_encoder,
            &tokenizer_adapter,
            &args.prompt,
            &args.negative_prompt,
            args.guidance_scale,
            transformer.dtype(),
            &device,
        )?;

        println!("  ✓ Prompt encoded: {:?}", encoded.prompt_embeds.dims());
        if encoded.negative_prompt_embeds.is_some() {
            println!(
                "  ✓ Negative prompt encoded (CFG enabled, scale={:.2})",
                args.guidance_scale
            );
        } else {
            println!(
                "  ✓ CFG disabled (guidance_scale={:.2} <= 1.0)",
                args.guidance_scale
            );
        }

        // Free text encoder - we're done with it
        // (It's on CPU so this doesn't affect GPU memory, but good practice)
        drop(text_encoder);
        drop(tokenizer_adapter);

        // Prepare latents
        println!("\nPreparing latents...");
        let seed = get_or_generate_seed(args.seed);
        let latents = prepare_latents(
            args.height,
            args.width,
            args.num_frames,
            &device,
            Some(seed),
        )?;
        let (latent_frames, latent_height, latent_width) =
            calculate_latent_dims(args.height, args.width, args.num_frames);
        println!("  ✓ Latents prepared: {:?}", latents.dims());
        println!(
            "    Shape: [1, {}, {}, {}, {}]",
            WAN_LATENT_CHANNELS, latent_frames, latent_height, latent_width
        );
        println!("  ✓ Seed: {}", seed);

        // Set up scheduler
        println!("\nSetting up scheduler...");
        let mut scheduler = create_scheduler(args.flow_shift, args.steps, &device)?;
        let timesteps = get_timesteps(&scheduler);
        println!("  ✓ Scheduler created (flow_shift={:.1})", args.flow_shift);
        println!("  ✓ Timesteps: {} steps", timesteps.len());

        // Run denoising loop
        let denoising_result = run_denoising_loop(
            &transformer,
            &mut scheduler,
            latents,
            &encoded,
            args.guidance_scale,
            &device,
        )?;
        println!(
            "  ✓ Final latents shape: {:?}",
            denoising_result.latents.dims()
        );

        // Free transformer memory before VAE decode
        // This is critical for fitting both transformer and VAE in VRAM
        // Following official Wan2.1 offload_model pattern
        println!("\nFreeing transformer memory before VAE decode...");
        drop(transformer);
        drop(encoded);
        drop(scheduler);

        // Force CUDA to release memory
        if let Device::Cuda(_) = &device {
            // Synchronize to ensure all operations are complete
            device.synchronize()?;
        }

        // Now load VAE (after transformer is freed)
        // This is the key memory optimization from official Wan2.1
        println!("\nLoading VAE for decoding...");
        let vae = loader.load_vae(&paths)?;
        println!("  ✓ VAE loaded");

        // Decode latents with VAE
        let decode_result = decode_latents(&vae, &denoising_result.latents, dtype)?;
        println!("  ✓ Video tensor shape: {:?}", decode_result.video.dims());

        // Save output
        save_video_output(
            &decode_result.video,
            &args.output_dir,
            args.frames,
            args.gif,
        )?;

        // Print final summary
        println!("\n================================");
        println!("Generation complete!");
        println!("  Seed: {}", seed);
        println!("  Output: {}/", args.output_dir);
        if args.frames {
            println!("  Format: PNG frames");
        } else {
            println!("  Format: GIF animation (video.gif)");
        }
    } else {
        // This branch should not be reached due to early return above
        unreachable!("Model paths should be resolved at this point");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that default values are applied correctly when no arguments provided
    #[test]
    fn test_default_values() {
        let args = Args::parse_from(["wan"]);

        assert_eq!(args.prompt, "A cat walking on the grass");
        assert_eq!(args.negative_prompt, "");
        assert_eq!(args.width, 832);
        assert_eq!(args.height, 480);
        assert_eq!(args.num_frames, 81);
        assert_eq!(args.steps, 50);
        assert_eq!(args.guidance_scale, 5.0);
        assert!(args.local_weights.is_none());
        assert_eq!(args.output_dir, "output");
        assert!(args.seed.is_none());
        assert!(!args.cpu);
        assert!(!args.frames);
        assert!(!args.gif);
        assert_eq!(args.flow_shift, 5.0);
    }

    /// Test parsing custom prompt
    #[test]
    fn test_custom_prompt() {
        let args = Args::parse_from(["wan", "--prompt", "A dog running in the park"]);
        assert_eq!(args.prompt, "A dog running in the park");
    }

    /// Test parsing negative prompt
    #[test]
    fn test_negative_prompt() {
        let args = Args::parse_from(["wan", "--negative-prompt", "blurry, low quality"]);
        assert_eq!(args.negative_prompt, "blurry, low quality");
    }

    /// Test parsing video dimensions
    #[test]
    fn test_dimensions() {
        let args = Args::parse_from(["wan", "--width", "1280", "--height", "720"]);
        assert_eq!(args.width, 1280);
        assert_eq!(args.height, 720);
    }

    /// Test parsing num_frames
    #[test]
    fn test_num_frames() {
        let args = Args::parse_from(["wan", "--num-frames", "121"]);
        assert_eq!(args.num_frames, 121);
    }

    /// Test parsing inference steps
    #[test]
    fn test_steps() {
        let args = Args::parse_from(["wan", "--steps", "30"]);
        assert_eq!(args.steps, 30);
    }

    /// Test parsing guidance scale
    #[test]
    fn test_guidance_scale() {
        let args = Args::parse_from(["wan", "--guidance-scale", "7.5"]);
        assert_eq!(args.guidance_scale, 7.5);
    }

    /// Test parsing local weights path
    #[test]
    fn test_local_weights() {
        let args = Args::parse_from(["wan", "--local-weights", "./models/Wan2.1-T2V-1.3B"]);
        assert_eq!(
            args.local_weights,
            Some("./models/Wan2.1-T2V-1.3B".to_string())
        );
    }

    /// Test parsing output directory
    #[test]
    fn test_output_dir() {
        let args = Args::parse_from(["wan", "--output-dir", "my_output"]);
        assert_eq!(args.output_dir, "my_output");
    }

    /// Test parsing seed
    #[test]
    fn test_seed() {
        let args = Args::parse_from(["wan", "--seed", "42"]);
        assert_eq!(args.seed, Some(42));
    }

    /// Test parsing CPU flag
    #[test]
    fn test_cpu_flag() {
        let args = Args::parse_from(["wan", "--cpu"]);
        assert!(args.cpu);
    }

    /// Test parsing frames flag
    #[test]
    fn test_frames_flag() {
        let args = Args::parse_from(["wan", "--frames"]);
        assert!(args.frames);
    }

    /// Test parsing gif flag
    #[test]
    fn test_gif_flag() {
        let args = Args::parse_from(["wan", "--gif"]);
        assert!(args.gif);
    }

    /// Test parsing flow shift
    #[test]
    fn test_flow_shift() {
        let args = Args::parse_from(["wan", "--flow-shift", "3.0"]);
        assert_eq!(args.flow_shift, 3.0);
    }

    /// Test parsing multiple arguments together
    #[test]
    fn test_multiple_args() {
        let args = Args::parse_from([
            "wan",
            "--prompt",
            "A beautiful sunset",
            "--width",
            "1280",
            "--height",
            "720",
            "--num-frames",
            "49",
            "--steps",
            "25",
            "--guidance-scale",
            "4.0",
            "--seed",
            "12345",
            "--cpu",
            "--frames",
            "--flow-shift",
            "3.0",
        ]);

        assert_eq!(args.prompt, "A beautiful sunset");
        assert_eq!(args.width, 1280);
        assert_eq!(args.height, 720);
        assert_eq!(args.num_frames, 49);
        assert_eq!(args.steps, 25);
        assert_eq!(args.guidance_scale, 4.0);
        assert_eq!(args.seed, Some(12345));
        assert!(args.cpu);
        assert!(args.frames);
        assert_eq!(args.flow_shift, 3.0);
    }

    // =========================================================================
    // Latent Initialization Tests
    // =========================================================================

    /// Test latent dimension calculation for default 480p settings
    #[test]
    fn test_calculate_latent_dims_480p() {
        // Default: 480x832, 81 frames
        let (latent_frames, latent_height, latent_width) = calculate_latent_dims(480, 832, 81);

        // F' = (81 - 1) / 4 + 1 = 80 / 4 + 1 = 20 + 1 = 21
        assert_eq!(latent_frames, 21);
        // H' = 480 / 8 = 60
        assert_eq!(latent_height, 60);
        // W' = 832 / 8 = 104
        assert_eq!(latent_width, 104);
    }

    /// Test latent dimension calculation for 720p settings
    #[test]
    fn test_calculate_latent_dims_720p() {
        // 720p: 720x1280, 81 frames
        let (latent_frames, latent_height, latent_width) = calculate_latent_dims(720, 1280, 81);

        // F' = (81 - 1) / 4 + 1 = 21
        assert_eq!(latent_frames, 21);
        // H' = 720 / 8 = 90
        assert_eq!(latent_height, 90);
        // W' = 1280 / 8 = 160
        assert_eq!(latent_width, 160);
    }

    /// Test latent dimension calculation with minimum frames
    #[test]
    fn test_calculate_latent_dims_min_frames() {
        // Minimum: 1 frame
        let (latent_frames, _, _) = calculate_latent_dims(480, 832, 1);
        // F' = (1 - 1) / 4 + 1 = 0 + 1 = 1
        assert_eq!(latent_frames, 1);
    }

    /// Test latent dimension calculation with various frame counts
    #[test]
    fn test_calculate_latent_dims_various_frames() {
        // Test formula: F' = (num_frames - 1) / 4 + 1
        let test_cases = [
            (1, 1),   // (1-1)/4+1 = 1
            (5, 2),   // (5-1)/4+1 = 2
            (9, 3),   // (9-1)/4+1 = 3
            (17, 5),  // (17-1)/4+1 = 5
            (33, 9),  // (33-1)/4+1 = 9
            (49, 13), // (49-1)/4+1 = 13
            (81, 21), // (81-1)/4+1 = 21
            (97, 25), // (97-1)/4+1 = 25
        ];

        for (num_frames, expected_latent_frames) in test_cases {
            let (latent_frames, _, _) = calculate_latent_dims(480, 832, num_frames);
            assert_eq!(
                latent_frames, expected_latent_frames,
                "Failed for num_frames={}: expected {}, got {}",
                num_frames, expected_latent_frames, latent_frames
            );
        }
    }

    /// Test prepare_latents creates tensor with correct shape
    #[test]
    fn test_prepare_latents_shape() {
        let device = Device::Cpu;
        let latents = prepare_latents(480, 832, 81, &device, Some(42)).unwrap();

        let dims = latents.dims();
        assert_eq!(dims.len(), 5, "Latents should be 5D tensor");
        assert_eq!(dims[0], 1, "Batch size should be 1");
        assert_eq!(dims[1], WAN_LATENT_CHANNELS, "Channels should be 16");
        assert_eq!(dims[2], 21, "Latent frames should be 21");
        assert_eq!(dims[3], 60, "Latent height should be 60");
        assert_eq!(dims[4], 104, "Latent width should be 104");
    }

    /// Test prepare_latents with deterministic seed produces same results
    #[test]
    fn test_prepare_latents_deterministic() {
        let device = Device::Cpu;
        let seed = 12345u64;

        let latents1 = prepare_latents(256, 256, 17, &device, Some(seed)).unwrap();
        let latents2 = prepare_latents(256, 256, 17, &device, Some(seed)).unwrap();

        // Same seed should produce identical latents
        let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2, "Same seed should produce identical latents");
    }

    /// Test prepare_latents with different seeds produces different results
    #[test]
    fn test_prepare_latents_different_seeds() {
        let device = Device::Cpu;

        let latents1 = prepare_latents(256, 256, 17, &device, Some(42)).unwrap();
        let latents2 = prepare_latents(256, 256, 17, &device, Some(43)).unwrap();

        // Different seeds should produce different latents
        let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(v1, v2, "Different seeds should produce different latents");
    }

    /// Test get_or_generate_seed returns provided seed
    #[test]
    fn test_get_or_generate_seed_provided() {
        let seed = get_or_generate_seed(Some(42));
        assert_eq!(seed, 42);
    }

    /// Test get_or_generate_seed generates seed when None
    #[test]
    fn test_get_or_generate_seed_generated() {
        let seed1 = get_or_generate_seed(None);
        // Sleep briefly to ensure different timestamp
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _seed2 = get_or_generate_seed(None);

        // Generated seeds should be non-zero (very unlikely to be 0)
        assert_ne!(seed1, 0);
        // Note: seeds might be same if called very quickly, but should generally differ
    }

    /// Test latent shape for small dimensions
    #[test]
    fn test_prepare_latents_small_dims() {
        let device = Device::Cpu;
        // Minimum valid dimensions: 16x16 (divisible by 16 for validation, but /8 for latents)
        let latents = prepare_latents(128, 128, 5, &device, Some(42)).unwrap();

        let dims = latents.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 16);
        assert_eq!(dims[2], 2); // (5-1)/4+1 = 2
        assert_eq!(dims[3], 16); // 128/8 = 16
        assert_eq!(dims[4], 16); // 128/8 = 16
    }

    // =========================================================================
    // Dimension Validation Tests
    // =========================================================================

    /// Test that valid dimensions (divisible by 16) pass validation
    #[test]
    fn test_validate_dimensions_valid() {
        // Common valid dimensions
        assert!(validate_dimensions(480, 832).is_ok());
        assert!(validate_dimensions(720, 1280).is_ok());
        assert!(validate_dimensions(256, 256).is_ok());
        assert!(validate_dimensions(16, 16).is_ok());
        assert!(validate_dimensions(1024, 1024).is_ok());
    }

    /// Test that invalid height (not divisible by 16) fails validation
    #[test]
    fn test_validate_dimensions_invalid_height() {
        let result = validate_dimensions(481, 832);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WanExampleError::InvalidDimensions(msg) => {
                assert!(msg.contains("height 481"));
                assert!(msg.contains("not divisible by 16"));
            }
            _ => panic!("Expected InvalidDimensions error"),
        }
    }

    /// Test that invalid width (not divisible by 16) fails validation
    #[test]
    fn test_validate_dimensions_invalid_width() {
        let result = validate_dimensions(480, 833);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WanExampleError::InvalidDimensions(msg) => {
                assert!(msg.contains("width 833"));
                assert!(msg.contains("not divisible by 16"));
            }
            _ => panic!("Expected InvalidDimensions error"),
        }
    }

    /// Test that both invalid height and width reports both errors
    #[test]
    fn test_validate_dimensions_both_invalid() {
        let result = validate_dimensions(481, 833);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WanExampleError::InvalidDimensions(msg) => {
                assert!(msg.contains("height 481"));
                assert!(msg.contains("width 833"));
            }
            _ => panic!("Expected InvalidDimensions error"),
        }
    }

    /// Test boundary cases around 16
    #[test]
    fn test_validate_dimensions_boundary() {
        // Just below 16
        assert!(validate_dimensions(15, 16).is_err());
        assert!(validate_dimensions(16, 15).is_err());

        // Exactly 16
        assert!(validate_dimensions(16, 16).is_ok());

        // Just above 16 but not divisible
        assert!(validate_dimensions(17, 16).is_err());
        assert!(validate_dimensions(16, 17).is_err());

        // Next valid value (32)
        assert!(validate_dimensions(32, 32).is_ok());
    }

    /// Test that error message includes nearest valid value
    #[test]
    fn test_validate_dimensions_suggests_nearest() {
        let result = validate_dimensions(481, 832);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WanExampleError::InvalidDimensions(msg) => {
                // 481 / 16 * 16 = 480
                assert!(msg.contains("nearest valid: 480"));
            }
            _ => panic!("Expected InvalidDimensions error"),
        }
    }
}
