use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::{
    AutoencoderKLLtxVideo, LtxVideoTransformer3DModel,
    loader::WeightLoader,
    quantized_t5_encoder::QuantizedT5EncoderModel,
    scheduler::FlowMatchEulerDiscreteScheduler,
    t2v_pipeline::{LtxPipeline, LtxVideoProcessor, OutputType, TextEncoder, Tokenizer},
    text_encoder::{T5EncoderConfig, T5TextEncoderWrapper},
};
use candle_video::utils::video_export::{
    PixelRange, save_gif_atomic_with_observer, save_png_frames_with_observer, tensor_to_rgb_frames,
};
use candle_video::{
    GenerationEvent, GenerationStage, ProgressObserver, RunManifest, ensure_not_cancelled,
    run_with_heartbeat, write_run_manifest,
};
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;

#[path = "../common/progress.rs"]
mod progress;

use progress::{CliProgressObserver, CliProgressOptions, LogFormat, ProgressMode};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "A video of a cute cat playing with a yarn ball")]
    prompt: String,

    #[arg(long, default_value = "")]
    negative_prompt: String,

    #[arg(long)]
    steps: Option<usize>,

    #[arg(long)]
    guidance_scale: Option<f32>,

    #[arg(long, default_value_t = 512)]
    height: usize,

    #[arg(long, default_value_t = 768)]
    width: usize,

    #[arg(long, default_value_t = 97)]
    num_frames: usize,

    #[arg(long, default_value = "0.9.8-2b-distilled")]
    ltxv_version: String,

    #[arg(long, default_value = "oxide-lab/LTX-Video-0.9.8-2B-distilled")]
    model_id: String,

    #[arg(long)]
    local_weights: Option<String>,

    #[arg(long, default_value = "output")]
    output_dir: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    seed: Option<u64>,

    /// Save individual PNG frames
    #[arg(long)]
    frames: bool,

    /// Save as GIF animation
    #[arg(long)]
    gif: bool,

    /// Output frame rate used for GIF timing and the run manifest.
    #[arg(long, default_value_t = 25)]
    fps: usize,

    /// Use VAE tiling (spatial)
    #[arg(long, default_value_t = false)]
    vae_tiling: bool,

    /// Use VAE slicing (batch)
    #[arg(long)]
    vae_slicing: bool,

    /// Use BF16 T5 instead of GGUF quantized (requires more VRAM but better quality)
    #[arg(long)]
    use_bf16_t5: bool,

    /// Load pre-computed embeddings from a safetensors file (bypasses T5 entirely)
    #[arg(long)]
    embeddings_file: Option<String>,

    /// Load initial latents from a safetensors file (for reproducible comparison)
    #[arg(long)]
    initial_latents_file: Option<String>,

    /// Save final latents to a file for comparison
    #[arg(long)]
    save_final_latents: Option<String>,

    /// Load from unified safetensors file (official LTX-Video format)
    /// This is a single file containing DiT+VAE weights (e.g., ltx-video-2b-v0.9.5.safetensors)
    #[arg(long)]
    unified_weights: Option<String>,

    /// Override STG scale (Not supported in this port, kept for CLI compatibility if needed but ignored)
    #[arg(long)]
    stg_scale: Option<f32>,

    /// Override rescaling scale (default from preset)
    #[arg(long)]
    rescaling_scale: Option<f32>,

    /// Override stochastic sampling (default from preset)
    #[arg(long)]
    stochastic_sampling: Option<bool>,

    /// Progress renderer: auto, always, or never.
    #[arg(long, value_enum, default_value_t = ProgressMode::Auto)]
    progress: ProgressMode,

    /// Human terminal events or JSONL events on stderr.
    #[arg(long, value_enum, default_value_t = LogFormat::Human)]
    log_format: LogFormat,

    #[arg(short = 'v', long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[arg(short = 'q', long)]
    quiet: bool,

    #[arg(long, default_value_t = 5)]
    heartbeat_seconds: u64,

    #[arg(long)]
    timings: bool,

    #[arg(long, visible_alias = "trace-stages")]
    diagnostics: bool,

    #[arg(long)]
    gpu_stats: bool,

    #[arg(long, visible_alias = "log-file")]
    events_jsonl: Option<PathBuf>,

    /// Write a run manifest next to the generated GIF or frame directory.
    #[arg(long)]
    dump_run_manifest: bool,

    /// Retain an incomplete `.partial` GIF when export fails.
    #[arg(long)]
    keep_partial: bool,
}

struct TokenizerAdapter {
    tokenizer: HfTokenizer,
    device: Device,
}

impl Tokenizer for TokenizerAdapter {
    fn encode_batch(
        &self,
        prompts: &[String],
        max_length: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let encodings = self
            .tokenizer
            .encode_batch(prompts.to_vec(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut ids_vec = Vec::new();
        let mut mask_vec = Vec::new();
        for e in encodings {
            let mut ids = e.get_ids().to_vec();
            let mut mask = e.get_attention_mask().to_vec();
            if ids.len() > max_length {
                ids.truncate(max_length);
                mask.truncate(max_length);
            } else {
                ids.resize(max_length, 0);
                mask.resize(max_length, 0);
            }
            ids_vec.push(ids);
            mask_vec.push(mask);
        }
        let b = ids_vec.len();
        let flat_ids: Vec<u32> = ids_vec.into_iter().flatten().collect();
        let flat_mask: Vec<u32> = mask_vec.into_iter().flatten().collect();
        let ids_t = Tensor::new(flat_ids, &self.device)?.reshape((b, max_length))?;
        let mask_t = Tensor::new(flat_mask, &self.device)?.reshape((b, max_length))?;
        Ok((ids_t, mask_t))
    }
    fn model_max_length(&self) -> usize {
        128
    }
}

struct DummyTextEncoder;
impl TextEncoder for DummyTextEncoder {
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn forward(&mut self, _: &Tensor) -> candle_core::Result<Tensor> {
        candle_core::bail!(
            "DummyTextEncoder: forward should not be called when embeddings are provided"
        )
    }
}

struct DummyTokenizer;
impl Tokenizer for DummyTokenizer {
    fn encode_batch(&self, _: &[String], _: usize) -> candle_core::Result<(Tensor, Tensor)> {
        candle_core::bail!(
            "DummyTokenizer: encode_batch should not be called when embeddings are provided"
        )
    }
    fn model_max_length(&self) -> usize {
        128
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.fps == 0 {
        anyhow::bail!("--fps must be greater than zero");
    }
    if args.frames && args.gif {
        anyhow::bail!("--frames and --gif are mutually exclusive");
    }
    let renderer = CliProgressObserver::new(CliProgressOptions {
        progress: args.progress,
        log_format: args.log_format,
        quiet: args.quiet,
        verbose: args.verbose,
        timings: args.timings,
        diagnostics: args.diagnostics,
        gpu_stats: args.gpu_stats,
        heartbeat_seconds: args.heartbeat_seconds,
        events_jsonl: args.events_jsonl.clone(),
    })?;
    CliProgressObserver::install_ctrlc(&renderer)?;
    let observer: Arc<dyn ProgressObserver> = renderer.clone();
    let ltxv_config =
        candle_video::models::ltx_video::configs::get_config_by_version(&args.ltxv_version);
    let num_inference_steps = args
        .steps
        .unwrap_or(ltxv_config.inference.num_inference_steps);
    let sigmas_from_config = ltxv_config.inference.timesteps.clone();
    let effective_steps = sigmas_from_config
        .as_ref()
        .map_or(num_inference_steps, Vec::len);
    let guidance_scale = args
        .guidance_scale
        .unwrap_or(ltxv_config.inference.guidance_scale);
    let rescaling_scale = args
        .rescaling_scale
        .unwrap_or(ltxv_config.inference.rescaling_scale);
    let stochastic_sampling = args
        .stochastic_sampling
        .unwrap_or(ltxv_config.inference.stochastic_sampling);
    let stg_scale = args.stg_scale.unwrap_or(ltxv_config.inference.stg_scale);

    eprintln!("LTX-Video Text-to-Video Generation");
    eprintln!("==================================");
    eprintln!("Prompt: {}", args.prompt);
    eprintln!(
        "Version: {} (requested_steps={}, schedule_steps={}, guidance={:.2}, rescale={:.2}, stg={:.2}, stochastic={})",
        args.ltxv_version,
        num_inference_steps,
        effective_steps,
        guidance_scale,
        rescaling_scale,
        stg_scale,
        stochastic_sampling
    );
    eprintln!(
        "Size: {}x{} [{} frames]",
        args.width, args.height, args.num_frames
    );

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    // Generate random seed if not provided
    let seed = args.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        now.as_secs() ^ (now.subsec_nanos() as u64)
    });
    // Candle's CPU backend does not expose a global RNG seed; latents use the
    // explicit Pcg32 stream below. CUDA/Metal still receive the seed for
    // kernels that consume the device RNG.
    if !device.is_cpu() {
        device.set_seed(seed)?;
    }
    eprintln!("Device: {:?}", device);
    eprintln!("Seed: {}", seed);

    let dtype = DType::BF16;

    let load_started = Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::LoadComponents,
        message: "Locating and loading LTX tokenizer, text encoder, transformer, and VAE"
            .to_string(),
    });

    // 1. Locate weights
    let mut detected_unified_weights = None;
    let (transformer_file, vae_file, t5_file, tokenizer_file) =
        if let Some(local_path) = &args.local_weights {
            let base = PathBuf::from(local_path);

            let unified_candidate = [
                base.join("ltxv-2b-0.9.8-distilled.safetensors"),
                base.join("ltxv-2b-0.9.8.safetensors"),
            ]
            .into_iter()
            .find(|path| path.exists());
            let use_unified = args.unified_weights.is_some() || unified_candidate.is_some();
            if args.unified_weights.is_none() {
                detected_unified_weights = unified_candidate
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned());
            }

            let transformer = if use_unified {
                // When using unified weights, transformer is embedded in the single file
                unified_candidate
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("dummy")) // explicit --unified-weights
            } else if base
                .join("transformer/diffusion_pytorch_model.safetensors")
                .exists()
            {
                base.join("transformer/diffusion_pytorch_model.safetensors")
            } else if base.join("ltx_video_transformer_3d.safetensors").exists() {
                base.join("ltx_video_transformer_3d.safetensors")
            } else {
                anyhow::bail!("Transformer weights not found in {:?}", base);
            };

            let vae = if use_unified {
                // When using unified weights, VAE is embedded in the single file
                unified_candidate
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("dummy")) // explicit --unified-weights
            } else if base
                .join("vae/diffusion_pytorch_model.safetensors")
                .exists()
            {
                base.join("vae/diffusion_pytorch_model.safetensors")
            } else if base.join("vae.safetensors").exists() {
                base.join("vae.safetensors")
            } else {
                anyhow::bail!("VAE weights not found in {:?}", base);
            };

            // T5 file: prefer BF16 if --use-bf16-t5, otherwise GGUF
            let t5 = if args.use_bf16_t5 {
                // BF16 model.safetensors
                let bf16_path = base.join("text_encoder").join("model.safetensors");
                if bf16_path.exists() {
                    bf16_path
                } else {
                    anyhow::bail!(
                        "BF16 T5 not found at {:?}. Use GGUF or download BF16 weights.",
                        bf16_path
                    );
                }
            } else {
                // GGUF quantized
                if base.join("t5-v1_1-xxl-encoder-Q8_0.gguf").exists() {
                    base.join("t5-v1_1-xxl-encoder-Q8_0.gguf")
                } else if base
                    .join("text_encoder_gguf")
                    .join("t5-v1_1-xxl-encoder-Q8_0.gguf")
                    .exists()
                {
                    base.join("text_encoder_gguf")
                        .join("t5-v1_1-xxl-encoder-Q8_0.gguf")
                } else if base.join("t5-xxl.gguf").exists() {
                    base.join("t5-xxl.gguf")
                } else if base
                    .join("text_encoder_gguf")
                    .join("t5-v1_1-xxl-encoder-Q5_K_M.gguf")
                    .exists()
                {
                    base.join("text_encoder_gguf")
                        .join("t5-v1_1-xxl-encoder-Q5_K_M.gguf")
                } else {
                    anyhow::bail!(
                        "T5 GGUF not found in {:?}. Try --use-bf16-t5 if you have BF16 weights.",
                        base
                    );
                }
            };
            let tokenizer = {
                let path1 = base.join("tokenizer").join("tokenizer.json");
                let path2 = base.join("text_encoder").join("tokenizer.json");
                let path3 = base.join("text_encoder_8bit").join("tokenizer.json");
                let path4 = base.join("text_encoder_gguf").join("tokenizer.json");
                if path1.exists() {
                    path1
                } else if path2.exists() {
                    path2
                } else if path3.exists() {
                    path3
                } else if path4.exists() {
                    path4
                } else {
                    eprintln!("    Tokenizer not found locally, downloading from HF...");
                    let _ = std::io::stdout().flush();
                    // Standard T5 tokenizer from google repository is usually safer
                    Api::new()?
                        .repo(Repo::new(
                            "google-t5/t5-v1_1-xxl".to_string(),
                            RepoType::Model,
                        ))
                        .get("tokenizer.json")?
                }
            };

            (transformer, vae, t5, tokenizer)
        } else {
            eprintln!("\nDownloading models from HuggingFace: {}", args.model_id);
            let api = Api::new()?;
            let repo = api.repo(Repo::with_revision(
                args.model_id.clone(),
                RepoType::Model,
                "main".into(),
            ));

            // Determine if we should use unified weights based on version or repository
            let is_098 = args.ltxv_version.contains("0.9.8");

            let (transformer, vae) = if is_098 {
                eprintln!("  Fetching unified weight file (official format)...");
                let unified = repo.get("ltxv-2b-0.9.8-distilled.safetensors")?;
                // Return same path for both, the loading logic will handle it if unified_weights is set
                // Actually, we'll force unified_weights logic later if we are in this branch
                (unified.clone(), unified)
            } else {
                let transformer = repo.get("transformer/diffusion_pytorch_model.safetensors")?;
                let _ = repo.get("transformer/config.json")?;
                let vae = repo.get("vae/diffusion_pytorch_model.safetensors")?;
                let _ = repo.get("vae/config.json")?;
                (transformer, vae)
            };

            let t5 = repo.get("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf")?;
            let tokenizer = repo.get("text_encoder_gguf/tokenizer.json")?;

            (transformer, vae, t5, tokenizer)
        };

    // If we've downloaded a 0.9.8 model and don't have local paths or explicit unified weights,
    // we should treat the transformer/vae paths (which are the same) as unified weights.
    // We'll update args.unified_weights to reflect this.
    // NOTE: This is a hack because args is immutable here, but we'll adapt the loading logic.
    let effective_unified_weights = args
        .unified_weights
        .clone()
        .or(detected_unified_weights)
        .or_else(|| {
            if args.ltxv_version.contains("0.9.8") && args.local_weights.is_none() {
                Some(transformer_file.to_string_lossy().to_string())
            } else {
                None
            }
        });

    if let Some(local_weights) = args.local_weights.as_ref() {
        eprintln!("\nLoading models from local path: {}", local_weights);
    }

    // 2. Step 1: Encode prompts with T5 (or load from file)
    let prompt_started = Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::EncodePrompt,
        message: "Encoding positive and negative prompts".to_string(),
    });
    let (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = if let Some(ref emb_file) = args.embeddings_file {
        // Load pre-computed embeddings from file
        eprintln!("  Loading embeddings from {}...", emb_file);
        let tensors = candle_core::safetensors::load(emb_file, &device)?;

        let p_emb = tensors
            .get("prompt_embeds")
            .ok_or_else(|| anyhow::anyhow!("prompt_embeds not found in file"))?
            .clone();
        let p_mask = tensors
            .get("prompt_attention_mask")
            .ok_or_else(|| anyhow::anyhow!("prompt_attention_mask not found in file"))?
            .clone();
        let n_emb = tensors
            .get("negative_prompt_embeds")
            .ok_or_else(|| anyhow::anyhow!("negative_prompt_embeds not found in file"))?
            .clone();
        let n_mask = tensors
            .get("negative_prompt_attention_mask")
            .ok_or_else(|| anyhow::anyhow!("negative_prompt_attention_mask not found in file"))?
            .clone();

        eprintln!(
            "  Loaded: prompt_embeds {:?}, mask {:?}",
            p_emb.dims(),
            p_mask.dims()
        );
        (p_emb, p_mask, n_emb, n_mask)
    } else {
        // Compute embeddings with T5
        let tokenizer_hf =
            HfTokenizer::from_file(&tokenizer_file).map_err(|e| anyhow::anyhow!(e))?;
        let tokenizer_adapter = TokenizerAdapter {
            tokenizer: tokenizer_hf,
            device: device.clone(),
        };

        let (p_ids, p_mask) =
            tokenizer_adapter.encode_batch(std::slice::from_ref(&args.prompt), 128)?;
        let (n_ids, n_mask) =
            tokenizer_adapter.encode_batch(std::slice::from_ref(&args.negative_prompt), 128)?;

        // Load T5 model (BF16 or GGUF)
        let (p_emb, n_emb) = if args.use_bf16_t5 {
            eprintln!("  Loading BF16 T5 model...");
            let config = T5EncoderConfig::t5_xxl();
            let mut t5_wrapper = T5TextEncoderWrapper::new(config, device.clone(), DType::BF16)?;

            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[&t5_file], DType::BF16, &device)? };
            t5_wrapper.load_model(vb)?;

            let p_emb = t5_wrapper.forward(&p_ids)?;
            let n_emb = t5_wrapper.forward(&n_ids)?;
            (p_emb, n_emb)
        } else {
            eprintln!("  Loading GGUF T5 model...");
            let t5_model = QuantizedT5EncoderModel::load(&t5_file, &device)?;
            let p_emb = t5_model.forward(&p_ids, Some(&p_mask))?;
            let n_emb = t5_model.forward(&n_ids, Some(&n_mask))?;
            (p_emb, n_emb)
        };

        (
            p_emb,
            p_mask.to_dtype(DType::F32)?,
            n_emb,
            n_mask.to_dtype(DType::F32)?,
        )
    };
    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::EncodePrompt,
        elapsed_secs: prompt_started.elapsed().as_secs_f64(),
    });

    // 3. Step 2: Load Transformer and VAE for generation
    // 3. Step 2: Load Transformer and VAE for generation
    eprintln!("Loading models...");

    // Check if using unified weights (official LTX-Video format)
    let (vae, transformer) = if let Some(ref unified_path) = effective_unified_weights {
        use candle_video::models::ltx_video::weight_format::KeyRemapper;

        eprintln!("  Loading from unified weights: {}", unified_path);
        let unified_file = PathBuf::from(unified_path);
        if !unified_file.exists() {
            anyhow::bail!("Unified weights file not found: {:?}", unified_file);
        }

        // Load all tensors from unified file
        let all_tensors = candle_core::safetensors::load(&unified_file, &device)?;
        let remapper = KeyRemapper::new();

        // Separate and remap tensors for VAE and Transformer
        let mut vae_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        let mut trans_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (key, tensor) in all_tensors {
            let remapped_key = remapper.remap_key(&key);
            if KeyRemapper::is_vae_key(&key) {
                // Remove "vae." prefix if present for VarBuilder compatibility
                let clean_key = remapped_key
                    .strip_prefix("vae.")
                    .unwrap_or(&remapped_key)
                    .to_string();
                vae_tensors.insert(clean_key, tensor);
            } else if KeyRemapper::is_transformer_key(&key) {
                // Remove native "model.diffusion_model." or "transformer." prefix
                let clean_key = remapped_key
                    .strip_prefix("model.diffusion_model.")
                    .or_else(|| remapped_key.strip_prefix("transformer."))
                    .unwrap_or(&remapped_key)
                    .to_string();
                trans_tensors.insert(clean_key, tensor);
            }
        }

        eprintln!(
            "    Found {} VAE tensors, {} Transformer tensors",
            vae_tensors.len(),
            trans_tensors.len()
        );

        // Create VarBuilders from tensors
        let vae_vb = VarBuilder::from_tensors(vae_tensors, dtype, &device);
        let trans_vb = VarBuilder::from_tensors(trans_tensors, dtype, &device);

        // Build VAE
        let mut vae_config = ltxv_config.vae.clone();
        vae_config.timestep_conditioning = true;
        let mut vae = AutoencoderKLLtxVideo::new(vae_config, vae_vb)?;
        vae.use_tiling = args.vae_tiling;
        vae.use_slicing = args.vae_slicing;
        vae.use_framewise_decoding = args.vae_tiling && args.num_frames > 16;

        // Build Transformer
        let trans_config = ltxv_config.transformer.clone();
        let transformer = LtxVideoTransformer3DModel::new(&trans_config, trans_vb)?;

        (vae, transformer)
    } else {
        // Original Diffusers format loading
        let vae_load = WeightLoader::new(device.clone(), dtype);
        let vae_vb = vae_load.load_single(&vae_file)?;
        let vae_dir = vae_file.parent().unwrap_or(Path::new("."));
        let mut vae_config = if vae_dir.join("config.json").exists() {
            let f = std::fs::File::open(vae_dir.join("config.json"))?;
            serde_json::from_reader(f)?
        } else {
            ltxv_config.vae.clone()
        };
        vae_config.timestep_conditioning = true;
        let mut vae = AutoencoderKLLtxVideo::new(vae_config, vae_vb)?;
        vae.use_tiling = args.vae_tiling;
        vae.use_slicing = args.vae_slicing;
        vae.use_framewise_decoding = args.vae_tiling && args.num_frames > 16;

        let trans_load = WeightLoader::new(device.clone(), dtype);
        let trans_vb = trans_load.load_single(&transformer_file)?;
        let trans_config = ltxv_config.transformer.clone();
        let transformer = LtxVideoTransformer3DModel::new(&trans_config, trans_vb)?;

        (vae, transformer)
    };

    // Scheduler
    let mut scheduler_config = ltxv_config.scheduler.clone();
    scheduler_config.stochastic_sampling = stochastic_sampling;
    let scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_config)?;

    // 4. Pipeline
    let mut pipeline = LtxPipeline::new(
        Box::new(scheduler),
        Box::new(vae),
        Box::new(DummyTextEncoder),
        Box::new(DummyTokenizer),
        Box::new(transformer),
        Box::new(LtxVideoProcessor::new(VaeConfig {
            scaling_factor: 1.0,
            timestep_conditioning: true,
        })),
    );
    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::LoadComponents,
        elapsed_secs: load_started.elapsed().as_secs_f64(),
    });

    // 4b. Prepare deterministic latents
    use candle_video::utils::deterministic_rng::Pcg32;
    let mut rng = Pcg32::new(seed, 1442695040888963407);

    // Config values for LTX Video
    let vae_spatial_compression = 32;
    let vae_temporal_compression = 8;
    let transformer_spatial_patch = 1;
    let transformer_temporal_patch = 1;

    let height = args.height;
    let width = args.width;
    let num_frames = args.num_frames;

    let latent_height = height / vae_spatial_compression;
    let latent_width = width / vae_spatial_compression;
    let latent_frames = (num_frames - 1) / vae_temporal_compression + 1;
    let channels = 128; // transformer in_channels

    let prepare_started = Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::PrepareLatents,
        message: "Preparing initial latent noise".to_string(),
    });

    // Generate or load initial latents
    let latents_packed = if let Some(ref latents_file) = args.initial_latents_file {
        eprintln!("  Loading initial latents from {}...", latents_file);
        let tensors = candle_core::safetensors::load(latents_file, &device)?;
        let latents = tensors
            .get("initial_latents")
            .ok_or_else(|| anyhow::anyhow!("initial_latents not found in file"))?
            .clone();
        eprintln!("  Loaded latents shape: {:?}", latents.dims());
        latents
    } else {
        // [B, C, F, H, W]
        let shape = (1, channels, latent_frames, latent_height, latent_width);
        let latents_unpacked = rng.randn(shape, &device)?;
        LtxPipeline::pack_latents(
            &latents_unpacked,
            transformer_spatial_patch,
            transformer_temporal_patch,
        )?
    };
    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::PrepareLatents,
        elapsed_secs: prepare_started.elapsed().as_secs_f64(),
    });
    observer.on_event(&GenerationEvent::ConfigResolved {
        requested_width: args.width,
        requested_height: args.height,
        adjusted_width: args.width,
        adjusted_height: args.height,
        requested_frames: args.num_frames,
        adjusted_frames: args.num_frames,
        latent_shape: latents_packed.dims().to_vec(),
        steps: effective_steps,
        guidance_scale,
        seed: Some(seed),
        scheduler: "ltx".to_string(),
    });

    // 5. Run Generation
    let generation_started = Instant::now();
    pipeline.guidance_rescale = rescaling_scale;

    let decode_timestep = ltxv_config
        .inference
        .decode_timestep
        .clone()
        .unwrap_or(vec![0.0]);
    let decode_noise_scale = ltxv_config.inference.decode_noise_scale.clone();

    let video_out = pipeline.call_with_observer(
        None, // prompt
        None, // negative_prompt
        args.height,
        args.width,
        args.num_frames,
        args.fps,
        effective_steps,
        None,               // timesteps
        sigmas_from_config, // sigmas_provided
        guidance_scale,
        rescaling_scale,
        stg_scale,
        1,
        Some(latents_packed), // Pass packed latents
        Some(prompt_embeds),
        Some(prompt_attention_mask),
        Some(negative_prompt_embeds),
        Some(negative_prompt_attention_mask),
        decode_timestep,
        decode_noise_scale,
        OutputType::Tensor,
        128,
        Some(ltxv_config.inference.skip_block_list.clone()),
        &device,
        Arc::clone(&observer),
    )?;

    let postprocess_started = Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::PostProcess,
        message: "Converting decoded tensor to RGB frames".to_string(),
    });
    let heartbeat_observer = Arc::clone(&observer);
    let (frame_data, w, h) = run_with_heartbeat(
        &heartbeat_observer,
        GenerationStage::PostProcess,
        "Converting decoded frames still running",
        || tensor_to_rgb_frames(&video_out.frames, PixelRange::ZeroTo255),
    )?;
    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::PostProcess,
        elapsed_secs: postprocess_started.elapsed().as_secs_f64(),
    });

    // 5. Save output
    let output_dir = PathBuf::from(&args.output_dir);
    std::fs::create_dir_all(&output_dir)?;
    let encode_started = Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::EncodeVideo,
        message: format!("Writing {} decoded frames", frame_data.len()),
    });

    // Exclusive mode: --frames saves ONLY frames
    if args.frames {
        save_png_frames_with_observer(&frame_data, w, h, &output_dir, Some(observer.as_ref()))?;
        eprintln!(
            "\nDone! Saved {} frames to {}",
            frame_data.len(),
            output_dir.display()
        );
    } else {
        eprintln!("Creating GIF animation with atomic publication...");
        let gif_path = output_dir.join("video.gif");
        let frame_delay = u16::try_from((100usize + args.fps / 2) / args.fps)
            .unwrap_or(u16::MAX)
            .max(1);
        save_gif_atomic_with_observer(
            &frame_data,
            w,
            h,
            &gif_path,
            frame_delay,
            args.keep_partial,
            Some(observer.as_ref()),
        )?;
        println!("\nDone! Saved GIF to {}", gif_path.display());
    }

    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::EncodeVideo,
        elapsed_secs: encode_started.elapsed().as_secs_f64(),
    });

    let generated_output = if args.frames {
        output_dir.display().to_string()
    } else {
        output_dir.join("video.gif").display().to_string()
    };

    if args.dump_run_manifest {
        let manifest_path = output_dir.join("run.json");
        let manifest = RunManifest {
            model: format!("LTX-Video-{}", args.ltxv_version),
            backend: if args.cpu { "cpu" } else { "cuda" }.to_string(),
            device: format!("{device:?}"),
            dtype: format!("{dtype:?}"),
            width: w,
            height: h,
            frames: frame_data.len(),
            fps: args.fps,
            steps: effective_steps,
            guidance_scale,
            seed: Some(seed),
            scheduler: "ltx-config".to_string(),
            total_seconds: generation_started.elapsed().as_secs_f64(),
            peak_vram_bytes: renderer.peak_vram_bytes(),
            output: generated_output.clone(),
        };
        let finalize_started = Instant::now();
        observer.on_event(&GenerationEvent::StageStarted {
            stage: GenerationStage::Finalize,
            message: format!("Writing run manifest to {}", manifest_path.display()),
        });
        write_run_manifest(&manifest_path, &manifest)?;
        observer.on_event(&GenerationEvent::StageFinished {
            stage: GenerationStage::Finalize,
            elapsed_secs: finalize_started.elapsed().as_secs_f64(),
        });
        println!("Saved run manifest to {}", manifest_path.display());
    }

    ensure_not_cancelled(observer.as_ref())?;
    observer.on_event(&GenerationEvent::GenerationFinished {
        output: Some(generated_output),
        elapsed_secs: generation_started.elapsed().as_secs_f64(),
    });

    Ok(())
}

use candle_video::models::ltx_video::t2v_pipeline::VaeConfig;
