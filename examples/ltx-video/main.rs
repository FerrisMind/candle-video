use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::{
    AutoencoderKLLtxVideo, LtxVideoTransformer3DModel,
    loader::WeightLoader,
    quantized_t5_encoder::QuantizedT5EncoderModel,
    scheduler::FlowMatchEulerDiscreteScheduler,
    t2v_pipeline::{LtxPipeline, LtxVideoProcessor, OutputType, TextEncoder, Tokenizer},
    text_encoder::{T5EncoderConfig, T5TextEncoderWrapper},
};
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer as HfTokenizer;

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

    #[arg(long, default_value = "0.9.5")]
    ltxv_version: String,

    #[arg(long, default_value = "Lightricks/LTX-Video")]
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
    let ltxv_config =
        candle_video::models::ltx_video::configs::get_config_by_version(&args.ltxv_version);
    let num_inference_steps = args
        .steps
        .unwrap_or(ltxv_config.inference.num_inference_steps);
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

    println!("LTX-Video Text-to-Video Generation");
    println!("==================================");
    println!("Prompt: {}", args.prompt);
    println!(
        "Version: {} (steps={}, guidance={:.2}, rescale={:.2}, stg={:.2}, stochastic={})",
        args.ltxv_version,
        num_inference_steps,
        guidance_scale,
        rescaling_scale,
        stg_scale,
        stochastic_sampling
    );
    println!(
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
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        now.as_secs() ^ (now.subsec_nanos() as u64)
    });
    device.set_seed(seed)?;
    println!("Device: {:?}", device);
    println!("Seed: {}", seed);

    let dtype = DType::BF16;

    // 1. Locate weights
    let (transformer_file, vae_file, t5_file, tokenizer_file) =
        if let Some(local_path) = &args.local_weights {
            let base = PathBuf::from(local_path);

            let transformer = if args.unified_weights.is_some() {
                // When using unified weights, transformer is embedded in the single file
                PathBuf::from("dummy") // Will be ignored
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

            let vae = if args.unified_weights.is_some() {
                // When using unified weights, VAE is embedded in the single file
                PathBuf::from("dummy") // Will be ignored
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
                    println!("    Tokenizer not found locally, downloading from HF...");
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
            println!("\nDownloading models from HuggingFace: oxide-lab/LTX-Video-0.9.5");
            let api = Api::new()?;
            let repo = api.repo(Repo::with_revision(
                "oxide-lab/LTX-Video-0.9.5".into(),
                RepoType::Model,
                "main".into(),
            ));

            let transformer = repo.get("transformer/diffusion_pytorch_model.safetensors")?;
            let _ = repo.get("transformer/config.json")?;

            let vae = repo.get("vae/diffusion_pytorch_model.safetensors")?;
            let _ = repo.get("vae/config.json")?;

            let t5 = repo.get("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf")?;

            let tokenizer = repo.get("text_encoder_gguf/tokenizer.json")?;

            (transformer, vae, t5, tokenizer)
        };

    if args.local_weights.is_some() {
        println!(
            "\nLoading models from local path: {}",
            args.local_weights.as_ref().unwrap()
        );
    }

    // 2. Step 1: Encode prompts with T5 (or load from file)
    println!("Encoding prompt...");
    let (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = if let Some(ref emb_file) = args.embeddings_file {
        // Load pre-computed embeddings from file
        println!("  Loading embeddings from {}...", emb_file);
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

        println!(
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
            println!("  Loading BF16 T5 model...");
            let config = T5EncoderConfig::t5_xxl();
            let mut t5_wrapper = T5TextEncoderWrapper::new(config, device.clone(), DType::BF16)?;

            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[&t5_file], DType::BF16, &device)? };
            t5_wrapper.load_model(vb)?;

            let p_emb = t5_wrapper.forward(&p_ids)?;
            let n_emb = t5_wrapper.forward(&n_ids)?;
            (p_emb, n_emb)
        } else {
            println!("  Loading GGUF T5 model...");
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

    // 3. Step 2: Load Transformer and VAE for generation
    // 3. Step 2: Load Transformer and VAE for generation
    println!("Loading models...");

    // Check if using unified weights (official LTX-Video format)
    let (vae, transformer) = if let Some(ref unified_path) = args.unified_weights {
        use candle_video::models::ltx_video::weight_format::KeyRemapper;

        println!("  Loading from unified weights: {}", unified_path);
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

        println!(
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

    // Generate or load initial latents
    let latents_packed = if let Some(ref latents_file) = args.initial_latents_file {
        println!("  Loading initial latents from {}...", latents_file);
        let tensors = candle_core::safetensors::load(latents_file, &device)?;
        let latents = tensors
            .get("initial_latents")
            .ok_or_else(|| anyhow::anyhow!("initial_latents not found in file"))?
            .clone();
        println!("  Loaded latents shape: {:?}", latents.dims());
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

    // 5. Run Generation
    println!(
        "\nStarting denoising loop ({} steps)...",
        num_inference_steps
    );
    pipeline.guidance_rescale = rescaling_scale;

    let sigmas_from_config = ltxv_config.inference.timesteps.clone();
    let decode_timestep = ltxv_config
        .inference
        .decode_timestep
        .clone()
        .unwrap_or(vec![0.0]);
    let decode_noise_scale = ltxv_config.inference.decode_noise_scale.clone();

    let video_out = pipeline.call(
        None, // prompt
        None, // negative_prompt
        args.height,
        args.width,
        args.num_frames,
        25,
        num_inference_steps,
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
    )?;

    // 5. Save output
    if !Path::new(&args.output_dir).exists() {
        std::fs::create_dir_all(&args.output_dir)?;
    }

    let (b, _c, f, h, w) = video_out.frames.dims5()?;

    // Convert frames to Vec<Vec<u8>> for reuse
    let mut frame_data = Vec::new();
    for i in 0..b {
        for j in 0..f {
            let frame = video_out.frames.i((i, .., j, .., ..))?;
            let frame = frame
                .permute((1, 2, 0))?
                .clamp(0.0, 255.0)?
                .to_dtype(DType::U8)?;
            let data = frame.flatten_all()?.to_vec1::<u8>()?;
            frame_data.push(data);
        }
    }

    // Exclusive mode: --frames saves ONLY frames
    if args.frames {
        for (j, data) in frame_data.iter().enumerate() {
            let filename = format!("{}/frame_{:04}.png", args.output_dir, j);
            image::save_buffer(&filename, data, w as u32, h as u32, image::ColorType::Rgb8)?;
        }
        println!(
            "\nDone! Saved {} frames to {}",
            frame_data.len(),
            args.output_dir
        );
        return Ok(());
    }

    // Default or --gif: Save GIF
    {
        use gif::{Encoder, Repeat};
        use rayon::prelude::*;
        use std::fs::File;

        println!("Creating GIF animation (default output, accelerated with rayon)...");
        let gif_path = format!("{}/video.gif", args.output_dir);
        let mut image_file = File::create(&gif_path)?;
        let mut encoder = Encoder::new(&mut image_file, w as u16, h as u16, &[])?;
        encoder.set_repeat(Repeat::Infinite)?;

        // Parallel quantization
        let frames: Vec<_> = frame_data
            .par_iter()
            .map(|data| {
                let mut frame = gif::Frame::from_rgb_speed(w as u16, h as u16, data, 30);
                frame.delay = 4; // ~25 FPS
                frame
            })
            .collect();

        // Sequential write
        for frame in frames {
            encoder.write_frame(&frame)?;
        }
        println!("\nDone! Saved GIF to {}", gif_path);
    }

    Ok(())
}

use candle_video::models::ltx_video::t2v_pipeline::VaeConfig;
