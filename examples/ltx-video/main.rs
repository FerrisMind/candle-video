use candle_core::{DType, Device, IndexOp, Tensor};
use candle_video::models::ltx_video::{
    AutoencoderKLLtxVideo, LtxVideoTransformer3DModel,
    loader::WeightLoader,
    ltx_transformer::LtxVideoTransformer3DModelConfig,
    quantized_t5_encoder::QuantizedT5EncoderModel,
    scheduler::{FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig},
    t2v_pipeline::{LtxPipeline, LtxVideoProcessor, OutputType, TextEncoder, Tokenizer},
    vae::AutoencoderKLLtxVideoConfig,
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

    #[arg(
        long,
        default_value = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
    )]
    negative_prompt: String,

    #[arg(long, default_value_t = 30)]
    steps: usize,

    #[arg(long, default_value_t = 512)]
    height: usize,

    #[arg(long, default_value_t = 768)]
    width: usize,

    #[arg(long, default_value_t = 97)]
    num_frames: usize,

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

    #[arg(long, default_value_t = 3.0)]
    guidance_scale: f32,

    /// Use VAE tiling (spatial)
    #[arg(long, default_value_t = false)]
    vae_tiling: bool,

    /// Use VAE slicing (batch)
    #[arg(long)]
    vae_slicing: bool,
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
    println!("LTX-Video Text-to-Video Generation");
    println!("==================================");
    println!("Prompt: {}", args.prompt);
    println!("Size: {}x{} [{} frames]", args.width, args.height, args.num_frames);
    println!("Steps: {}", args.steps);
    println!("Guidance scale: {:.2}", args.guidance_scale);

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

            let transformer = if base
                .join("transformer/diffusion_pytorch_model.safetensors")
                .exists()
            {
                base.join("transformer/diffusion_pytorch_model.safetensors")
            } else if base.join("ltx_video_transformer_3d.safetensors").exists() {
                base.join("ltx_video_transformer_3d.safetensors")
            } else {
                anyhow::bail!("Transformer weights not found in {:?}", base);
            };

            let vae = if base
                .join("vae/diffusion_pytorch_model.safetensors")
                .exists()
            {
                base.join("vae/diffusion_pytorch_model.safetensors")
            } else if base.join("vae.safetensors").exists() {
                base.join("vae.safetensors")
            } else {
                anyhow::bail!("VAE weights not found in {:?}", base);
            };

            let t5 = if base.join("t5-v1_1-xxl-encoder-Q5_K_M.gguf").exists() {
                base.join("t5-v1_1-xxl-encoder-Q5_K_M.gguf")
            } else if base
                .join("text_encoder_gguf")
                .join("t5-v1_1-xxl-encoder-Q5_K_M.gguf")
                .exists()
            {
                base.join("text_encoder_gguf")
                    .join("t5-v1_1-xxl-encoder-Q5_K_M.gguf")
            } else if base.join("t5-xxl.gguf").exists() {
                base.join("t5-xxl.gguf")
            } else {
                anyhow::bail!("T5 GGUF not found in {:?}", base);
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
        println!("\nLoading models from local path: {}", args.local_weights.as_ref().unwrap());
    }

    // 2. Step 1: Encode prompts with T5
    println!("Encoding prompt...");
    let (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = {
        let t5_model = QuantizedT5EncoderModel::load(&t5_file, &device)?;

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

        let p_emb = t5_model.forward(&p_ids, Some(&p_mask))?;
        let n_emb = t5_model.forward(&n_ids, Some(&n_mask))?;

        (
            p_emb,
            p_mask.to_dtype(DType::F32)?,
            n_emb,
            n_mask.to_dtype(DType::F32)?,
        )
    };
    // T5 and tokenizer are dropped here

    // 3. Step 2: Load Transformer and VAE for generation
    // 3. Step 2: Load Transformer and VAE for generation
    println!("Loading models...");

    // VAE
    let vae_load = WeightLoader::new(device.clone(), dtype);
    let vae_vb = vae_load.load_single(&vae_file)?;
    let vae_dir = vae_file.parent().unwrap_or(Path::new("."));
    let mut vae_config = if vae_dir.join("config.json").exists() {
        let f = std::fs::File::open(vae_dir.join("config.json"))?;
        serde_json::from_reader(f)?
    } else {
        AutoencoderKLLtxVideoConfig::default()
    };
    vae_config.timestep_conditioning = true;
    let mut vae = AutoencoderKLLtxVideo::new(vae_config, vae_vb)?;
    vae.use_tiling = args.vae_tiling;
    vae.use_slicing = args.vae_slicing;
    // Auto-enable framewise decoding for long sequences if tiling is on
    vae.use_framewise_decoding = args.vae_tiling && args.num_frames > 16;

    // Transformer
    let trans_load = WeightLoader::new(device.clone(), dtype);
    let trans_vb = trans_load.load_single(&transformer_file)?;
    let trans_config = LtxVideoTransformer3DModelConfig::default();
    let transformer = LtxVideoTransformer3DModel::new(&trans_config, trans_vb)?;

    // Scheduler
    let scheduler_config = FlowMatchEulerDiscreteSchedulerConfig::default();
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
    let channels = 128; // T5/Transformer hidden dim? No, transformer in_channels.

    // [B, C, F, H, W]
    let shape = (1, channels, latent_frames, latent_height, latent_width);
    let latents_unpacked = rng.randn(shape, &device)?;

    let latents_packed = LtxPipeline::pack_latents(
        &latents_unpacked,
        transformer_spatial_patch,
        transformer_temporal_patch,
    )?;

    // 5. Run Generation
    println!("\nStarting denoising loop ({} steps)...", args.steps);
    let video_out = pipeline.call(
        None, // prompt
        None, // negative_prompt
        args.height,
        args.width,
        args.num_frames,
        25,
        args.steps,
        None,
        args.guidance_scale,
        0.0,
        1,
        Some(latents_packed), // Pass packed latents
        Some(prompt_embeds),
        Some(prompt_attention_mask),
        Some(negative_prompt_embeds),
        Some(negative_prompt_attention_mask),
        vec![0.05],
        None,
        OutputType::Tensor,
        128,
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
        println!("\nDone! Saved {} frames to {}", frame_data.len(), args.output_dir);
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
