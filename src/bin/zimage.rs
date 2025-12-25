use anyhow::anyhow;
use candle_core::{DType, Device, utils};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use candle_transformers::models::stable_diffusion::vae;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;
use clap::Parser;
use enthalpy::models::z_image;
use enthalpy::models::z_image::pipeline::{GenerateConfig, Model};
use enthalpy::models::z_image::text_encoder::Qwen3TextEncoder;
use enthalpy::models::z_image::transformer::ZImageTransformer2DModel;
use enthalpy::util::modelscope::ModelScopeRepo;
use std::path::PathBuf;
use tokenizers::models::bpe::BPE;
use tracing::Level;
use z_image::{scheduler, transformer};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the model directory
    #[clap(short, long, default_value = "ckpts/Z-Image-Turbo")]
    model_path: PathBuf,

    /// Output image path
    #[clap(short, long, default_value = "example.png")]
    output_path: PathBuf,

    /// Prompt for image generation
    #[clap(
        short,
        long,
        default_value = "A beautiful landscape with mountains and lakes"
    )]
    prompt: String,

    /// Image height
    #[clap(long, default_value = "1024")]
    height: usize,

    /// Image width
    #[clap(long, default_value = "1024")]
    width: usize,

    /// Number of inference steps
    #[clap(long, default_value = "8")]
    num_inference_steps: usize,

    /// Guidance scale
    #[clap(long, default_value = "0.0")]
    guidance_scale: f64,

    /// Random seed
    #[clap(long, default_value = "42")]
    seed: u64,
}

struct RequiredPaths {
    tokenizer_vocab: PathBuf,
    tokenizer_merges: PathBuf,
    vae_config: PathBuf,
    vae: PathBuf,
    text_encoder_config: PathBuf,
    text_encoder: PathBuf,
    transformer_config: PathBuf,
    transformer: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .compact()
        .init();

    let device = match utils::metal_is_available() {
        true => Device::new_metal(0)?,
        false => match utils::cuda_is_available() {
            true => Device::new_cuda(0)?,
            false => Device::Cpu,
        },
    };

    let model_dir = PathBuf::from("D:/Users/entropy");

    tracing::info!("Loading model from {}", model_dir.display());

    // Load or download required files
    let repo = ModelScopeRepo::new(
        "Tongyi-MAI/Z-Image-Turbo",
        model_dir.join(".cache/modelscope/hub/models"),
    );
    let paths = RequiredPaths {
        tokenizer_vocab: repo.get("tokenizer/vocab.json").await?,
        tokenizer_merges: repo.get("tokenizer/merges.txt").await?,
        vae_config: repo.get("vae/config.json").await?,
        vae: repo.get("vae/diffusion_pytorch_model.safetensors").await?,
        text_encoder_config: repo.get("text_encoder/config.json").await?,
        text_encoder: PathBuf::from("D:/ComfyUI/models/text_encoders/qwen_3_4b.safetensors"),
        transformer_config: repo.get("transformer/config.json").await?,
        transformer: PathBuf::from(
            "D:/ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors",
        ),
    };
    
    // Load the transformer
    let transformer = {
        tracing::info!("Loading transformer from {}", paths.transformer.display());
        let config = std::fs::read(paths.transformer_config)?;
        let config: transformer::ZImageTransformerConfig = serde_json::from_slice(&config)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[paths.transformer], DType::F32, &device)
        }?;
        ZImageTransformer2DModel::new(config, vb)?
    };

    // Load the tokenizer
    let tokenizer = {
        tracing::info!("Loading tokenizer from {}", paths.tokenizer_vocab.display());
        let vocab = paths
            .tokenizer_vocab
            .to_str()
            .ok_or_else(|| anyhow!("Invalid vocab path"))?;
        let merges = paths
            .tokenizer_merges
            .to_str()
            .ok_or_else(|| anyhow!("Invalid merges path"))?;
        let bpe_builder = BPE::from_file(vocab, merges);
        let bpe = bpe_builder.build().map_err(|e| anyhow!(e))?;
        tokenizers::Tokenizer::new(bpe)
    };

    // Load the VAE
    let vae = {
        tracing::info!("Loading VAE from {}", paths.vae.display());
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[paths.vae], DType::F32, &device)? };
        let config = std::fs::read(paths.vae_config)?;
        #[derive(serde::Deserialize)]
        struct _AutoEncoderKLConfig {
            pub block_out_channels: Vec<usize>,
            pub layers_per_block: usize,
            pub latent_channels: usize,
            pub norm_num_groups: usize,
            pub use_quant_conv: bool,
            pub use_post_quant_conv: bool,
        }
        let config: _AutoEncoderKLConfig = serde_json::from_slice(&config)?;
        let config = AutoEncoderKLConfig {
            block_out_channels: config.block_out_channels,
            layers_per_block: config.layers_per_block,
            latent_channels: config.latent_channels,
            norm_num_groups: config.norm_num_groups,
            use_quant_conv: config.use_quant_conv,
            use_post_quant_conv: config.use_post_quant_conv,
        };
        vae::AutoEncoderKL::new(vb, 3, 3, config)?
    };

    // Load the text encoder
    let text_encoder = {
        tracing::info!("Loading text encoder from {}", paths.text_encoder.display());
        let config = std::fs::read(paths.text_encoder_config)?;
        let config = serde_json::from_slice::<qwen3::Config>(&config)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[paths.text_encoder], DType::F32, &device)
        }?;
        Box::from(Qwen3TextEncoder::new(&config, vb)?)
    };


    // Init the scheduler
    let scheduler = scheduler::FlowMatchEulerDiscreteScheduler::new(1000, 1.0, false);

    let mut model = Model {
        vae,
        tokenizer,
        text_encoder,
        transformer,
        scheduler,
    };

    let config = GenerateConfig {
        height: args.height,
        width: args.width,
        num_inference_steps: args.num_inference_steps,
        guidance_scale: args.guidance_scale,
        seed: args.seed,
    };

    let image_data = model.generate(&args.prompt, config)?;

    std::fs::write(&args.output_path, image_data)?;
    println!("Image saved to {:?}", args.output_path);

    Ok(())
}
