use anyhow::bail;
use candle_core::{DType, Device, IndexOp, Tensor, utils};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;
use enthalpy::util::modelscope::ModelScopeRepo;
use std::path::PathBuf;
use std::fs::File;
use std::io::Read;
use candle_core::safetensors::load_buffer;

struct RequiredPaths {
    vae_config: PathBuf,
    vae: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = match utils::metal_is_available() {
        true => Device::new_metal(0)?,
        false => match utils::cuda_is_available() {
            true => Device::new_cuda(0)?,
            false => Device::Cpu,
        },
    };

    // Load or download required files
    let repo = ModelScopeRepo::new(
        "Tongyi-MAI/Z-Image-Turbo",
        "/Users/entropy/.cache/modelscope/hub/models/",
    );
    let paths = RequiredPaths {
        vae_config: repo.get("vae/config.json").await?,
        vae: repo.get("vae/diffusion_pytorch_model.safetensors").await?,
    };

    // Load the VAE
    let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[paths.vae], DType::F32, &device)? };
    let vae_config = std::fs::read(paths.vae_config)?;
    #[derive(serde::Deserialize)]
    struct _AutoEncoderKLConfig {
        pub block_out_channels: Vec<usize>,
        pub layers_per_block: usize,
        pub latent_channels: usize,
        pub norm_num_groups: usize,
        pub use_quant_conv: bool,
        pub use_post_quant_conv: bool,
    }
    let vae_config: _AutoEncoderKLConfig = serde_json::from_slice(&vae_config)?;
    let vae_config = AutoEncoderKLConfig {
        block_out_channels: vae_config.block_out_channels,
        layers_per_block: vae_config.layers_per_block,
        latent_channels: vae_config.latent_channels,
        norm_num_groups: vae_config.norm_num_groups,
        use_quant_conv: vae_config.use_quant_conv,
        use_post_quant_conv: vae_config.use_post_quant_conv,
    };

    let latents = load_latent("ComfyUI_00001_.latent")?.samples.to_device(&device)?;
    println!("latents:{}", &latents);

    let autoencoder = vae::AutoEncoderKL::new(vae_vb, 3, 3, vae_config)?;

    println!("Generating image...");
    let img = autoencoder.decode(&latents)?;
    println!("Saving image...");
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    save_image(&img.i(0)?, "out.jpg")?;

    Ok(())
}

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> anyhow::Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle_core::Error::wrap)?;
    Ok(())
}

// Converted from Python LoadLatent class
pub struct Latent {
    pub samples: Tensor,
}

pub fn load_latent<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Latent> {
    let path = path.as_ref();
    
    // Read the file into a buffer
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Load tensors from the buffer
    let tensors = load_buffer(&buffer, &Device::Cpu)?;
    
    // Check for latent format version
    let multiplier = if tensors.contains_key("latent_format_version_0") {
        1.0
    } else {
        1.0 / 0.18215
    };
    
    // Get the latent tensor and apply multiplier
    let latent_tensor = tensors.get("latent_tensor")
        .ok_or_else(|| anyhow::anyhow!("Missing 'latent_tensor' in latent file"))?;
    let samples = (latent_tensor.as_ref().clone() * multiplier)?;
    
    Ok(Latent { samples })
}