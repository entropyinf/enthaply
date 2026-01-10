//! Z-Image pipeline implementation

use crate::Res;
use crate::models::z_image::scheduler::FlowMatchEulerDiscreteScheduler;
use crate::models::z_image::transformer::ZImageTransformer2DModel;
use crate::models::z_image::utils::{calculate_shift, rand_latents};
use anyhow::bail;
use candle_core::{Device, Module, Tensor, error};
use candle_transformers::models::stable_diffusion;
use stable_diffusion::vae;
use tokenizers::Tokenizer;

const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;
const DEFAULT_HEIGHT: usize = 1024;
const DEFAULT_WIDTH: usize = 1024;
const DEFAULT_INFERENCE_STEPS: usize = 8;
const DEFAULT_GUIDANCE_SCALE: f64 = 0.0;

pub struct GenerateConfig {
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub seed: u64,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            height: DEFAULT_HEIGHT,
            width: DEFAULT_WIDTH,
            num_inference_steps: DEFAULT_INFERENCE_STEPS,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
            seed: 42,
        }
    }
}

pub struct Model {
    pub transformer: ZImageTransformer2DModel,
    pub vae: vae::AutoEncoderKL,
    pub text_encoder: Box<dyn Module>,
    pub tokenizer: Tokenizer,
    pub scheduler: FlowMatchEulerDiscreteScheduler,
}

impl Model {
    pub fn generate(&mut self, prompt: &str, config: GenerateConfig) -> Res<Tensor> {
        let device = &Device::Cpu;

        let vae_scale_factor = 8;
        let vae_scale = vae_scale_factor * 2;

        if config.height % vae_scale != 0 {
            bail!("Height must be divisible by {}", vae_scale);
        }

        if config.width % vae_scale != 0 {
            bail!("Width must be divisible by {}", vae_scale);
        }

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| error::Error::msg(format!("{e}")))?;

        let tokens = tokens
            .get_ids()
            .into_iter()
            .map(|x| *x)
            .collect::<Vec<u32>>();

        let length = tokens.len();
        let tokens = Tensor::from_vec(tokens, (1, length), &device)?;
        let prompt_embeds = self.text_encoder.forward(&tokens)?;

        // Prepare latent variables
        let batch_size = 1;
        let num_channels_latents = 16; // transformer.in_channels
        let height_latent = 2 * (config.height / vae_scale);
        let width_latent = 2 * (config.width / vae_scale);

        let shape = &[
            batch_size,
            num_channels_latents,
            height_latent,
            width_latent,
        ];

        let latents = rand_latents(shape, config.seed, device)?;

        // Calculate shift
        let image_seq_len = (latents.dims()[2] / 2) * (latents.dims()[3] / 2);
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );

        // Set timesteps
        let mut scheduler = self.scheduler.clone();
        scheduler.set_timesteps(config.num_inference_steps, device, Some(mu))?;

        // Denoising loop
        let mut current_latents = latents;
        for &timestep in &scheduler.timesteps[..scheduler.timesteps.len() - 1] {
            // Prepare latent model input
            let latent_model_input = current_latents.copy()?;

            // Convert timestep to tensor
            let timestep_tensor =
                Tensor::new(&[timestep as f32], device)?.broadcast_as(current_latents.dims())?;

            // Model prediction
            let (noise_pred, _) = self.transformer.forward(
                &latent_model_input.unsqueeze(2)?,
                &timestep_tensor,
                &prompt_embeds,
                None,
                None,
            )?;

            current_latents =
                self.scheduler
                    .step(&noise_pred.squeeze(2)?, timestep, &current_latents)?;
        }

        let decoded = self.vae.decode(&current_latents)?;

        Ok(decoded)
    }
}
