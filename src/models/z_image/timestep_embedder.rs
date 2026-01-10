use crate::quantized_var_builder::VarBuilder;
use crate::{Res, quantized_nn};
use candle_core::{DType, Module, Tensor};

const FREQUENCY_EMBEDDING_SIZE: usize = 256;
const MAX_PERIOD: f64 = 10000.0;

pub struct TimestepEmbedder {
    mlp: candle_nn::Sequential,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(out_size: usize, mid_size: Option<usize>, vb: VarBuilder) -> Res<Self> {
        let mid_size = mid_size.unwrap_or(out_size);
        let mlp = candle_nn::seq()
            .add(quantized_nn::linear(
                FREQUENCY_EMBEDDING_SIZE,
                mid_size,
                vb.pp("mlp.0"),
            )?)
            .add(candle_nn::Activation::Silu)
            .add(quantized_nn::linear(mid_size, out_size, vb.pp("mlp.2"))?);

        Ok(Self {
            mlp,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    fn timestep_embedding(t: &Tensor, dim: usize, max_period: f64) -> candle_core::Result<Tensor> {
        let device = t.device();
        let half = dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(max_period as f32).ln() * (i as f32) / (half as f32)).exp())
            .collect();

        let freqs = Tensor::from_vec(freqs, (1, half), device)?;
        let args = t.unsqueeze(1)?.broadcast_mul(&freqs)?;

        let cos_args = args.cos()?;
        let sin_args = args.sin()?;
        let embedding = Tensor::cat(&[&cos_args, &sin_args], 2)?;

        if dim % 2 == 1 {
            let zeros = Tensor::zeros(
                (embedding.dim(0)?, embedding.dim(1)?, 1),
                DType::F32,
                device,
            )?;
            Tensor::cat(&[&embedding, &zeros], 2)
        } else {
            Ok(embedding)
        }
    }

    pub fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size, MAX_PERIOD)?;
        self.mlp.forward(&t_freq)
    }
}
