use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, LayerNorm, LayerNormConfig, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

// Constants for model configuration
const DEFAULT_TRANSFORMER_PATCH_SIZE: usize = 2;
const DEFAULT_TRANSFORMER_F_PATCH_SIZE: usize = 1;
const DEFAULT_TRANSFORMER_IN_CHANNELS: usize = 16;
const DEFAULT_TRANSFORMER_DIM: usize = 3840;
const DEFAULT_TRANSFORMER_N_LAYERS: usize = 30;
const DEFAULT_TRANSFORMER_N_REFINER_LAYERS: usize = 2;
const DEFAULT_TRANSFORMER_N_HEADS: usize = 30;
const DEFAULT_TRANSFORMER_N_KV_HEADS: usize = 30;
const DEFAULT_TRANSFORMER_NORM_EPS: f64 = 1e-5;
const DEFAULT_TRANSFORMER_QK_NORM: bool = true;
const DEFAULT_TRANSFORMER_CAP_FEAT_DIM: usize = 2560;
const DEFAULT_TRANSFORMER_T_SCALE: f64 = 1000.0;

const ROPE_THETA: f64 = 256.0;
const ROPE_AXES_DIMS: [usize; 3] = [32, 48, 48];
const ROPE_AXES_LENS: [usize; 3] = [1536, 512, 512];
const FREQUENCY_EMBEDDING_SIZE: usize = 256;
const MAX_PERIOD: f64 = 10000.0;
const ADALN_EMBED_DIM: usize = 3840; // min(dim, ADALN_EMBED_DIM) where dim=3840

#[derive(Debug, Clone, Deserialize)]
pub struct ZImageTransformerConfig {
    pub all_patch_size: Vec<usize>,
    pub all_f_patch_size: Vec<usize>,
    pub in_channels: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub norm_eps: f64,
    pub qk_norm: bool,
    pub cap_feat_dim: usize,
    pub rope_theta: f64,
    pub t_scale: f64,
    pub axes_dims: Vec<usize>,
    pub axes_lens: Vec<usize>,
}

impl Default for ZImageTransformerConfig {
    fn default() -> Self {
        Self {
            all_patch_size: vec![DEFAULT_TRANSFORMER_PATCH_SIZE],
            all_f_patch_size: vec![DEFAULT_TRANSFORMER_F_PATCH_SIZE],
            in_channels: DEFAULT_TRANSFORMER_IN_CHANNELS,
            dim: DEFAULT_TRANSFORMER_DIM,
            n_layers: DEFAULT_TRANSFORMER_N_LAYERS,
            n_refiner_layers: DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
            n_heads: DEFAULT_TRANSFORMER_N_HEADS,
            n_kv_heads: DEFAULT_TRANSFORMER_N_KV_HEADS,
            norm_eps: DEFAULT_TRANSFORMER_NORM_EPS,
            qk_norm: DEFAULT_TRANSFORMER_QK_NORM,
            cap_feat_dim: DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
            rope_theta: ROPE_THETA,
            t_scale: DEFAULT_TRANSFORMER_T_SCALE,
            axes_dims: ROPE_AXES_DIMS.to_vec(),
            axes_lens: ROPE_AXES_LENS.to_vec(),
        }
    }
}

// Helper function for RoPE
fn precompute_freqs_cis(
    dim: &[usize],
    end: &[usize],
    theta: f64,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut freqs_cis = Vec::new();
    for (i, (&d, &e)) in dim.iter().zip(end.iter()).enumerate() {
        let freqs: Vec<f64> = (0..d)
            .step_by(2)
            .map(|j| 1.0 / (theta.powf((j as f64) / (d as f64))))
            .collect();
        let freqs_len = freqs.len();
        let freqs = Tensor::from_vec(freqs, (freqs_len,), device)?.to_dtype(DType::F32)?;

        let timestep: Vec<f64> = (0..e).map(|x| x as f64).collect();
        let timestep_len = timestep.len();
        let timestep = Tensor::from_vec(timestep, (timestep_len,), device)?.to_dtype(DType::F32)?;

        let freqs = timestep.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;
        let cos_freqs = freqs.cos()?;
        let sin_freqs = freqs.sin()?;
        let cis = Tensor::cat(&[&cos_freqs.unsqueeze(2)?, &sin_freqs.unsqueeze(2)?], 2)?;
        freqs_cis.push(cis);
    }
    Ok(freqs_cis)
}

// RopeEmbedder equivalent
#[derive(Debug)]
struct RopeEmbedder {
    theta: f64,
    axes_dims: Vec<usize>,
    axes_lens: Vec<usize>,
    freqs_cis: Option<Vec<Tensor>>,
}

impl RopeEmbedder {
    fn new(theta: f64, axes_dims: Vec<usize>, axes_lens: Vec<usize>) -> Self {
        Self {
            theta,
            axes_dims,
            axes_lens,
            freqs_cis: None,
        }
    }

    fn call(&mut self, ids: &Tensor) -> Result<Tensor> {
        let device = ids.device();

        // Ensure ids is 2D
        if ids.rank() != 2 {
            candle_core::bail!("ids must be 2D");
        }

        if ids.dims()[1] != self.axes_dims.len() {
            candle_core::bail!("ids.shape[-1] must equal axes_dims length");
        }

        if self.freqs_cis.is_none() {
            self.freqs_cis = Some(precompute_freqs_cis(
                &self.axes_dims,
                &self.axes_lens,
                self.theta,
                device,
            )?);
        }

        let mut result = Vec::new();
        for i in 0..self.axes_dims.len() {
            let index = ids.i((.., i))?;
            let freq_cis = &self.freqs_cis.as_ref().unwrap()[i];
            let selected_freqs = freq_cis.index_select(&index.flatten_all()?, 0)?;
            result.push(selected_freqs);
        }

        Tensor::cat(&result, 1)
    }
}

// TimestepEmbedder equivalent
struct TimestepEmbedder {
    mlp: candle_nn::Sequential,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    fn new(out_size: usize, mid_size: Option<usize>, vb: VarBuilder) -> Result<Self> {
        let mid_size = mid_size.unwrap_or(out_size);
        let mlp = candle_nn::seq()
            .add(candle_nn::linear(
                FREQUENCY_EMBEDDING_SIZE,
                mid_size,
                vb.pp("mlp.0"),
            )?)
            .add(Activation::Silu)
            .add(candle_nn::linear(mid_size, out_size, vb.pp("mlp.2"))?);

        Ok(Self {
            mlp,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    fn timestep_embedding(t: &Tensor, dim: usize, max_period: f64) -> Result<Tensor> {
        let device = t.device();
        let half = dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-((max_period as f32).ln()) * (i as f32) / (half as f32)).exp())
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

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size, MAX_PERIOD)?;
        self.mlp.forward(&t_freq)
    }
}

// FeedForward equivalent
#[derive(Debug)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = candle_nn::linear(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = candle_nn::linear(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = candle_nn::linear(dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.w1.forward(x)?.silu()?;
        let x3 = self.w3.forward(x)?;
        self.w2.forward(&(x1 * x3)?)
    }
}

// FinalLayer equivalent
struct FinalLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: candle_nn::Sequential,
}

impl FinalLayer {
    fn new(hidden_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = candle_nn::layer_norm(
            hidden_size,
            LayerNormConfig {
                eps: 1e-6,
                remove_mean: true,
                affine: false,
            },
            vb.pp("norm_final"),
        )?;

        let linear = candle_nn::linear(hidden_size, out_channels, vb.pp("linear"))?;

        let ada_ln_modulation = candle_nn::seq()
            .add(Activation::Silu)
            .add(candle_nn::linear(
                hidden_size.min(ADALN_EMBED_DIM),
                hidden_size,
                vb.pp("adaLN_modulation.1"),
            )?);

        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let scale = (self.ada_ln_modulation.forward(c)? + 1.0)?;
        let x_norm = self.norm_final.forward(x)?;
        let x_scaled = x_norm.broadcast_mul(&scale.unsqueeze(1)?)?;
        self.linear.forward(&x_scaled)
    }
}

// ZImageTransformerBlock equivalent
struct ZImageTransformerBlock {
    attention_norm1: RmsNorm,
    ffn_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm2: RmsNorm,
    attention: ZImageAttention,
    feed_forward: FeedForward,
    ada_ln_modulation: Option<candle_nn::Sequential>,
    modulation: bool,
}

impl ZImageTransformerBlock {
    fn new(
        layer_id: usize,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        norm_eps: f64,
        qk_norm: bool,
        modulation: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attention_norm1 = candle_nn::rms_norm(dim, norm_eps, vb.pp("attention_norm1"))?;
        let ffn_norm1 = candle_nn::rms_norm(dim, norm_eps, vb.pp("ffn_norm1"))?;
        let attention_norm2 = candle_nn::rms_norm(dim, norm_eps, vb.pp("attention_norm2"))?;
        let ffn_norm2 = candle_nn::rms_norm(dim, norm_eps, vb.pp("ffn_norm2"))?;

        let attention = ZImageAttention::new(
            dim,
            n_heads,
            n_kv_heads,
            qk_norm,
            norm_eps,
            vb.pp("attention"),
        )?;
        let feed_forward = FeedForward::new(dim, dim * 8 / 3, vb.pp("feed_forward"))?;

        let ada_ln_modulation = if modulation {
            Some(candle_nn::seq().add(candle_nn::linear(
                dim.min(ADALN_EMBED_DIM),
                4 * dim,
                vb.pp("adaLN_modulation.0"),
            )?))
        } else {
            None
        };

        Ok(Self {
            attention_norm1,
            ffn_norm1,
            attention_norm2,
            ffn_norm2,
            attention,
            feed_forward,
            ada_ln_modulation,
            modulation,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        freqs_cis: Option<&Tensor>,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.modulation {
            if let (Some(ada_ln), Some(adaln_input)) = (&self.ada_ln_modulation, adaln_input) {
                let modulations = ada_ln.forward(adaln_input)?.unsqueeze(1)?;
                let chunks = modulations.chunk(4, 2)?;
                let scale_msa = (&chunks[0] + 1.0)?;
                let gate_msa = chunks[1].tanh()?;
                let scale_mlp = (&chunks[2] + 1.0)?;
                let gate_mlp = chunks[3].tanh()?;

                let attn_out = self.attention.forward(
                    &(&self.attention_norm1.forward(x)? * &scale_msa)?,
                    attn_mask,
                    freqs_cis,
                )?;

                let x =
                    (x + &gate_msa.broadcast_mul(&self.attention_norm2.forward(&attn_out)?)?)?;

                let ffn_input = self.ffn_norm1.forward(&x)?;
                let ffn_output = self.feed_forward.forward(&(&ffn_input * &scale_mlp)?)?;
                let ffn_normed = self.ffn_norm2.forward(&ffn_output)?;
                let x = (x + &gate_mlp.broadcast_mul(&ffn_normed)?)?;

                Ok(x)
            } else {
                candle_core::bail!("Modulation enabled but adaln_input not provided")
            }
        } else {
            let attn_out =
                self.attention
                    .forward(&self.attention_norm1.forward(x)?, attn_mask, freqs_cis)?;
            let x = (x + self.attention_norm2.forward(&attn_out)?)?;
            let x = (x.clone()
                + self
                    .ffn_norm2
                    .forward(&self.feed_forward.forward(&self.ffn_norm1.forward(&x)?)?)?)?;
            Ok(x)
        }
    }
}

// ZImageAttention equivalent
#[derive(Debug)]
struct ZImageAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Option<RmsNorm>,
    norm_k: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl ZImageAttention {
    fn new(
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        qk_norm: bool,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / n_heads;
        let to_q = candle_nn::linear(dim, n_heads * head_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(dim, n_kv_heads * head_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(dim, n_kv_heads * head_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(n_heads * head_dim, dim, vb.pp("to_out.0"))?;

        let norm_q = if qk_norm {
            Some(candle_nn::rms_norm(head_dim, eps, vb.pp("norm_q"))?)
        } else {
            None
        };

        let norm_k = if qk_norm {
            Some(candle_nn::rms_norm(head_dim, eps, vb.pp("norm_k"))?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            norm_q,
            norm_k,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        _attention_mask: Option<&Tensor>,
        _freqs_cis: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, n_tokens, _) = x.dims3()?;
        let query = self
            .to_q
            .forward(x)?
            .reshape((b_sz, n_tokens, self.n_heads, self.head_dim))?;
        let key =
            self.to_k
                .forward(x)?
                .reshape((b_sz, n_tokens, self.n_kv_heads, self.head_dim))?;
        let value =
            self.to_v
                .forward(x)?
                .reshape((b_sz, n_tokens, self.n_kv_heads, self.head_dim))?;

        // Apply normalization if needed
        let query = if let Some(norm_q) = &self.norm_q {
            norm_q.forward(&query)?
        } else {
            query
        };

        let key = if let Some(norm_k) = &self.norm_k {
            norm_k.forward(&key)?
        } else {
            key
        };

        // Apply RoPE if needed
        // Simplified handling - actual RoPE implementation would be more complex
        let (_query, _key) = if let Some(_freq_cis) = _freqs_cis {
            // TODO: Implement RoPE application
            (query, key)
        } else {
            (query, key)
        };

        // Perform attention calculation
        // This is a simplified version - full implementation would be more involved
        let out = self.to_out.forward(&value.flatten_from(2)?)?;

        Ok(out)
    }
}

pub struct ZImageTransformer2DModel {
    config: ZImageTransformerConfig,
    all_x_embedder: HashMap<String, Linear>,
    all_final_layer: HashMap<String, FinalLayer>,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    t_embedder: TimestepEmbedder,
    cap_embedder: candle_nn::Sequential,
    x_pad_token: Tensor,
    cap_pad_token: Tensor,
    layers: Vec<ZImageTransformerBlock>,
    rope_embedder: RopeEmbedder,
    in_channels: usize,
    out_channels: usize,
}

impl ZImageTransformer2DModel {
    pub fn new(config: ZImageTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let mut all_x_embedder = HashMap::new();
        let mut all_final_layer = HashMap::new();

        for (patch_size, f_patch_size) in config
            .all_patch_size
            .iter()
            .zip(config.all_f_patch_size.iter())
        {
            let key = format!("{}-{}", patch_size, f_patch_size);
            let x_embedder = candle_nn::linear(
                f_patch_size * patch_size * patch_size * config.in_channels,
                config.dim,
                vb.pp(format!("all_x_embedder.{}", key)),
            )?;
            all_x_embedder.insert(key.clone(), x_embedder);

            let final_layer = FinalLayer::new(
                config.dim,
                patch_size * patch_size * f_patch_size * config.in_channels,
                vb.pp(format!("all_final_layer.{}", key)),
            )?;
            all_final_layer.insert(key, final_layer);
        }

        let mut noise_refiner = Vec::new();
        for layer_id in 0..config.n_refiner_layers {
            noise_refiner.push(ZImageTransformerBlock::new(
                1000 + layer_id,
                config.dim,
                config.n_heads,
                config.n_kv_heads,
                config.norm_eps,
                config.qk_norm,
                true, // modulation
                vb.pp(format!("noise_refiner.{}", layer_id)),
            )?);
        }

        let mut context_refiner = Vec::new();
        for layer_id in 0..config.n_refiner_layers {
            context_refiner.push(ZImageTransformerBlock::new(
                layer_id,
                config.dim,
                config.n_heads,
                config.n_kv_heads,
                config.norm_eps,
                config.qk_norm,
                false, // modulation
                vb.pp(format!("context_refiner.{}", layer_id)),
            )?);
        }

        let t_embedder = TimestepEmbedder::new(
            config.dim.min(ADALN_EMBED_DIM),
            Some(1024),
            vb.pp("t_embedder"),
        )?;

        let cap_embedder = candle_nn::seq()
            .add(candle_nn::rms_norm(
                config.cap_feat_dim,
                config.norm_eps,
                vb.pp("cap_embedder.0"),
            )?)
            .add(candle_nn::linear(
                config.cap_feat_dim,
                config.dim,
                vb.pp("cap_embedder.1"),
            )?);

        let x_pad_token = vb.get((1, config.dim), "x_pad_token")?;
        let cap_pad_token = vb.get((1, config.dim), "cap_pad_token")?;

        let mut layers = Vec::new();
        for layer_id in 0..config.n_layers {
            layers.push(ZImageTransformerBlock::new(
                layer_id,
                config.dim,
                config.n_heads,
                config.n_kv_heads,
                config.norm_eps,
                config.qk_norm,
                true, // modulation
                vb.pp(format!("layers.{}", layer_id)),
            )?);
        }

        let rope_embedder = RopeEmbedder::new(
            config.rope_theta,
            config.axes_dims.clone(),
            config.axes_lens.clone(),
        );

        Ok(Self {
            config: config.clone(),
            all_x_embedder,
            all_final_layer,
            noise_refiner,
            context_refiner,
            t_embedder,
            cap_embedder,
            x_pad_token,
            cap_pad_token,
            layers,
            rope_embedder,
            in_channels: config.in_channels,
            out_channels: config.in_channels,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        patch_size: Option<usize>,
        f_patch_size: Option<usize>,
    ) -> Result<(Tensor, ())> {
        let patch_size = patch_size.unwrap_or(DEFAULT_TRANSFORMER_PATCH_SIZE);
        let f_patch_size = f_patch_size.unwrap_or(DEFAULT_TRANSFORMER_F_PATCH_SIZE);

        // Validate patch sizes
        if !self.config.all_patch_size.contains(&patch_size) {
            candle_core::bail!("Invalid patch_size: {}", patch_size);
        }
        if !self.config.all_f_patch_size.contains(&f_patch_size) {
            candle_core::bail!("Invalid f_patch_size: {}", f_patch_size);
        }

        // Apply time scaling
        let t_scaled = (t * self.config.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?;

        // In a real implementation, this would process the input through the transformer layers
        // For now, we'll just return the input tensor with no changes as a placeholder
        // A complete implementation would involve:
        // 1. Patchifying and embedding
        // 2. Applying noise refiner layers
        // 3. Processing caption features
        // 4. Applying context refiner layers
        // 5. Unified processing through main transformer layers
        // 6. Unpatchifying the result

        Ok((x.clone(), ()))
    }
}
