use crate::models::z_image::feed_foward::FeedForward;
use crate::models::z_image::rope_embedder::RopeEmbedder;
use crate::models::z_image::timestep_embedder::TimestepEmbedder;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, LayerNorm, LayerNormConfig, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

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
const SEQ_MULTI_OF: usize = 4096; // Default value, may need to be adjusted based on actual usage

const ROPE_THETA: f64 = 256.0;
const ROPE_AXES_DIMS: [usize; 3] = [32, 48, 48];
const ROPE_AXES_LENS: [usize; 3] = [1536, 512, 512];
const ADALN_EMBED_DIM: usize = 256;

// Apply rotary embedding to query and key tensors
fn apply_rotary_emb(x_in: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
    // Python equivalent:
    // x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
    // freqs_cis = freqs_cis.unsqueeze(2)
    // x_out = torch.view_as_real(x * freqs_cis).flatten(3)
    // return x_out.type_as(x_in)

    let x_dtype = x_in.dtype();
    let x = x_in.to_dtype(DType::F32)?;

    // Get the shape of the input tensor
    let x_shape = x.dims();
    let last_dim = *x_shape.last().unwrap();

    // Reshape x to separate real and imaginary parts: [..., -1, 2]
    // We need to reshape the last dimension into (last_dim/2, 2) to form complex numbers
    let x_reshaped = x.reshape((x_shape[0], x_shape[1], x_shape[2], last_dim / 2, 2))?;

    // Split into real and imaginary parts
    let x_real = x_reshaped.i((.., .., .., .., 0))?;
    let x_imag = x_reshaped.i((.., .., .., .., 1))?;

    // freqs_cis shape: [seq_len, num_freqs, 2] where last dim is (cos, sin)
    // Need to make sure freqs_cis is compatible with x dimensions
    // We need to reshape freqs_cis to match x dimensions
    let freqs_cis_expanded = if freqs_cis.dims().len() == 2 {
        // If freqs_cis is [seq_len, head_dim], reshape to [1, seq_len, 1, head_dim/2, 2]
        let (seq_len, head_dim) = freqs_cis.dims2()?;
        freqs_cis.reshape((1, seq_len, 1, head_dim / 2, 2))?
    } else if freqs_cis.dims().len() == 3 {
        // If freqs_cis is [batch, seq_len, head_dim], reshape to [batch, seq_len, 1, head_dim/2, 2]
        let (batch, seq_len, head_dim) = freqs_cis.dims3()?;
        freqs_cis.reshape((batch, seq_len, 1, head_dim / 2, 2))?
    } else if freqs_cis.dims().len() == 4 {
        // If freqs_cis is [batch, seq_len, head_dim/2, 2], reshape to [batch, seq_len, 1, head_dim/2, 2]
        let (batch, seq_len, head_dim_div2, _) = freqs_cis.dims4()?;
        freqs_cis.reshape((batch, seq_len, 1, head_dim_div2, 2))?
    } else {
        freqs_cis.clone()
    };

    let freqs_real = freqs_cis_expanded.i((.., .., .., .., 0))?;
    let freqs_imag = freqs_cis_expanded.i((.., .., .., .., 1))?;

    // Apply the rotary embedding formula: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    // Where x_real and x_imag are 'a' and 'b', and freqs_real and freqs_imag are 'c' and 'd'
    let x_rotated_real =
        (x_real.broadcast_mul(&freqs_real)? - x_imag.broadcast_mul(&freqs_imag)?)?;
    let x_rotated_imag =
        (x_real.broadcast_mul(&freqs_imag)? + x_imag.broadcast_mul(&freqs_real)?)?;

    // Stack the real and imaginary parts back together
    let x_rotated = Tensor::cat(
        &[&x_rotated_real.unsqueeze(4)?, &x_rotated_imag.unsqueeze(4)?],
        4,
    )?;

    // Flatten the last dimension back to the original shape
    let result = x_rotated.flatten_from(3)?;

    // Convert back to original dtype
    result.to_dtype(x_dtype)
}

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
        _layer_id: usize,
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

struct ZImageAttention {
    qkv: Linear,
    out: Linear,
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
        let qkv =
            candle_nn::linear_no_bias(dim, (n_heads + 2 * n_kv_heads) * head_dim, vb.pp("qkv"))?;
        let out = candle_nn::linear_no_bias(n_heads * head_dim, dim, vb.pp("out"))?;

        let norm_q = if qk_norm {
            Some(candle_nn::rms_norm(head_dim, eps, vb.pp("q_norm"))?)
        } else {
            None
        };

        let norm_k = if qk_norm {
            Some(candle_nn::rms_norm(head_dim, eps, vb.pp("k_norm"))?)
        } else {
            None
        };

        Ok(Self {
            qkv,
            out,
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
        attention_mask: Option<&Tensor>,
        freqs_cis: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, n_tokens, _) = x.dims3()?;

        let qkv = self.qkv.forward(x)?;

        // Split qkv into query, key, value
        let q_start = 0;
        let q_end = self.n_heads * self.head_dim;
        let k_start = q_end;
        let k_end = k_start + self.n_kv_heads * self.head_dim;
        let v_start = k_end;
        let v_end = v_start + self.n_kv_heads * self.head_dim;

        let query = qkv.i((.., .., q_start..q_end))?.reshape((
            b_sz,
            n_tokens,
            self.n_heads,
            self.head_dim,
        ))?;
        let key = qkv.i((.., .., k_start..k_end))?.reshape((
            b_sz,
            n_tokens,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let value = qkv.i((.., .., v_start..v_end))?.reshape((
            b_sz,
            n_tokens,
            self.n_kv_heads,
            self.head_dim,
        ))?;

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
        let (query, key) = if let Some(freqs_cis) = freqs_cis {
            // Apply RoPE to query and key
            // Need to reshape for RoPE application
            let query_reshaped = query
                .transpose(1, 2)? // [b, n_tokens, n_heads, head_dim] -> [b, n_heads, n_tokens, head_dim]
                .contiguous()?;
            let key_reshaped = key
                .transpose(1, 2)? // [b, n_tokens, n_kv_heads, head_dim] -> [b, n_kv_heads, n_tokens, head_dim]
                .contiguous()?;

            let query_rotated = apply_rotary_emb(&query_reshaped, freqs_cis)?
                .transpose(1, 2)? // Back to [b, n_tokens, n_heads, head_dim]
                .contiguous()?;
            let key_rotated = apply_rotary_emb(&key_reshaped, freqs_cis)?
                .transpose(1, 2)? // Back to [b, n_tokens, n_kv_heads, head_dim]
                .contiguous()?;

            (query_rotated, key_rotated)
        } else {
            (query, key)
        };

        // Perform attention calculation
        // First, reshape query, key, and value for attention computation
        let query = query.transpose(1, 2)?; // [b, n_tokens, n_heads, head_dim]
        let key = key.transpose(1, 2)?; // [b, n_tokens, n_kv_heads, head_dim] 
        let value = value.transpose(1, 2)?; // [b, n_tokens, n_kv_heads, head_dim]

        // For multi-query attention, we need to repeat key and value to match query heads
        let key = if self.n_heads != self.n_kv_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            key.repeat((1, 1, repeats, 1))?
        } else {
            key
        };
        let value = if self.n_heads != self.n_kv_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            value.repeat((1, 1, repeats, 1))?
        } else {
            value
        };

        // Calculate attention scores
        let attn_scores = query.matmul(&key.t()?)?; // [b, n_tokens, n_heads, n_tokens]
        let attn_scores = (attn_scores / (self.head_dim as f64).sqrt())?;

        // Apply attention mask if provided
        let attn_scores = if let Some(mask) = attention_mask {
            attn_scores.broadcast_add(mask)?
        } else {
            attn_scores
        };

        // Apply softmax to get attention weights
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;

        // Apply attention weights to values
        let attn_output = attn_weights.matmul(&value)?; // [b, n_tokens, n_heads, head_dim]

        // Reshape to prepare for output projection
        let attn_output = attn_output
            .transpose(1, 2)? // [b, n_heads, n_tokens, head_dim]
            .reshape((b_sz, n_tokens, self.n_heads * self.head_dim))?;

        // Apply output projection
        let out = self.out.forward(&attn_output)?;

        Ok(out)
    }
}

pub struct ZImageTransformer2DModel {
    config: ZImageTransformerConfig,
    x_embedders: HashMap<String, Linear>,
    final_layers: HashMap<String, FinalLayer>,
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
        let mut x_embedders = HashMap::new();
        let mut final_layers = HashMap::new();

        for (patch_size, f_patch_size) in config
            .all_patch_size
            .iter()
            .zip(config.all_f_patch_size.iter())
        {
            let key = format!("{}-{}", patch_size, f_patch_size);
            let x_embedder = candle_nn::linear(
                f_patch_size * patch_size * patch_size * config.in_channels,
                config.dim,
                vb.pp("x_embedder"),
            )?;
            x_embedders.insert(key.clone(), x_embedder);

            let final_layer = FinalLayer::new(
                config.dim,
                patch_size * patch_size * f_patch_size * config.in_channels,
                VarBuilder::zeros(vb.dtype, vb.device()),
            )?;
            final_layers.insert(key, final_layer);
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
                true,
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
                false,
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
                true,
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
            x_embedders,
            final_layers,
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

    fn create_coordinate_grid(
        &self,
        size: &[usize],
        start: Option<&[usize]>,
        device: &Device,
    ) -> Result<Tensor> {
        // This creates a coordinate grid similar to the Python implementation
        // Since candle doesn't have a direct equivalent to torch.meshgrid, we implement a simplified version
        let total_elements: usize = size.iter().product();
        let mut coords = Vec::with_capacity(total_elements);

        // Generate coordinates based on size and start values
        let mut idx = vec![0; size.len()];
        let mut finished = false;

        while !finished {
            let mut coord_value = 0u32;
            for i in 0..size.len() {
                let start_val = start.map(|s| s[i]).unwrap_or(0);
                coord_value += (start_val + idx[i]) as u32;
            }
            coords.push(coord_value);

            // Increment indices like counting
            let mut carry = 1;
            for i in (0..size.len()).rev() {
                // Iterate in reverse to handle multi-dim indexing properly
                idx[i] += carry as usize;
                if idx[i] >= size[i] {
                    idx[i] = 0;
                    carry = 1;
                } else {
                    carry = 0;
                    break;
                }
            }

            if carry == 1 {
                finished = true;
            }
        }

        Tensor::from_vec(coords, size, device)
    }

    fn patchify_and_embed(
        &self,
        all_image: &[Tensor],
        all_cap_feats: &[Tensor],
        patch_size: usize,
        f_patch_size: usize,
    ) -> Result<(
        Vec<Tensor>,
        Vec<Tensor>,
        Vec<(usize, usize, usize)>,
        Vec<Tensor>,
        Vec<Tensor>,
        Vec<Tensor>,
        Vec<Tensor>,
    )> {
        let mut all_image_out = Vec::new();
        let mut all_image_size = Vec::new();
        let mut all_image_pos_ids = Vec::new();
        let mut all_image_pad_mask = Vec::new();
        let mut all_cap_pos_ids = Vec::new();
        let mut all_cap_pad_mask = Vec::new();
        let mut all_cap_feats_out = Vec::new();

        let device = all_image[0].device();
        let p_h = patch_size;
        let p_w = patch_size;
        let p_f = f_patch_size;

        for (image, cap_feat) in all_image.iter().zip(all_cap_feats.iter()) {
            // Process caption features
            let cap_ori_len = cap_feat.dims2()?.0; // Assuming cap_feat is [seq_len, feat_dim]
            let cap_padding_len = (SEQ_MULTI_OF - (cap_ori_len % SEQ_MULTI_OF)) % SEQ_MULTI_OF;

            // Create coordinate grid for caption
            let cap_size = vec![cap_ori_len + cap_padding_len, 1, 1];
            let cap_start = vec![1, 0, 0];
            let cap_padded_pos_ids =
                self.create_coordinate_grid(&cap_size, Some(&cap_start), device)?;
            all_cap_pos_ids.push(cap_padded_pos_ids.flatten(0, 2)?);

            // Create caption pad mask
            let mut cap_pad_mask_data = vec![0u32; cap_ori_len];
            if cap_padding_len > 0 {
                cap_pad_mask_data.extend(vec![1; cap_padding_len]);
            }
            let cap_pad_mask =
                Tensor::from_vec(cap_pad_mask_data, (cap_ori_len + cap_padding_len,), device)?;
            all_cap_pad_mask.push(cap_pad_mask);

            // Create padded caption features
            if cap_padding_len > 0 {
                // Get the last element of cap_feat and repeat it cap_padding_len times
                let last_idx = cap_feat.narrow(0, cap_ori_len - 1, 1)?;
                let mut cap_feat_out = cap_feat.clone();
                for _ in 0..cap_padding_len {
                    cap_feat_out = Tensor::cat(&[cap_feat_out.clone(), last_idx.clone()], 0)?;
                }
                all_cap_feats_out.push(cap_feat_out);
            } else {
                all_cap_feats_out.push(cap_feat.clone());
            }

            // Process image
            let (_c, f, h, w) = image.dims4()?;
            all_image_size.push((f, h, w));
            let f_tokens = f / p_f;
            let h_tokens = h / p_h;
            let w_tokens = w / p_w;

            // Reshape image into patches
            // image shape: [C, F, H, W] -> [F_tokens, H_tokens, W_tokens, p_f, p_h, p_w, C]
            let image_reshaped = image
                .reshape(vec![
                    self.in_channels,
                    f_tokens,
                    p_f,
                    h_tokens,
                    p_h,
                    w_tokens,
                    p_w,
                ])?
                .permute(vec![1, 3, 5, 2, 4, 6, 0])?
                .reshape((
                    f_tokens * h_tokens * w_tokens,
                    p_f * p_h * p_w * self.in_channels,
                ))?;

            let image_ori_len = image_reshaped.dims2()?.0;
            let image_padding_len = (SEQ_MULTI_OF - (image_ori_len % SEQ_MULTI_OF)) % SEQ_MULTI_OF;

            // Create image position IDs
            let image_size = vec![f_tokens, h_tokens, w_tokens];
            let image_start = vec![cap_ori_len + cap_padding_len + 1, 0, 0];
            let image_ori_pos_ids =
                self.create_coordinate_grid(&image_size, Some(&image_start), device)?;
            let image_ori_pos_ids = image_ori_pos_ids.flatten(0, 2)?;

            // Create padded position IDs
            let pad_pos_ids = Tensor::zeros((image_padding_len,), DType::U32, device)?;
            let image_padded_pos_ids = if image_padding_len > 0 {
                Tensor::cat(&[image_ori_pos_ids, pad_pos_ids], 0)?
            } else {
                image_ori_pos_ids
            };
            all_image_pos_ids.push(image_padded_pos_ids);

            // Create image pad mask
            let mut image_pad_mask_data = vec![0u32; image_ori_len];
            if image_padding_len > 0 {
                image_pad_mask_data.extend(vec![1u32; image_padding_len]);
            }
            let image_pad_mask = Tensor::from_vec(
                image_pad_mask_data,
                (image_ori_len + image_padding_len,),
                device,
            )?;
            all_image_pad_mask.push(image_pad_mask);

            // Create padded image features
            if image_padding_len > 0 {
                let last_idx = image_reshaped.narrow(0, image_ori_len - 1, 1)?;
                let mut image_out = image_reshaped.clone();
                for _ in 0..image_padding_len {
                    image_out = Tensor::cat(&[image_out.clone(), last_idx.clone()], 0)?;
                }
                all_image_out.push(image_out);
            } else {
                all_image_out.push(image_reshaped);
            }
        }

        Ok((
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        ))
    }

    fn unpatchify(
        &self,
        x: Vec<Tensor>,
        size: Vec<(usize, usize, usize)>,
        patch_size: usize,
        f_patch_size: usize,
    ) -> Result<Vec<Tensor>> {
        let bsz = x.len();
        assert_eq!(size.len(), bsz);

        let p_h = patch_size;
        let p_w = patch_size;
        let p_f = f_patch_size;

        let mut result = Vec::new();
        for i in 0..bsz {
            let (f, h, w) = size[i];
            let ori_len = (f / p_f) * (h / p_h) * (w / p_w);

            let x_i = &x[i];
            let seq_len = x_i.dims2()?.0;
            let actual_len = std::cmp::min(ori_len, seq_len);

            // Limit to original length
            let x_limited = x_i.narrow(0, 0, actual_len)?;

            // Reshape from patches back to original format
            let reshaped = x_limited.reshape(vec![
                f / p_f,
                h / p_h,
                w / p_w,
                p_f,
                p_h,
                p_w,
                self.out_channels,
            ])?;
            let final_tensor = reshaped.permute(vec![6, 0, 3, 1, 4, 2, 5])?.reshape((
                self.out_channels,
                f,
                h,
                w,
            ))?;

            result.push(final_tensor);
        }

        Ok(result)
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
        let t_emb = self.t_embedder.forward(&t_scaled)?;

        // Process the input tensors through the patchify_and_embed method
        // Since x and cap_feats are single tensors rather than lists in this simplified version,
        // we'll create single-element vectors to pass to patchify_and_embed
        let all_x = vec![x.clone()];
        let all_cap_feats = vec![cap_feats.clone()];

        let (
            x_out,
            cap_feats_out,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            _x_inner_pad_mask,
            _cap_inner_pad_mask,
        ) = self.patchify_and_embed(&all_x, &all_cap_feats, patch_size, f_patch_size)?;

        // Concatenate the outputs
        let x_concat = Tensor::cat(&x_out, 0)?;
        let cap_feats_concat = Tensor::cat(&cap_feats_out, 0)?;

        // Get the embedder for the specific patch size
        let embedder_key = format!("{}-{}", patch_size, f_patch_size);
        let x_embedder = &self.x_embedders[&embedder_key];

        // Embed the concatenated input
        let x_embed = x_embedder.forward(&x_concat)?;

        // Apply pad tokens where needed (simplified version)
        // In the Python implementation, this is done using indexing: x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        // For now, we'll skip this step in this simplified version

        // Apply RoPE embeddings using the rope_embedder
        let all_pos_ids = Tensor::cat(&[&x_pos_ids[0], &cap_pos_ids[0]], 0)?.unsqueeze(0)?;
        let freqs_cis = self.rope_embedder.call(&all_pos_ids)?;

        // Process through noise refiner layers
        let mut x_current = x_embed;
        for layer in &self.noise_refiner {
            x_current = layer.forward(&x_current, None, Some(&freqs_cis), Some(&t_emb))?;
        }

        // Process caption features
        let mut cap_current = self.cap_embedder.forward(&cap_feats_concat)?;
        for layer in &self.context_refiner {
            cap_current = layer.forward(&cap_current, None, Some(&freqs_cis), None)?;
        }

        // Combine x and caption features
        let combined = Tensor::cat(&[x_current, cap_current], 0)?;

        // Process through main transformer layers
        let mut unified = combined;
        for layer in &self.layers {
            unified = layer.forward(&unified, None, Some(&freqs_cis), Some(&t_emb))?;
        }

        // Apply final layer
        let final_layer = &self.final_layers[&embedder_key];
        let result = final_layer.forward(&unified, &t_emb)?;

        // Unpatchify the result
        let result_vec = vec![result];
        let unpatchified = self.unpatchify(result_vec, x_size, patch_size, f_patch_size)?;
        let final_result = unpatchified[0].clone();

        Ok((final_result, ()))
    }
}
