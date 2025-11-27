use crate::Res;
use crate::models::sense_voice_small::config::EncoderConfig;
use crate::var_builder::{Linear, VarBuilder};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Dropout, LayerNorm, Module};

pub struct Encoder {
    /// Embedding module
    embed: SinusoidalPositionEncoder,
    /// First encoder layers
    encoders0: Vec<EncoderLayerSANM>,
    /// Main encoder layers
    encoders: Vec<EncoderLayerSANM>,
    /// TP encoder layers
    tp_encoders: Vec<EncoderLayerSANM>,
    /// Layer normalization after main encoders
    after_norm: LayerNorm,
    /// Layer normalization after TP encoders
    tp_norm: LayerNorm,
    /// Output size
    output_size: usize,
}

impl Encoder {
    /// Create a new SenseVoiceEncoderSmall instance
    ///
    /// # Arguments
    /// * `config` - Configuration for the encoder
    /// * `vb` - VarBuilder for creating layers
    pub fn new_with_config(cfg: EncoderConfig, vb: VarBuilder) -> Res<Self> {
        // Create embedding module
        let embed = SinusoidalPositionEncoder;

        let create_layer =
            |input_size: usize, output_size: usize, vb: VarBuilder| -> Res<EncoderLayerSANM> {
                let self_attn = MultiHeadedAttentionSANM::new(
                    cfg.attention_heads,
                    input_size,
                    output_size,
                    cfg.attention_dropout_rate,
                    cfg.kernel_size,
                    cfg.sanm_shfit,
                    vb.pp("self_attn"),
                )?;

                let position_wise_layer = PositionwiseFeedForward::new(
                    cfg.output_size,
                    cfg.linear_units,
                    cfg.dropout_rate,
                    vb.pp("feed_forward"),
                )?;

                let encoder_layer = EncoderLayerSANM::new(
                    input_size,
                    output_size,
                    self_attn,
                    position_wise_layer,
                    cfg.dropout_rate,
                    cfg.normalize_before,
                    cfg.concat_after,
                    vb,
                )?;

                Ok(encoder_layer)
            };

        let vb = vb.pp("encoder");

        // Create first encoder layers (1 block)
        let encoders0 = {
            let mut encoders0 = Vec::new();
            let vb = vb.pp("encoders0");
            for i in 0..1 {
                encoders0.push(create_layer(cfg.input_size, cfg.output_size, vb.pp(i))?);
            }

            encoders0
        };

        // Create main encoder layers (num_blocks - 1 blocks)
        let encoders = {
            let mut encoders = Vec::new();
            let vb = vb.pp("encoders");
            for i in 0..(cfg.num_blocks - 1) {
                encoders.push(create_layer(cfg.output_size, cfg.output_size, vb.pp(i))?);
            }

            encoders
        };

        // Create TP encoder layers
        let tp_encoders = {
            let mut tp_encoders = Vec::new();
            let vb = vb.pp("tp_encoders");
            for i in 0..cfg.tp_blocks {
                tp_encoders.push(create_layer(cfg.output_size, cfg.output_size, vb.pp(i))?);
            }

            tp_encoders
        };

        // Create normalization layers

        let after_norm = vb.pp("after_norm").layer_norm(cfg.output_size, 1e-5)?;
        let tp_norm = vb.pp("tp_norm").layer_norm(cfg.output_size, 1e-5)?;

        Ok(Self {
            embed,
            encoders0,
            encoders,
            tp_encoders,
            after_norm,
            tp_norm,
            output_size: cfg.output_size,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `xs_pad` - Padded audio sequences (batch, time, size)
    /// * `ilens` - Input sequence lengths (batch,)
    ///
    /// # Returns
    /// * Output tensor (batch, time, size)
    /// * Output sequence lengths (batch,)
    pub fn forward(&self, xs_pad: &Tensor) -> Res<Tensor> {
        // Scale audio
        let xs_pad = (xs_pad * (self.output_size as f64).sqrt())?;

        // Apply embedding
        let mut xs_pad = self.embed.forward(&xs_pad)?;

        // Forward through first encoder layers
        for encoder_layer in &self.encoders0 {
            xs_pad = encoder_layer.forward(&xs_pad)?;
        }

        // Forward through main encoder layers
        for encoder_layer in &self.encoders {
            xs_pad = encoder_layer.forward(&xs_pad)?;
        }

        // Apply normalization after main encoders
        xs_pad = self.after_norm.forward(&xs_pad)?;

        // Forward through TP encoder layers
        for encoder_layer in &self.tp_encoders {
            xs_pad = encoder_layer.forward(&xs_pad)?;
        }

        // Apply normalization after TP encoders
        xs_pad = self.tp_norm.forward(&xs_pad)?;

        Ok(xs_pad)
    }
}

pub struct EncoderLayerSANM {
    /// Self attention module
    self_attn: MultiHeadedAttentionSANM,
    /// Feed forward module
    feed_forward: PositionwiseFeedForward,
    /// Layer normalization for attention
    norm1: LayerNorm,
    /// Layer normalization for feed forward
    norm2: LayerNorm,
    /// Dropout module
    dropout: Dropout,
    /// Input size
    in_size: usize,
    /// Hidden size
    size: usize,
    /// Whether to normalize before the computation
    normalize_before: bool,
    /// Whether to concatenate after attention
    concat_after: bool,
    /// Concatenation linear layer (when concat_after is true)
    concat_linear: Option<Linear>,
}

impl EncoderLayerSANM {
    /// Construct an EncoderLayerSANM object
    ///
    /// # Arguments
    /// * `in_size` - Input size
    /// * `size` - Hidden size
    /// * `self_attn` - Self attention module
    /// * `feed_forward` - Feed forward module
    /// * `dropout_rate` - Dropout rate
    /// * `normalize_before` - Whether to normalize before the computation
    /// * `concat_after` - Whether to concatenate after attention
    /// * `stochastic_depth_rate` - Stochastic depth rate
    pub fn new(
        in_size: usize,
        size: usize,
        self_attn: MultiHeadedAttentionSANM,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: f32,
        normalize_before: bool,
        concat_after: bool,
        vb: VarBuilder,
    ) -> Res<Self> {
        let norm1 = vb.pp("norm1").layer_norm(in_size, 1e-5)?;
        let norm2 = vb.pp("norm2").layer_norm(size, 1e-5)?;
        let dropout = Dropout::new(dropout_rate);

        let concat_linear = if concat_after {
            Some(vb.pp("concat_linear").linear(size + size, size)?)
        } else {
            None
        };

        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            dropout,
            in_size,
            size,
            normalize_before,
            concat_after,
            concat_linear,
        })
    }

    /// Compute encoded features
    ///
    /// # Arguments
    /// * `x` - Input tensor (batch, time, size)
    /// * `mask` - Mask tensor for the audio (batch, time)
    /// * `cache` - Cache tensor of the audio (batch, time - 1, size)
    ///
    /// # Returns
    /// * Output tensor (batch, time, size)
    /// * Mask tensor (batch, time)
    pub fn forward(&self, x: &Tensor) -> Res<Tensor> {
        let stoch_layer_coeff = 1.0;

        let residual = x.clone();
        let mut x = x.clone();
        if self.normalize_before {
            x = self.norm1.forward(&x)?;
        }

        if self.concat_after {
            let attn = self.self_attn.forward(&x)?;
            let x_concat = Tensor::cat(&[&x, &attn], 2)?;

            if let Some(concat_linear) = &self.concat_linear {
                x = if self.in_size == self.size {
                    let concat_res = concat_linear.forward(&x_concat)?;
                    (&residual + (stoch_layer_coeff * &concat_res)?)?
                } else {
                    concat_linear.forward(&x_concat)?
                };
            }
        } else {
            let attn = self.self_attn.forward(&x)?;
            let dropout_attn = self.dropout.forward(&attn, false)?;

            if self.in_size == self.size {
                x = (&residual + (stoch_layer_coeff * &dropout_attn)?)?;
            } else {
                x = (stoch_layer_coeff * &dropout_attn)?;
            }
        }

        if !self.normalize_before {
            x = self.norm1.forward(&x)?;
        }

        let residual = x.clone();
        if self.normalize_before {
            x = self.norm2.forward(&x)?;
        }

        let ff_res = self.feed_forward.forward(&x)?;
        let dropout_ff = self.dropout.forward(&ff_res, false)?;
        x = (&residual + (stoch_layer_coeff * &dropout_ff)?)?;

        if !self.normalize_before {
            x = self.norm2.forward(&x)?;
        }

        Ok(x)
    }
}

pub struct MultiHeadedAttentionSANM {
    /// The dimension of each head
    d_k: usize,
    /// The number of heads
    h: usize,
    /// Linear layer for output transformation
    linear_out: Linear,
    /// Combined linear layer for Q, K, V transformations
    linear_q_k_v: Linear,
    /// FSMN convolution block
    fsmn_block: Conv1d,
    /// Padding values for FSMN
    left_padding: usize,
    /// Padding values for FSMN
    right_padding: usize,
    /// Dropout layer
    dropout: Dropout,
}

impl MultiHeadedAttentionSANM {
    /// Creates a new MultiHeadedAttentionSANM instance
    ///
    /// # Arguments
    /// * `n_head` - The number of heads
    /// * `in_feat` - The audio feature size
    /// * `n_feat` - The feature size
    /// * `dropout_rate` - Dropout rate
    /// * `kernel_size` - Kernel size for FSMN
    /// * `sanm_shfit` - Shift for SANM (default: 0)
    /// * `vb` - VarBuilder for creating layers
    pub fn new(
        n_head: usize,
        in_feat: usize,
        n_feat: usize,
        dropout_rate: f32,
        kernel_size: usize,
        sanm_shfit: usize,
        vb: VarBuilder,
    ) -> Res<Self> {
        assert_eq!(n_feat % n_head, 0, "n_feat must be divisible by n_head");

        // We assume d_v always equals d_k
        let d_k = n_feat / n_head;
        let h = n_head;

        let linear_out = vb.pp("linear_out").linear(n_feat, n_feat)?;
        let linear_q_k_v = vb.pp("linear_q_k_v").linear(in_feat, n_feat * 3)?;

        // FSMN block - Conv1d with groups=n_feat and no bias
        let fsmn_block = vb.pp("fsmn_block").conv1d_no_bias_d(
            n_feat,
            n_feat,
            kernel_size,
            Conv1dConfig {
                groups: n_feat,
                stride: 1,
                padding: 0,
                ..Conv1dConfig::default()
            },
            // Metal run slowly conv1d, so we use CPU
            &Device::Cpu,
        )?;

        // Calculate padding
        let left_padding = (kernel_size - 1) / 2 + sanm_shfit;
        let right_padding = kernel_size - 1 - left_padding;

        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            d_k,
            h,
            linear_out,
            linear_q_k_v,
            fsmn_block,
            left_padding,
            right_padding,
            dropout,
        })
    }

    /// Forward pass for FSMN
    fn forward_fsmn(&self, inputs: &Tensor) -> Res<Tensor> {
        let x = inputs.transpose(1, 2)?;
        let x = x.pad_with_zeros(2, self.left_padding, self.right_padding)?;
        let x = self.fsmn_block.forward(&x.to_device(&Device::Cpu)?)?;
        let x = x.to_device(inputs.device())?;
        let x = x.transpose(1, 2)?;
        let x = (x + inputs)?;
        let x = self.dropout.forward(&x, false)?;

        Ok(x)
    }

    /// Transform query, key and value
    fn forward_qkv(&self, x: &Tensor) -> Res<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3()?;

        let q_k_v = self.linear_q_k_v.forward(x)?;
        let chunks = q_k_v.chunk(3, 2)?; // Split into 3 chunks along dim 2
        let q = &chunks[0];
        let k = &chunks[1];
        let v = &chunks[2];

        let q_h = q.reshape((b, t, self.h, self.d_k))?.transpose(1, 2)?; // (batch, head, time1, d_k)
        let k_h = k.reshape((b, t, self.h, self.d_k))?.transpose(1, 2)?; // (batch, head, time2, d_k)
        let v_h = v.reshape((b, t, self.h, self.d_k))?.transpose(1, 2)?; // (batch, head, time2, d_k)

        Ok((q_h, k_h, v_h, v.clone()))
    }

    /// Compute attention context vector
    fn forward_attention(&self, value: &Tensor, scores: &Tensor) -> Res<Tensor> {
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let p_attn = self.dropout.forward(&attn, false)?;
        let x = p_attn.matmul(&value.contiguous()?)?; // (batch, head, time1, d_k)
        let x = x.transpose(1, 2)?.flatten_from(2)?; // (batch, time1, d_model)

        let out = self.linear_out.forward(&x)?; // (batch, time1, d_model)

        Ok(out)
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Res<Tensor> {
        let (q_h, k_h, v_h, v) = self.forward_qkv(x)?;

        let fsmn_memory = self.forward_fsmn(&v)?;

        // Scale query
        let scale = (self.d_k as f32).powf(-0.5);
        let scale_tensor = Tensor::new(scale, q_h.device())?;

        let q_h = q_h.broadcast_mul(&scale_tensor)?;

        let k_h = k_h.transpose(2, 3)?;
        let k_h = k_h.contiguous()?;

        let scores = q_h.matmul(&k_h)?;

        let att_outs = self.forward_attention(&v_h, &scores)?;

        Ok((att_outs + fsmn_memory)?)
    }
}

pub struct PositionwiseFeedForward {
    w_1: Linear,
    w_2: Linear,
    dropout: Dropout,
}

impl PositionwiseFeedForward {
    /// Create a new PositionwiseFeedForward instance
    ///
    /// # Arguments
    /// * `idim` - Input dimension
    /// * `hidden_units` - Number of hidden units
    /// * `dropout_rate` - Dropout rate
    /// * `vb` - VarBuilder for creating linear layers
    pub fn new(in_dim: usize, hidden_units: usize, dropout_rate: f32, vb: VarBuilder) -> Res<Self> {
        let w_1 = vb.pp("w_1").linear(in_dim, hidden_units)?;
        let w_2 = vb.pp("w_2").linear(hidden_units, in_dim)?;
        let dropout = Dropout::new(dropout_rate);

        Ok(Self { w_1, w_2, dropout })
    }

    pub fn forward(&self, x: &Tensor) -> Res<Tensor> {
        let x = self.w_1.forward(x)?.relu()?;
        let x = self.dropout.forward(&x, false)?;
        let out = self.w_2.forward(&x)?;

        Ok(out)
    }
}

pub struct SinusoidalPositionEncoder;

impl SinusoidalPositionEncoder {
    pub fn encode(
        &self,
        positions: &Tensor,
        depth: usize,
        dtype: DType,
        device: &Device,
    ) -> Res<Tensor> {
        let batch_size = positions.shape().dim(1)?;

        // Calculate logarithmic time scale increment
        let log_timescale_increment = Tensor::new(10000.0f32, device)?
            .log()?
            .div(&Tensor::try_from((depth / 2 - 1) as f32)?.to_device(device)?)?;

        // Calculate inverse time scales
        let inv_timescales = Tensor::arange(0., (depth / 2) as f32, device)?
            .to_dtype(dtype)?
            .mul(
                &log_timescale_increment
                    .neg()?
                    .to_dtype(dtype)?
                    .broadcast_as(depth / 2)?,
            )?
            .exp()?;

        // Reshape inverse time scales to match broadcast dimensions
        let inv_timescales = inv_timescales.reshape((1, 1, depth / 2))?;

        // Calculate scaled time
        let scaled_time = positions
            .to_dtype(dtype)?
            .reshape((1, batch_size, 1))?
            .broadcast_mul(&inv_timescales)?;

        // Calculate sine and cosine encodings and concatenate
        let sin_encoding = scaled_time.sin()?;
        let cos_encoding = scaled_time.cos()?;

        let encoding = Tensor::cat(&[sin_encoding, cos_encoding], 2)?;

        Ok(encoding)
    }

    /// Forward propagation, adding positional encoding to input tensor
    pub fn forward(&self, x: &Tensor) -> Res<Tensor> {
        let shape = x.shape();
        let (_, timesteps, input_dim) = (shape.dim(0)?, shape.dim(1)?, shape.dim(2)?);
        let device = x.device();
        let dtype = x.dtype();

        // Create position tensor [1, timesteps]
        let positions =
            Tensor::arange(1i64, (timesteps + 1) as i64, device)?.reshape((1, timesteps))?;

        // Generate positional encoding
        let position_encoding = self.encode(&positions, input_dim, dtype, device)?;

        Ok((x + position_encoding)?)
    }
}

