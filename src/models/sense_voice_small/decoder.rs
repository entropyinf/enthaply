use crate::Res;
use crate::var_builder::{Linear, VarBuilder};
use candle_core::Tensor;
use serde_json::Value;
use std::cmp::max;
use std::fs::File;
use std::path::Path;

#[derive(Debug)]
pub struct Token {
    pub text: String,
    pub start: u32,
    pub end: u32,
}

pub struct Decoder {
    ctc: CTCLoss,
    tokens: Value,
}

impl Decoder {
    pub fn new(tokens_file: &dyn AsRef<Path>, vb: VarBuilder) -> Res<Self> {
        let tokens_file = File::open(tokens_file)?;

        let tokens: Value = serde_json::from_reader(tokens_file)?;
        let ctc = CTCLoss::new(25055, 512, true, vb.pp("ctc"))?;
        Ok(Decoder { ctc, tokens })
    }

    pub fn decode(&self, encoder_out: &Tensor) -> Res<Vec<Token>> {
        let ctc_logits = self.ctc.log_softmax(encoder_out)?;
        let ids = ctc_logits.argmax(2)?;
        let ids = ids.flatten(0, 1)?.to_vec1::<u32>()?;

        let mut results = Vec::<Token>::new();

        let mut start = 0i32;
        let mut active = true;
        for (index, id) in ids.into_iter().enumerate() {
            let index = index as i32;
            if let Some(v) = self.tokens.get(id as usize) {
                let text = format!("{}", v.as_str().unwrap_or_default().replace("▁", " "));

                // build in
                if text.starts_with("<|") {
                    continue;
                }

                if id == 0 {
                    active = true;
                    continue;
                }

                if !active {
                    continue;
                }

                let open = max(start * 60 - 30, 0);
                let close = max(index * 60 - 30, 0);
                start = index;

                results.push(Token {
                    text,
                    start: open as u32,
                    end: close as u32,
                });
                active = false;
            }
        }

        Ok(results)
    }
}

pub struct CTCLoss {
    ctc_lo: Option<Linear>,
}

impl CTCLoss {
    /// Create a new CTC instance
    ///
    /// # Arguments
    /// * `odim` - Output dimension
    /// * `encoder_output_size` - Encoder output size
    /// * `dropout_rate` - Dropout rate (0.0 ~ 1.0)
    /// * `ctc_type` - CTC type ("builtin", etc.)
    /// * `reduce` - Whether to reduce CTC loss to scalar
    /// * `ignore_nan_grad` - Whether to ignore NaN gradients
    /// * `extra_linear` - Whether to use an extra linear layer
    pub fn new(
        odim: usize,
        encoder_output_size: usize,
        extra_linear: bool,
        vb: VarBuilder,
    ) -> Res<Self> {
        let ctc_lo = if extra_linear {
            Some(vb.pp("ctc_lo").linear(encoder_output_size, odim)?)
        } else {
            None
        };

        Ok(Self { ctc_lo })
    }

    /// Apply log_softmax to frame activations
    ///
    /// # Arguments
    /// * `hs_pad` - 3D tensor (B, Tmax, eprojs)
    /// # Returns
    /// * 3D tensor with log softmax applied (B, Tmax, odim)
    pub fn log_softmax(&self, hs_pad: &Tensor) -> candle_core::Result<Tensor> {
        if let Some(ctc_lo) = &self.ctc_lo {
            candle_nn::ops::log_softmax(&ctc_lo.forward(hs_pad)?, 2)
        } else {
            candle_nn::ops::log_softmax(hs_pad, 2)
        }
    }
}
