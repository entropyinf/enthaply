use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder};

#[derive(Debug)]
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let w1 = candle_nn::linear_no_bias(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = candle_nn::linear_no_bias(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = candle_nn::linear_no_bias(dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x1 = self.w1.forward(x)?.silu()?;
        let x3 = self.w3.forward(x)?;
        self.w2.forward(&(x1 * x3)?)
    }
}
