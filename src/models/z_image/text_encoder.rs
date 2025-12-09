use candle_core::{Module, Tensor};
use candle_transformers::models::qwen3::{Config, Model};
use std::sync::RwLock;

pub struct Qwen3TextEncoder(RwLock<Model>);

impl Qwen3TextEncoder {
    pub fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> candle_core::Result<Self> {
        Ok(Self(RwLock::new(Model::new(&cfg, vb)?)))
    }
}

impl Module for Qwen3TextEncoder {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.0.write().unwrap().forward(input, 0)
    }
}
