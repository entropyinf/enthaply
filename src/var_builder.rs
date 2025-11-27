use crate::{Res, quantized_nn, quantized_var_builder};
use VarBuilder::{Normal, Quantiled};
use anyhow::Error;
use candle_core::{DType, Device, Module, Var};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, VarMap, init, var_builder};
use std::path::Path;

pub type Linear = Box<dyn Module + Send + Sync>;

pub enum VarBuilder<'a> {
    Normal(var_builder::VarBuilder<'a>),
    Quantiled(quantized_var_builder::VarBuilder),
}

impl<'a> VarBuilder<'a> {
    pub fn from_file<P: AsRef<Path>>(path: P, device: &Device) -> Res<VarBuilder<'a>> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .ok_or_else(|| Error::msg("No extension found"))?;

        if ext == "bin" || ext == "pt" {
            return Self::from_pt(path, device);
        }

        if ext == "gguf" {
            let vb = quantized_var_builder::VarBuilder::from_gguf(path, device)?;
            return Ok(Quantiled(vb));
        }

        Err(Error::msg("Unsupported file extension"))
    }

    fn from_pt(path: &Path, device: &Device) -> Res<Self> {
        let tensors = candle_core::pickle::read_all(path)?;
        let vm = VarMap::new();

        {
            let mut vm_data_map = vm.data().lock().map_err(|e| Error::msg(e.to_string()))?;
            for (name, tensor) in tensors.iter() {
                vm_data_map.insert(
                    String::from(name),
                    Var::from_tensor(&tensor.to_device(device)?)?,
                );
            }
        }
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, device);

        Ok(Normal(vb))
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        match self {
            Normal(vb) => Normal(vb.pp(s)),
            Quantiled(vb) => Quantiled(vb.pp(s)),
        }
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        match self {
            Normal(vb) => vb.contains_tensor(name),
            Quantiled(vb) => vb.contains_tensor(name),
        }
    }
    pub fn linear(self, in_dim: usize, out_dim: usize) -> Res<Linear> {
        match self {
            Normal(vb) => Ok(Box::new(candle_nn::linear(in_dim, out_dim, vb)?)),
            Quantiled(vb) => Ok(Box::new(quantized_nn::linear(in_dim, out_dim, vb)?)),
        }
    }

    pub fn layer_norm<C: Into<LayerNormConfig>>(self, size: usize, config: C) -> Res<LayerNorm> {
        let out = match self {
            Normal(vb) => candle_nn::layer_norm(size, config.into(), vb)?,
            Quantiled(vb) => quantized_nn::layer_norm(size, config.into().eps, vb)?,
        };

        Ok(out)
    }

    pub fn conv1d_no_bias(
        self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        cfg: Conv1dConfig,
    ) -> Res<Conv1d> {
        let device = match &self {
            Normal(vb) => vb.device(),
            Quantiled(vb) => vb.device(),
        }
        .clone();

        self.conv1d_no_bias_d(in_channels, out_channels, kernel_size, cfg, &device)
    }

    pub fn conv1d_no_bias_d(
        self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        cfg: Conv1dConfig,
        device: &Device,
    ) -> Res<Conv1d> {
        let out = match self {
            Normal(vb) => {
                let init_ws = init::DEFAULT_KAIMING_NORMAL;
                let ws = vb
                    .get_with_hints(
                        (out_channels, in_channels / cfg.groups, kernel_size),
                        "weight",
                        init_ws,
                    )?
                    .to_device(device)?;
                Conv1d::new(ws, None, cfg)
            }
            Quantiled(vb) => {
                let weight = vb
                    .get(
                        (out_channels, in_channels / cfg.groups, kernel_size),
                        "weight",
                    )?
                    .dequantize(device)?;

                Conv1d::new(weight, None, cfg)
            }
        };

        Ok(out)
    }

    pub fn device(&self) -> &Device {
        match self {
            Normal(vb) => vb.device(),
            Quantiled(vb) => vb.device(),
        }
    }
}

impl Clone for VarBuilder<'_> {
    fn clone(&self) -> Self {
        match self {
            Normal(vb) => Normal(vb.clone()),
            Quantiled(vb) => Quantiled(vb.clone()),
        }
    }
}
