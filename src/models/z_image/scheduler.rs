//! FlowMatchEulerDiscreteScheduler implementation for Z-Image

use candle_core::{Device, Result, Tensor};
use std::f64::consts::E as EXP;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub use_dynamic_shifting: bool,
}

impl SchedulerConfig {
    pub fn new(num_train_timesteps: usize, shift: f64, use_dynamic_shifting: bool) -> Self {
        Self {
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: false,
        }
    }
}

#[derive(Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub config: SchedulerConfig,
    pub timesteps: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    step_index: Option<usize>,
    begin_index: Option<usize>,
    pub num_inference_steps: usize,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(num_train_timesteps: usize, shift: f64, use_dynamic_shifting: bool) -> Self {
        let config = SchedulerConfig::new(num_train_timesteps, shift, use_dynamic_shifting);

        // Create timesteps array (reverse linspace from 1 to num_train_timesteps)
        let mut timesteps: Vec<f64> = (0..num_train_timesteps)
            .map(|i| (num_train_timesteps - i) as f64)
            .collect();

        let mut sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| t / num_train_timesteps as f64)
            .collect();

        if !use_dynamic_shifting {
            sigmas = sigmas
                .iter()
                .map(|&s| shift * s / (1.0 + (shift - 1.0) * s))
                .collect();
        }

        timesteps = sigmas
            .iter()
            .map(|&s| s * num_train_timesteps as f64)
            .collect();

        let sigma_min = *sigmas.last().unwrap_or(&0.0);
        let sigma_max = *sigmas.first().unwrap_or(&1.0);

        Self {
            config,
            timesteps,
            sigmas,
            sigma_min,
            sigma_max,
            step_index: None,
            begin_index: None,
            num_inference_steps: 0,
        }
    }

    pub fn set_timesteps(
        &mut self,
        num_inference_steps: usize,
        device: &Device, // Just for compatibility with the Python API
        mu: Option<f64>,
    ) -> Result<()> {
        self.num_inference_steps = num_inference_steps;

        // Create timesteps (linspace from sigma_max to sigma_min)
        let t_start = self._sigma_to_t(self.sigma_max);
        let t_end = self._sigma_to_t(self.sigma_min);

        // Generate evenly spaced values
        let step_size = (t_start - t_end) / num_inference_steps as f64;
        let timesteps: Vec<f64> = (0..num_inference_steps)
            .map(|i| t_start - i as f64 * step_size)
            .collect();

        let mut sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| t / self.config.num_train_timesteps as f64)
            .collect();

        if self.config.use_dynamic_shifting {
            if let Some(mu_val) = mu {
                sigmas = sigmas
                    .iter()
                    .map(|&t| self.time_shift(mu_val, 1.0, t))
                    .collect();
            }
        } else {
            sigmas = sigmas
                .iter()
                .map(|&s| self.config.shift * s / (1.0 + (self.config.shift - 1.0) * s))
                .collect();
        }

        // Add zero sigma at the end
        sigmas.push(0.0);

        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.step_index = None;
        self.begin_index = None;

        Ok(())
    }

    fn _sigma_to_t(&self, sigma: f64) -> f64 {
        sigma * self.config.num_train_timesteps as f64
    }

    fn time_shift(&self, mu: f64, sigma: f64, t: f64) -> f64 {
        EXP.powf(mu) / (EXP.powf(mu) + (1.0 / t - 1.0).powf(sigma))
    }

    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep: f64,
        sample: &Tensor,
    ) -> Result<Tensor> {
        if self.step_index.is_none() {
            self._init_step_index(timestep);
        }

        let sigma_idx = self.step_index.unwrap();
        let sigma = self.sigmas[sigma_idx];
        let sigma_next = self.sigmas[sigma_idx + 1];

        let dt = sigma_next - sigma;

        // prev_sample = sample + dt * model_output
        let dt_tensor = Tensor::full(dt as f32, model_output.dims(), model_output.device())?;
        let dt_model_output = model_output.broadcast_mul(&dt_tensor)?;
        let prev_sample = sample.broadcast_add(&dt_model_output)?;

        self.step_index = Some(sigma_idx + 1);

        Ok(prev_sample)
    }

    fn _init_step_index(&mut self, timestep: f64) {
        if self.begin_index.is_none() {
            self.step_index = Some(self.index_for_timestep(timestep));
        } else {
            self.step_index = self.begin_index;
        }
    }

    fn index_for_timestep(&self, timestep: f64) -> usize {
        // Find the index of the timestep in self.timesteps
        self.timesteps
            .iter()
            .position(|&t| (t - timestep).abs() < 1e-6)
            .unwrap_or(0)
    }
}
