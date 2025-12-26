use candle_core::{DType, Device, IndexOp, Tensor};

fn precompute_freqs_cis(
    dim: &[usize],
    end: &[usize],
    theta: f64,
    device: &Device,
) -> candle_core::Result<Vec<Tensor>> {
    let mut freqs_cis = Vec::new();
    for (_, (&d, &e)) in dim.iter().zip(end.iter()).enumerate() {
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
pub struct RopeEmbedder {
    theta: f64,
    axes_dims: Vec<usize>,
    axes_lens: Vec<usize>,
    freqs_cis: Option<Vec<Tensor>>,
}

impl RopeEmbedder {
    pub fn new(theta: f64, axes_dims: Vec<usize>, axes_lens: Vec<usize>) -> Self {
        Self {
            theta,
            axes_dims,
            axes_lens,
            freqs_cis: None,
        }
    }

    pub fn call(&mut self, ids: &Tensor) -> candle_core::Result<Tensor> {
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
