use crate::Res;
use anyhow::Error;
use candle_core::{Device, Tensor};
use std::f64::consts::PI;

pub struct Resampler {
    orig_freq: i32,
    new_freq: i32,
    gcd: i32,
    kernel: Tensor,
    width: i32,
}

impl Resampler {
    pub fn new(orig_freq: u32, new_freq: u32) -> Res<Self> {
        let (orig_freq, new_freq) = (orig_freq as i32, new_freq as i32);
        let (kernels, width) =
            get_sinc_resample_kernel(orig_freq, new_freq, 16000, 6, 0.99f32, None)?;

        Ok(Self {
            orig_freq,
            new_freq,
            gcd: 16000,
            kernel: kernels,
            width,
        })
    }

    pub fn apply_resample(&self, waveform: &[f32]) -> Res<Vec<f32>> {
        let out = apply_sinc_resample_kernel(
            waveform,
            self.orig_freq,
            self.new_freq,
            self.gcd,
            &self.kernel,
            self.width,
        )?;
        Ok(out)
    }
}

/// Get sinc resampling kernel
///
/// # Arguments
/// * `orig_freq` - Original sampling rate
/// * `new_freq` - New sampling rate
/// * `gcd` - Greatest common divisor
/// * `lowpass_filter_width` - Low-pass filter width
/// * `rolloff` - Roll-off factor
/// * `resampling_method` - Resampling method ("sinc_interp_hann" or "sinc_interp_kaiser")
/// * `beta` - Beta parameter for Kaiser window
///
/// # Returns
/// * Kernel tensor and width
fn get_sinc_resample_kernel(
    orig_freq: i32,
    new_freq: i32,
    gcd: i32,
    lowpass_filter_width: i32,
    rolloff: f32,
    _beta: Option<f64>,
) -> Res<(Tensor, i32)> {
    if orig_freq as f32 != orig_freq as f32 || new_freq as f32 != new_freq as f32 {
        return Err(Error::msg(
            r#"Frequencies must be of integer type to ensure quality resampling computation.
             To work around this, manually convert both frequencies to integer values
             that maintain their resampling rate ratio before passing them into the function."#,
        ));
    }

    let orig_freq = orig_freq / gcd;
    let new_freq = new_freq / gcd;

    if lowpass_filter_width <= 0 {
        return Err(Error::msg("Low pass filter width should be positive."));
    }

    let base_freq = (orig_freq.min(new_freq)) as f32;
    let base_freq = base_freq * rolloff;

    let width = ((lowpass_filter_width as f32) * (orig_freq as f32) / base_freq).ceil() as i32;

    let idx_len = (2 * width + orig_freq) as usize;
    let mut idx = Vec::with_capacity(idx_len);
    for i in -width..=width + orig_freq - 1 {
        idx.push(i as f32 / orig_freq as f32);
    }

    let t_rows = new_freq as usize;
    let mut t = vec![vec![0.0; idx_len]; t_rows];

    for j in 0..new_freq {
        for (i, &idx_val) in idx.iter().enumerate() {
            let val = -(j as f32) / new_freq as f32 + idx_val;
            t[j as usize][i] = val * base_freq;

            // Limit range
            if t[j as usize][i] > lowpass_filter_width as f32 {
                t[j as usize][i] = lowpass_filter_width as f32;
            } else if t[j as usize][i] < -(lowpass_filter_width as f32) {
                t[j as usize][i] = -(lowpass_filter_width as f32);
            }
        }
    }
    let pi = PI as f32;

    let mut window = vec![vec![0.0 as f32; idx_len]; t_rows];
    for j in 0..t_rows {
        for i in 0..idx_len {
            let cos_val = ((t[j][i] * pi / lowpass_filter_width as f32) / 2.0).cos();
            window[j][i] = cos_val * cos_val;
        }
    }

    let mut kernels = vec![vec![0.0 as f32; idx_len]; t_rows];
    let scale = base_freq / orig_freq as f32;

    for j in 0..t_rows {
        for i in 0..idx_len {
            t[j][i] *= pi;

            let sinc_val = if t[j][i] == 0.0 {
                1.0
            } else {
                t[j][i].sin() / t[j][i]
            };

            kernels[j][i] = sinc_val * window[j][i] * scale;
        }
    }

    let kernels = Tensor::from_vec(
        kernels.into_iter().flatten().collect::<Vec<f32>>(),
        (t_rows, idx_len),
        &Device::Cpu,
    )?;

    Ok((kernels, width))
}

/// Apply sinc resampling kernel
///
/// # Arguments
/// * `waveform` - Input waveform data
/// * `orig_freq` - Original sampling rate
/// * `new_freq` - New sampling rate
/// * `gcd` - Greatest common divisor
/// * `kernel` - Resampling kernel
/// * `width` - Kernel width
///
/// # Returns
/// * Resampled waveform data
fn apply_sinc_resample_kernel(
    waveform: &[f32],
    orig_freq: i32,
    new_freq: i32,
    gcd: i32,
    kernel: &Tensor,
    width: i32,
) -> Res<Vec<f32>> {
    let orig_freq = orig_freq / gcd;
    let new_freq = new_freq / gcd;

    let length = waveform.len();
    let padded_length = length + (2 * width + orig_freq) as usize;

    let mut padded_waveform = vec![0.0; padded_length];
    for i in 0..width as usize {
        padded_waveform[i] = waveform[0]; // Left padding
    }
    padded_waveform[width as usize..width as usize + length].copy_from_slice(waveform);
    for i in 0..(width + orig_freq) as usize {
        if width as usize + length + i < padded_length {
            padded_waveform[width as usize + length + i] = *waveform.last().unwrap_or(&0.0); // Right padding
        }
    }

    let padded_tensor = Tensor::from_vec(padded_waveform, padded_length, &Device::Cpu)?;

    let kernel_stride = orig_freq as usize;
    let output_length = ((new_freq as f32) * (length as f32) / (orig_freq as f32)).ceil() as usize;

    let mut resampled = Vec::with_capacity(output_length);

    for j in 0..output_length {
        let start_idx = j * kernel_stride;
        let end_idx = (start_idx + kernel.dims()[1]).min(padded_length);

        if start_idx < padded_length {
            let waveform_slice = padded_tensor.narrow(0, start_idx, end_idx - start_idx)?;

            let kernel_row = kernel.get(j % kernel.dims()[0])?;

            let result = waveform_slice
                .broadcast_mul(&kernel_row)?
                .sum_all()?
                .to_scalar::<f32>()?;
            resampled.push(result);
        } else {
            resampled.push(0.0);
        }
    }

    Ok(resampled)
}