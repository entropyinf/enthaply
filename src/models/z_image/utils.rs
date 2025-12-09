//! Utility functions for Z-Image

use candle_core::{Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Calculate shift value based on image sequence length
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

/// Create random latents tensor
pub fn rand_latents(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0f32; shape.iter().product::<usize>()];
    for elem in data.iter_mut() {
        *elem = rng.random::<f32>();
    }

    Tensor::from_vec(data, shape, device)
}
