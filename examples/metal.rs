use candle_core::{Device, Tensor};
use enthalpy::Res;

fn main() -> Res<()> {
    let device = Device::new_metal(0)?;
    let left = Tensor::rand(0.0f32, 1.0, (1, 4, 128, 16), &device)?;
    let right = Tensor::rand(0.0f32, 1.0, (1, 4, 16, 128), &device)?;
    let out = left.matmul(&right)?;
    println!("{}", out);

    Ok(())
}
