use crate::Res;
use anyhow::Error;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, HostId, Stream, SupportedStreamConfig};
use serde::Serialize;
use std::sync::Arc;
use std::sync::mpsc::Receiver;

pub struct AudioInput {
    pub config: SupportedStreamConfig,
    stream: Arc<Stream>,
    rx: Option<Receiver<Vec<f32>>>,
}

unsafe impl Sync for AudioInput {}
unsafe impl Send for AudioInput {}

impl AudioInput {
    pub fn with_host_device(host_name: &str, device_name: &str) -> Res<Self> {
        let host = Self::host_of_name(host_name)?;
        let device = host
            .input_devices()?
            .into_iter()
            .find(|d| d.name().unwrap_or_default() == device_name)
            .ok_or_else(|| Error::msg("Device not found"))?;

        Ok(Self::new(device)?)
    }

    #[cfg(target_os = "macos")]
    pub fn from_screen_capture_kit() -> Res<Self> {
        let host = cpal::host_from_id(HostId::ScreenCaptureKit)?;
        let device = host
            .default_input_device()
            .ok_or_else(|| Error::msg("No default input device"))?;

        Ok(Self::new(device)?)
    }

    pub fn host_names() -> Vec<String> {
        Self::host_ids()
            .iter()
            .map(|h| h.name())
            .map(String::from)
            .collect::<Vec<String>>()
    }

    pub fn all_inputs() -> Res<Vec<HostDevice>> {
        let mut out = Vec::new();

        for host_id in Self::host_ids() {
            let host = Self::host_of_name(host_id.name())?;
            for device in host.input_devices()?.into_iter() {
                out.push(HostDevice::new(
                    host_id.name().to_string(),
                    device.name()?.to_string(),
                ))
            }
        }

        Ok(out)
    }

    pub fn devices_of_host(host: &Host) -> Res<Vec<Device>> {
        Ok(host.input_devices()?.into_iter().collect::<Vec<Device>>())
    }

    pub fn host_of_name(host_name: &str) -> Res<Host> {
        let host_id = Self::host_ids()
            .into_iter()
            .find(|h| h.name() == host_name)
            .ok_or_else(|| Error::msg(format!("Host not found: {}", host_name)))?;

        Ok(cpal::host_from_id(host_id)?)
    }

    pub fn new(device: Device) -> Res<Self> {
        let config = device.default_input_config()?;
        let channel_count = config.channels() as usize;
        let (tx, rx) = std::sync::mpsc::channel();
        let stream = device.build_input_stream(
            &config.config(),
            move |pcm: &[f32], _: &cpal::InputCallbackInfo| {
                let pcm = pcm
                    .iter()
                    .step_by(channel_count)
                    .copied()
                    .collect::<Vec<f32>>();

                if !pcm.is_empty() {
                    if let Err(_) = tx.send(pcm) {
                        return;
                    }
                }
            },
            move |err| {
                eprintln!("an error occurred on stream: {}", err);
            },
            None,
        )?;

        let out = AudioInput {
            config,
            stream: Arc::new(stream),
            rx: Some(rx),
        };

        Ok(out)
    }

    pub fn play(&mut self) -> Res<Receiver<Vec<f32>>> {
        self.stream.play()?;
        let rx = self.rx.take().ok_or(Error::msg("is playing"))?;
        Ok(rx)
    }

    fn host_ids() -> Vec<HostId> {
        cpal::available_hosts()
            .iter()
            .copied()
            .collect::<Vec<HostId>>()
    }
}

#[derive(Serialize)]
pub struct HostDevice {
    pub host: String,
    pub device: String,
}

impl HostDevice {
    pub fn new(host: String, device: String) -> Self {
        Self { host, device }
    }
}
