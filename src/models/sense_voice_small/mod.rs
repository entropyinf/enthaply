use crate::audio::resample::Resampler;
use crate::audio::silero_vad::{Segment, VadConfig, VadProcessor};
use crate::audio::{WavFrontend, WavFrontendConfig};
use crate::config::ConfigRefresher;
use crate::util::modelscope::{FileInfo, ModelScopeRepo, RepoFile};
use crate::var_builder::VarBuilder;
use crate::Res;
use anyhow::Error;
use candle_core::{Device, Module, Tensor};
use candle_nn::Embedding;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{event, Level};

mod config;
mod decoder;
mod encoder;

use crate::models::sense_voice_small::config::EncoderConfig;
use crate::models::sense_voice_small::encoder::Encoder;
pub use decoder::Token;
use crate::models::sense_voice_small::decoder::Decoder;

const EMBEDDING_DIM: usize = 560;

#[derive(Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SenseVoiceSmallConfig {
    pub model_dir: PathBuf,
    pub vad: VadConfig,
    pub resample: Option<(u32, u32)>,
    pub use_gpu: bool,
}

pub struct SenseVoiceSmall {
    device: Device,
    resampler: Option<Resampler>,
    vad: VadProcessor,
    embed: Embedding,
    frontend: WavFrontend,
    encoder: Encoder,
    decoder: Decoder,
}

impl SenseVoiceSmall {
    pub async fn with_config(cfg: SenseVoiceSmallConfig) -> Res<Self> {
        if cfg.use_gpu {
            if candle_core::utils::cuda_is_available() {
                let device = Device::new_cuda(0)?;
                return Self::new(cfg, &device).await;
            }
            if candle_core::utils::metal_is_available() {
                let device = Device::new_metal(0)?;
                return Self::new(cfg, &device).await;
            }
        }
        Self::new(cfg, &Device::Cpu).await
    }

    pub async fn new(cfg: SenseVoiceSmallConfig, device: &Device) -> Res<Self> {
        let device = device.clone();

        let repo = Self::model_repo(cfg.model_dir).await;

        let cmvn_file = repo.get("am.mvn").await?;
        let weight_file = repo.get("model.pt").await?;
        let tokens_file = repo.get("tokens.json").await?;

        let embed = Self::init_embedding()?;
        let frontend = Self::init_frontend(&cmvn_file)?;
        let (encoder, decoder) = Self::init_encoder_decoder(weight_file, tokens_file, &device)?;
        let vad = Self::init_vad(&cfg.vad)?;
        let resampler = Self::init_resampler(cfg.resample)?;

        Ok(Self {
            device,
            resampler,
            vad,
            embed,
            frontend,
            encoder,
            decoder,
        })
    }

    pub async fn get_required_files<P: Into<PathBuf>>(model_dir: P) -> Res<Vec<FileInfo>> {
        let repo = Self::model_repo(model_dir).await;

        let cmvn_file = repo.get_file_info("am.mvn").await?;
        let weight_file = repo.get_file_info("model.pt").await?;
        let tokens_file = repo.get_file_info("tokens.json").await?;

        Ok(vec![cmvn_file, weight_file, tokens_file])
    }

    pub async fn check_required_files<P: Into<PathBuf>>(model_dir: P) -> bool {
        let files = Self::get_required_files(model_dir).await;
        match files {
            Ok(files) => files.iter().all(|file| file.existed),
            Err(e) => {
                event!(Level::ERROR, "Failed to get required files: {}", e);
                false
            }
        }
    }

    pub async fn model_repo<P: Into<PathBuf>>(model_dir: P) -> ModelScopeRepo {
        let repo = ModelScopeRepo::new("iic/SenseVoiceSmall", model_dir.into());

        repo.set_repo_files(vec![
            RepoFile {
                name: "am.mvn".into(),
                path: "am.mvn".into(),
                size: 11203,
                sha256: "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"
                    .to_string(),
            },
            RepoFile {
                name: "model.pt".into(),
                path: "model.pt".into(),
                size: 936291369,
                sha256: "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
                    .to_string(),
            },
            RepoFile {
                name: "tokens.json".into(),
                path: "tokens.json".into(),
                size: 352064,
                sha256: "a2594fc1474e78973149cba8cd1f603ebed8c39c7decb470631f66e70ce58e97"
                    .to_string(),
            },
        ])
        .await;

        repo
    }

    fn init_embedding() -> Res<Embedding> {
        let lid_dict_len = 7;
        let textnorm_dict_len = 2;
        let num_embeddings = 7 + lid_dict_len + textnorm_dict_len;
        let weight =
            Tensor::randn::<_, f32>(0.0, 1.0, (num_embeddings, EMBEDDING_DIM), &Device::Cpu)?;
        let embed = Embedding::new(weight, EMBEDDING_DIM);
        Ok(embed)
    }

    fn init_encoder_decoder(
        weight_file: PathBuf,
        tokens_file: PathBuf,
        device: &Device,
    ) -> Res<(Encoder, Decoder)> {
        let vb = VarBuilder::from_file(&weight_file, &device)?;

        let encoder = Encoder::new_with_config(
            EncoderConfig {
                input_size: EMBEDDING_DIM,
                output_size: 512,
                attention_heads: 4,
                linear_units: 2048,
                num_blocks: 50,
                tp_blocks: 20,
                dropout_rate: 0.1,
                attention_dropout_rate: 0.1,
                kernel_size: 11,
                sanm_shfit: 0,
                normalize_before: true,
                concat_after: false,
            },
            vb.clone(),
        )?;

        let decoder = Decoder::new(&tokens_file, vb)?;

        Ok((encoder, decoder))
    }

    fn init_resampler(sample_config: Option<(u32, u32)>) -> Res<Option<Resampler>> {
        Ok(match sample_config {
            Some((from, to)) => Some(Resampler::new(from, to)?),
            None => None,
        })
    }

    fn init_vad(cfg: &VadConfig) -> Res<VadProcessor> {
        VadProcessor::new(cfg.clone())
    }

    fn init_frontend(cmvn_file: &PathBuf) -> Res<WavFrontend> {
        WavFrontend::new(WavFrontendConfig {
            cmvn_file: Some(cmvn_file.clone()),
            ..WavFrontendConfig::default()
        })
    }

    pub fn segment(&mut self, waveform: &mut [f32]) -> Res<Vec<Segment>> {
        let waveform = match &self.resampler {
            None => waveform,
            Some(sampler) => &mut sampler.apply_resample(&waveform)?,
        };

        self.vad.push(waveform);

        Ok(self.vad.segment())
    }

    pub fn transpose(&mut self, segments: &mut [Segment]) -> Res<Vec<Token>> {
        let mut out = Vec::with_capacity(segments.len());
        for seg in segments {
            let text = self.process(&mut seg.data)?;
            out.push(Token {
                text,
                start: seg.start,
                end: seg.end,
            });
        }

        Ok(out)
    }

    pub fn transpose_vad_cache(&mut self) -> Res<Vec<Token>> {
        let segment = self.vad.samples();
        let mut out = Vec::with_capacity(1);
        if let Some(mut seg) = segment {
            let text = self.process(&mut seg.data)?;
            out.push(Token {
                text,
                start: seg.start,
                end: seg.end,
            });
        }
        Ok(out)
    }

    fn process(&mut self, waveform: &mut [f32]) -> Res<String> {
        let mut text = String::with_capacity(1024);
        let features = self.frontend(waveform)?;
        let encoder_out = self.encoder.forward(&features)?;
        let out = self.decoder.decode(&encoder_out)?;

        for item in out.iter() {
            text += &item.text;
        }

        Ok(text)
    }

    fn frontend(&self, waveform: &mut [f32]) -> Res<Tensor> {
        let cpu = Device::Cpu;
        let speech = self
            .frontend
            .extract_features_f32(waveform)
            .map_err(|e| Error::msg(e.to_string()))?
            .to_device(&cpu)?
            .unsqueeze(0)?;

        let language_query = Tensor::new(&[[0i64]], &cpu)?;
        let language_query = self.embed.forward(&language_query)?;

        let text_norm_query = Tensor::new(&[[15i64]], &cpu)?;
        let text_norm_query = self.embed.forward(&text_norm_query)?;

        let event_emo_query = Tensor::new(&[[1i64, 2]], &cpu)?;
        let event_emo_query = self.embed.forward(&event_emo_query)?;

        let speech = Tensor::cat(&[&text_norm_query, &speech], 1)?;
        let input_query = Tensor::cat(&[&language_query, &event_emo_query], 1)?;
        let speech = Tensor::cat(&[&input_query, &speech], 1)?;

        let speech = speech.to_device(&self.device)?;

        Ok(speech)
    }
}

impl ConfigRefresher<SenseVoiceSmallConfig> for SenseVoiceSmall {
    fn refresh(&mut self, old: &SenseVoiceSmallConfig, new: &SenseVoiceSmallConfig) -> Res<()> {
        if old.vad != new.vad {
            event!(Level::DEBUG, "Refreshing VAD");
            self.vad = Self::init_vad(&new.vad)?;
        }

        if old.resample != new.resample {
            event!(Level::DEBUG, "Refreshing resampler");
            self.resampler = Self::init_resampler(new.resample)?;
        }

        Ok(())
    }
}
