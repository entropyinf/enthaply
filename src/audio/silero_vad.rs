use crate::Res;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use voice_activity_detector::{IteratorExt, VoiceActivityDetector};

/// Number of samples processed in each chunk
const CHUNK_SIZE: usize = 512;

/// Configuration parameters for Voice Activity Detection
#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    /// Sample rate in Hz (e.g., 16000)
    pub sample_rate: u32,
    /// Threshold for speech detection, values above this are considered voice activity
    pub speech_threshold: f32,
    /// Maximum silence duration in milliseconds, exceeding this ends a speech segment
    pub silence_max_ms: f32,
    /// Minimum speech duration in milliseconds, segments shorter than this are ignored
    pub speech_min_ms: f32,
    /// Average speech duration in milliseconds, used to dynamically adjust silence detection parameters
    pub speech_avg_ms: f32,
    /// Factor to adjust silence detection sensitivity after long speech segments
    pub silence_attenuation_factor: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            speech_threshold: 0.5,
            silence_max_ms: 400.0,
            speech_min_ms: 2000.0,
            speech_avg_ms: 5000.0,
            silence_attenuation_factor: 0.707,
        }
    }
}

/// Processes audio data to detect voice activity and segment speech
pub struct VadProcessor {
    /// Voice activity detector instance
    vad: VoiceActivityDetector,
    /// VAD configuration parameters
    config: VadConfig,
    /// Total number of processed audio chunks
    chunk_total: f32,
    /// Time length of each audio chunk in milliseconds
    chunk_ms: f32,
    /// Maximum number of silence chunks allowed
    silence_max_count: f32,
    /// Minimum number of speech chunks required
    speech_min_count: f32,
    /// Average number of speech chunks for parameter adjustment
    speech_avg_count: f32,
    /// Buffer for storing incoming audio data
    buffer: VecDeque<f32>,
    /// Stores currently detected speech samples
    samples: VecDeque<f32>,
    /// Current processing state (silence or speech)
    status: Status,
}

impl VadProcessor {
    pub fn new(config: VadConfig) -> Res<Self> {
        let vad = VoiceActivityDetector::builder()
            .sample_rate(config.sample_rate)
            .chunk_size(CHUNK_SIZE)
            .build()?;

        let chunk_ms = (CHUNK_SIZE as f32 / config.sample_rate as f32) * 1000.0;

        Ok(Self {
            vad,
            chunk_total: 0.0,
            chunk_ms,
            silence_max_count: config.silence_max_ms / chunk_ms,
            speech_min_count: config.speech_min_ms / chunk_ms,
            speech_avg_count: config.speech_avg_ms / chunk_ms,
            buffer: VecDeque::with_capacity(CHUNK_SIZE * 2),
            samples: VecDeque::with_capacity(CHUNK_SIZE * 1024),
            status: Status::Silence,
            config,
        })
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.buffer.extend(samples);
    }
    pub fn process(&mut self, samples: &[f32]) -> Vec<Segment> {
        self.push(samples);
        self.segment()
    }

    pub fn segment(&mut self) -> Vec<Segment> {
        let len = self.buffer.len();
        let chunk_count = len / CHUNK_SIZE;
        if chunk_count == 0 {
            return Vec::with_capacity(0);
        }
        let chunk_size = chunk_count * CHUNK_SIZE;

        let mut segments = Vec::new();

        // VAD
        for (data, pred) in self.buffer.drain(0..chunk_size).predict(&mut self.vad) {
            let config = &self.config;
            let speech = pred > config.speech_threshold;

            match self.status {
                Status::Silence => {
                    if speech {
                        self.status = Status::Speech {
                            start: (self.chunk_total * self.chunk_ms) as u32,
                            speech_count: 1,
                            silence_count: 0,
                        };
                        self.samples.extend(data);
                    } else {
                        let len = self.samples.len();
                        if len < (self.silence_max_count * CHUNK_SIZE as f32) as usize {
                            self.samples.extend(data);
                        };
                    }
                }
                Status::Speech {
                    start,
                    speech_count,
                    silence_count,
                } => {
                    self.samples.extend(data);

                    self.status = match speech {
                        true => Status::Speech {
                            start,
                            speech_count: speech_count + 1,
                            silence_count: 0,
                        },
                        false => Status::Speech {
                            start,
                            speech_count: speech_count + 1,
                            silence_count: silence_count + 1,
                        },
                    };

                    if speech_count as f32 >= self.speech_min_count {
                        let mut silence_max_count = self.silence_max_count;

                        if speech_count as f32 > self.speech_avg_count {
                            let rate = speech_count as f32 / self.speech_avg_count - 1.0;
                            let factor = (-rate * config.silence_attenuation_factor).exp();
                            silence_max_count = silence_max_count * factor;
                        }

                        if silence_count as f32 >= silence_max_count {
                            self.status = Status::Silence;
                            let len = self.samples.len();
                            let end_idx =
                                len - (silence_max_count * CHUNK_SIZE as f32 / 2.0) as usize;
                            segments.push(Segment {
                                start,
                                end: start + speech_count * self.chunk_ms as u32,
                                data: self.samples.drain(0..end_idx).collect(),
                            })
                        }
                    }
                }
            }

            self.chunk_total += 1.0;
        }

        segments
    }

    pub fn samples(&self) -> Option<Segment> {
        if let Status::Speech {
            start,
            speech_count,
            ..
        } = self.status
        {
            let data = self.samples.iter().map(|x| *x).collect::<Vec<f32>>();
            Some(Segment {
                start,
                end: start + speech_count * self.chunk_ms as u32,
                data,
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
enum Status {
    Silence,
    Speech {
        start: u32,
        speech_count: u32,
        silence_count: u32,
    },
}

pub struct Segment {
    pub start: u32,
    pub end: u32,
    pub data: Vec<f32>,
}
