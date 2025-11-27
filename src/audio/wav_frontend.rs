use std::ffi::CStr;
use std::fs::File;
use std::ops::Deref;
use std::path::PathBuf;
use candle_core::{Device, Tensor};
use kaldi_fbank_rust_kautism::{FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank};
use crate::Res;

/// Configuration for the `WavFrontend` audio feature extraction system.
///
/// This structure defines parameters for processing waveforms into mel-frequency features.
pub struct WavFrontendConfig {
    /// Sample rate of the audio in Hz (e.g., 16000).
    pub sample_rate: i32,
    /// Length of each frame in milliseconds.
    pub frame_length_ms: f32,
    /// Shift between consecutive frames in milliseconds.
    pub frame_shift_ms: f32,
    /// Number of mel filter banks.
    pub n_mels: usize,
    /// Number of frames to stack for low frame rate (LFR) processing.
    pub lfr_m: usize,
    /// Frame interval for LFR processing.
    pub lfr_n: usize,
    /// Optional path to the CMVN (cepstral mean and variance normalization) file.
    pub cmvn_file: Option<PathBuf>,
}

/// Implementation of the `Default` trait for `WavFrontendConfig`.
impl Default for WavFrontendConfig {
    /// Creates a `WavFrontendConfig` instance with default values.
    ///
    /// # Returns
    ///
    /// A `WavFrontendConfig` instance with commonly used default settings.
    fn default() -> Self {
        WavFrontendConfig {
            sample_rate: 16000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            n_mels: 80,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_file: None,
        }
    }
}

/// Audio feature extraction audio for processing waveforms.
///
/// This structure handles the extraction of mel-frequency features from audio data, optionally applying LFR and CMVN.
pub struct WavFrontend {
    /// Configuration settings for feature extraction.
    config: WavFrontendConfig,
    /// Optional array of mean values for CMVN.
    cmvn_means: Option<Tensor>,
    /// Optional array of variance values for CMVN.
    cmvn_vars: Option<Tensor>,
}

/// Implementation of methods for `WavFrontend`.
impl WavFrontend {
    /// Creates a new `WavFrontend` instance with the specified configuration.
    ///
    /// If a CMVN file is provided in the config, it loads the mean and variance values for normalization.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration settings for the audio.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `WavFrontend` instance or an error if CMVN loading fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the CMVN file cannot be opened or parsed.
    ///
    /// # Example
    ///
    /// ```
    /// use wavfrontend::{WavFrontend, WavFrontendConfig};
    ///
    /// let config = WavFrontendConfig::default();
    /// let audio = WavFrontend::new(config).expect("Failed to initialize WavFrontend");
    /// ```
    pub fn new(config: WavFrontendConfig) -> Res<Self> {
        let (cmvn_means, cmvn_vars) = if let Some(cmvn_path) = &config.cmvn_file {
            let (means, vars) = Self::load_cmvn(cmvn_path)?;
            (Some(means), Some(vars))
        } else {
            (None, None)
        };
        Ok(WavFrontend {
            config,
            cmvn_means,
            cmvn_vars,
        })
    }

    pub fn extract_features_f32(&self, waveform: &mut [f32]) -> Res<Tensor> {
        let fbank = self.compute_fbank_features(waveform)?;
        let lfr_feats = self.apply_lfr(&fbank, self.config.lfr_m, self.config.lfr_n)?;
        let feats = self.apply_cmvn(&lfr_feats)?;
        Ok(feats)
    }

    /// Computes mel-frequency filterbank (fbank) features from a waveform.
    ///
    /// Uses the `kaldi_fbank_rust` library to extract fbank features based on the configured parameters.
    ///
    /// # Arguments
    ///
    /// * `waveform` - Slice of audio samples as 32-bit floats.
    ///
    /// # Returns
    ///
    /// A `Result` containing a 2D array of fbank features with shape `(frames, n_mels)`.
    ///
    /// # Errors
    ///
    /// Returns an error if feature computation fails or if the output array cannot be constructed.
    pub fn compute_fbank_features(&self, waveform: &mut [f32]) -> Res<Tensor> {
        let opt = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: self.config.sample_rate as f32,
                window_type: CStr::from_bytes_with_nul(b"hamming\0").unwrap().as_ptr(),
                dither: 1.0,
                frame_shift_ms: self.config.frame_shift_ms,
                frame_length_ms: self.config.frame_length_ms,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: self.config.n_mels as i32,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };

        let mut fbank = OnlineFbank::new(opt);

        let scale = (1 << 15) as f32;
        waveform.iter_mut().for_each(|x| *x *= scale);

        fbank.accept_waveform(self.config.sample_rate as f32, waveform);

        let frames = fbank.num_ready_frames();

        let mut fbank_feats = Vec::with_capacity(frames as usize);
        for i in 0..frames {
            let frame = fbank.get_frame(i).expect("Should have frame");
            fbank_feats.push(frame.to_vec());
        }

        let fbank_flat: Vec<f32> = fbank_feats.into_iter().flatten().collect();
        let fbank_tensor = Tensor::from_vec(fbank_flat, (frames as usize, self.config.n_mels), &Device::Cpu)?;

        Ok(fbank_tensor)
    }

    /// Loads CMVN statistics from a file.
    ///
    /// Parses a CMVN file to extract mean and variance vectors for normalization.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CMVN file.
    ///
    /// # Returns
    ///
    /// A `Result` containing a tuple of mean and variance arrays (`Array1<f32>`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or if the CMVN data is malformed.
    fn load_cmvn(path: &PathBuf) -> Res<(Tensor, Tensor)> {
        let file = File::open(path.deref())?;
        let reader = std::io::BufReader::new(file);
        let mut lines = std::io::BufRead::lines(reader);

        let mut means = Vec::new();
        let mut vars = Vec::new();
        let mut is_means = false;
        let mut is_vars = false;

        while let Some(line) = lines.next() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts[0] == "<AddShift>" {
                is_means = true;
                continue;
            } else if parts[0] == "<Rescale>" {
                is_vars = true;
                continue;
            } else if parts[0] == "<LearnRateCoef>" && is_means {
                means = parts[3..parts.len() - 1]
                    .iter()
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect();
                is_means = false;
            } else if parts[0] == "<LearnRateCoef>" && is_vars {
                vars = parts[3..parts.len() - 1]
                    .iter()
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect();
                is_vars = false;
            }
        }

        let means_len = means.len();
        let vars_len = vars.len();

        let means_tensor = Tensor::from_vec(means, means_len, &Device::Cpu)?;
        let vars_tensor = Tensor::from_vec(vars, vars_len, &Device::Cpu)?;

        Ok((means_tensor, vars_tensor))
    }

    /// Applies low frame rate (LFR) processing to fbank features.
    ///
    /// Stacks and subsamples frames to reduce the frame rate, padding as necessary.
    ///
    /// # Arguments
    ///
    /// * `fbank` - 2D array of fbank features.
    /// * `lfr_m` - Number of frames to stack.
    /// * `lfr_n` - Frame interval for subsampling.
    ///
    /// # Returns
    ///
    /// A 2D array of LFR-processed features with shape `(t_lfr, n_mels * lfr_m)`.
    fn apply_lfr(&self, fbank: &Tensor, lfr_m: usize, lfr_n: usize) -> Res<Tensor> {
        let shape = fbank.dims();
        let t = shape[0];
        let t_lfr = ((t as f32) / lfr_n as f32).ceil() as usize;
        let left_padding_rows = (lfr_m - 1) / 2;

        // Create padded fbank
        let mut padded_data = Vec::new();
        let fbank_data = fbank.to_vec2::<f32>()?;

        // Add left padding rows (copy first row)
        for _ in 0..left_padding_rows {
            padded_data.extend_from_slice(&fbank_data[0]);
        }

        // Add original data
        for row in fbank_data.iter() {
            padded_data.extend_from_slice(row);
        }

        let padded_fbank = Tensor::from_vec(padded_data, (t + left_padding_rows, shape[1]), &Device::Cpu)?;

        let feat_dim = self.config.n_mels * lfr_m;
        let mut lfr_data = Vec::with_capacity(t_lfr * feat_dim);

        for i in 0..t_lfr {
            let start = i * lfr_n;
            let end = if lfr_m <= t + left_padding_rows - start {
                start + lfr_m
            } else {
                t + left_padding_rows
            };

            // Extract frame and flatten
            let frame_slice = padded_fbank.narrow(0, start, end - start)?;
            let frame_data = frame_slice.flatten_all()?.to_vec1::<f32>()?;
            lfr_data.extend_from_slice(&frame_data);

            // Handle case where we need to pad with last row
            if end < start + lfr_m {
                let last_row = padded_fbank.narrow(0, t + left_padding_rows - 1, 1)?;
                let last_row_data = last_row.squeeze(0)?.to_vec1::<f32>()?;
                for _ in end - start..lfr_m {
                    lfr_data.extend_from_slice(&last_row_data);
                }
            }
        }

        let lfr_tensor = Tensor::from_vec(lfr_data, (t_lfr, feat_dim), &Device::Cpu)?;

        Ok(lfr_tensor)
    }

    /// Applies cepstral mean and variance normalization (CMVN) to features.
    ///
    /// Normalizes the features using precomputed mean and variance values if available.
    ///
    /// # Arguments
    ///
    /// * `feats` - 2D array of features to normalize.
    ///
    /// # Returns
    ///
    /// A 2D array of normalized features, or the original features if no CMVN data is provided.
    fn apply_cmvn(&self, feats: &Tensor) -> Res<Tensor> {
        if let (Some(means), Some(vars)) = (&self.cmvn_means, &self.cmvn_vars) {
            let feats_broadcast = feats.broadcast_add(means)?;
            let normalized = feats_broadcast.broadcast_mul(vars)?;
            Ok(normalized)
        } else {
            Ok(feats.clone())
        }
    }
}