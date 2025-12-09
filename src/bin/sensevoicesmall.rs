use anyhow::anyhow;
use clap::{Parser, Subcommand};
use enthalpy::Res;
use enthalpy::audio::input::AudioInput;
use enthalpy::audio::load_audio;
use enthalpy::audio::silero_vad::VadConfig;
use enthalpy::models::sense_voice_small::{SenseVoiceSmall, SenseVoiceSmallConfig};
use std::path::PathBuf;
use tokio::time::Instant;
use tracing::Level;

#[derive(Parser)]
#[clap(
    name = "sensevoicesmall",
    about = "Speech recognition tool",
    version = "0.1.0"
)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe an audio file
    Transcribe(TranscribeArgs),
    /// Stream real-time audio transcription
    Stream(StreamArgs),
    /// List available audio input devices
    ListDevices,
}

#[derive(Parser)]
struct TranscribeArgs {
    /// Input audio file path
    #[clap(short, long, value_parser)]
    input: String,

    /// Model directory path
    #[clap(
        short,
        long,
        value_parser,
        default_value = "/Users/entropy/.cache/modelscope/hub/models/"
    )]
    model_dir: String,

    /// Enable GPU acceleration
    #[clap(long, action)]
    gpu: bool,

    /// VAD sample rate in Hz
    #[clap(long, default_value_t = 16000)]
    vad_sample_rate: u32,

    /// VAD speech threshold
    #[clap(long, default_value_t = 0.5)]
    vad_speech_threshold: f32,

    /// VAD maximum silence duration in milliseconds
    #[clap(long, default_value_t = 400.0)]
    vad_silence_max_ms: f32,

    /// VAD minimum speech duration in milliseconds
    #[clap(long, default_value_t = 2000.0)]
    vad_speech_min_ms: f32,

    /// VAD average speech duration in milliseconds
    #[clap(long, default_value_t = 5000.0)]
    vad_speech_avg_ms: f32,

    /// VAD silence attenuation factor
    #[clap(long, default_value_t = 0.707)]
    vad_silence_attenuation_factor: f32,
}

#[derive(Parser)]
struct StreamArgs {
    /// Model directory path
    #[clap(
        short,
        long,
        value_parser,
        default_value = "/Users/entropy/.cache/modelscope/hub/models/"
    )]
    model_dir: String,

    /// Enable GPU acceleration
    #[clap(long, action)]
    gpu: bool,

    /// Audio sample rate (only used in stream mode)
    #[clap(long, default_value_t = 16000)]
    sample_rate: u32,

    /// VAD sample rate in Hz
    #[clap(long, default_value_t = 16000)]
    vad_sample_rate: u32,

    /// VAD speech threshold
    #[clap(long, default_value_t = 0.5)]
    vad_speech_threshold: f32,

    /// VAD maximum silence duration in milliseconds
    #[clap(long, default_value_t = 400.0)]
    vad_silence_max_ms: f32,

    /// VAD minimum speech duration in milliseconds
    #[clap(long, default_value_t = 2000.0)]
    vad_speech_min_ms: f32,

    /// VAD average speech duration in milliseconds
    #[clap(long, default_value_t = 5000.0)]
    vad_speech_avg_ms: f32,

    /// VAD silence attenuation factor
    #[clap(long, default_value_t = 0.707)]
    vad_silence_attenuation_factor: f32,

    /// Audio host name (used in stream mode)
    #[clap(long, value_parser)]
    host: Option<String>,

    /// Audio device name (used in stream mode)
    #[clap(long, value_parser)]
    device: Option<String>,
}

#[tokio::main]
async fn main() -> Res<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .compact()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Transcribe(args) => {
            transpose_file(&args).await?;
        }
        Commands::Stream(args) => {
            transpose_stream(&args).await?;
        }
        Commands::ListDevices => {
            list_audio_devices()?;
        }
    }

    Ok(())
}

fn list_audio_devices() -> Res<()> {
    let inputs = AudioInput::all_inputs()?;
    if inputs.is_empty() {
        println!("No audio input devices found.");
        return Ok(());
    }

    println!("Available audio input devices:");
    println!("{:<20} {}", "Host", "Device");
    println!("{}", "-".repeat(50));
    for input in inputs {
        println!("{:<20} {}", input.host, input.device);
    }
    Ok(())
}

async fn transpose_file(args: &TranscribeArgs) -> Res<()> {
    let (mut data, sample_rate) = load_audio(&args.input)?;

    let vad_config = VadConfig {
        sample_rate: args.vad_sample_rate,
        speech_threshold: args.vad_speech_threshold,
        silence_max_ms: args.vad_silence_max_ms,
        speech_min_ms: args.vad_speech_min_ms,
        speech_avg_ms: args.vad_speech_avg_ms,
        silence_attenuation_factor: args.vad_silence_attenuation_factor,
    };

    let cfg = SenseVoiceSmallConfig {
        model_dir: PathBuf::from(&args.model_dir),
        vad: vad_config,
        resample: Some((sample_rate, 16000)),
        use_gpu: args.gpu,
    };

    let mut model = SenseVoiceSmall::with_config(cfg).await?;

    let start = Instant::now();
    let mut segments = model.segment(&mut data)?;
    let tokens = model.transpose(&mut segments)?;
    println!("Processing time: {:.2}s", start.elapsed().as_secs_f32());

    for token in tokens {
        println!(
            "[{:.1}s - {:.1}s]: {}",
            token.start as f32 / 1000.0,
            token.end as f32 / 1000.0,
            token.text
        );
    }

    Ok(())
}

async fn transpose_stream(args: &StreamArgs) -> Res<()> {
    let mut input = match (&args.host, &args.device) {
        (Some(host), Some(device)) => AudioInput::with_host_device(host, device)?,
        _ => {
            return Err(anyhow!("Please specify host and device for stream mode"));
        }
    };

    let sample_rate = input.config.sample_rate().0;

    let vad_config = VadConfig {
        sample_rate: args.vad_sample_rate,
        speech_threshold: args.vad_speech_threshold,
        silence_max_ms: args.vad_silence_max_ms,
        speech_min_ms: args.vad_speech_min_ms,
        speech_avg_ms: args.vad_speech_avg_ms,
        silence_attenuation_factor: args.vad_silence_attenuation_factor,
    };

    let cfg = SenseVoiceSmallConfig {
        model_dir: PathBuf::from(&args.model_dir),
        vad: vad_config,
        resample: Some((sample_rate, 16000)),
        use_gpu: args.gpu,
    };

    let mut model = SenseVoiceSmall::with_config(cfg).await?;

    println!("Starting real-time audio transcription...");

    let pcm_data = input.play()?;
    while let Ok(mut chunk) = pcm_data.recv() {
        let start = Instant::now();
        let mut segments = model.segment(&mut chunk)?;
        let tokens = model.transpose(&mut segments)?;

        for token in tokens {
            println!(
                "Time cost:{:.2}s, [{:.1}s - {:.1}s]: {}",
                start.elapsed().as_secs_f32(),
                token.start as f32 / 1000.0,
                token.end as f32 / 1000.0,
                token.text
            );
        }
    }

    Ok(())
}
