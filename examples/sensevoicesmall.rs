use enthalpy::Res;
use enthalpy::audio::input::AudioInput;
use enthalpy::audio::load_audio;
use enthalpy::audio::silero_vad::VadConfig;
use enthalpy::models::sense_voice_small::{SenseVoiceSmall, SenseVoiceSmallConfig};
use std::path::PathBuf;
use tokio::time::Instant;
use tracing::Level;

#[tokio::main]
async fn main() -> Res<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .with_env_filter("enthalpy=TRACE")
        .compact()
        .init();

    // transpose_stream().await?;
    transpose_file().await?;

    Ok(())
}

async fn transpose_file() -> Res<()> {
    let (mut data, sample_rate) =
        load_audio("/Users/entropy/Documents/NCE1-英音-(MP3+LRC)/001&002－Excuse Me.mp3")?;

    let cfg = SenseVoiceSmallConfig {
        model_dir: PathBuf::from("/Users/entropy/.cache/modelscope/hub/models/"),
        vad: VadConfig::default(),
        resample: Some((sample_rate, 16000)),
        use_gpu: true,
    };

    let mut model = SenseVoiceSmall::with_config(cfg).await?;

    let start = Instant::now();
    let mut segments = model.segment(&mut data)?;
    let tokens = model.transpose(&mut segments)?;
    println!("{:.2}", start.elapsed().as_secs_f32());
    for token in tokens {
        println!(
            "[{:.1}s,{:.1}s]:{}",
            token.start as f32 / 1000.0,
            token.end as f32 / 1000.0,
            token.text
        );
    }

    Ok(())
}

async fn transpose_stream() -> Res<()> {
    let mut input = AudioInput::from_screen_capture_kit()?;
    let sample_rate = input.config.sample_rate().0;

    let cfg = SenseVoiceSmallConfig {
        model_dir: PathBuf::from("/Users/entropy/.cache/modelscope/hub/models/"),
        vad: VadConfig::default(),
        resample: Some((sample_rate, 16000)),
        use_gpu: false,
    };

    let mut model = SenseVoiceSmall::with_config(cfg).await?;

    let pcm_data = input.play()?;
    while let Ok(mut chunk) = pcm_data.recv() {
        let start = Instant::now();
        let mut segments = model.segment(&mut chunk)?;
        let tokens = model.transpose(&mut segments)?;

        for token in tokens {
            print!("cost:{:.2},", start.elapsed().as_secs_f32());
            println!(
                "[{:.1}s,{:.1}s]:{}",
                token.start as f32 / 1000.0,
                token.end as f32 / 1000.0,
                token.text
            );
        }
    }

    Ok(())
}
