use enthalpy::Res;
pub use enthalpy::util::modelscope::ModelScopeRepo;

#[tokio::main]
async fn main() -> Res<()> {
    let repo = ModelScopeRepo::new(
        "lovemefan/SenseVoiceGGUF",
        "/Users/entropy/.cache/modelscope/hub/models/",
    );

    let file = repo.get("README.md").await?;

    println!("{:?}", file);

    Ok(())
}
