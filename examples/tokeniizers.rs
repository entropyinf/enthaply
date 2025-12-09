use enthalpy::Res;
use enthalpy::util::modelscope;
use modelscope::ModelScopeRepo;
use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use std::fs;

#[tokio::main]
async fn main() -> Res<()> {
    let repo = ModelScopeRepo::new(
        "Tongyi-MAI/Z-Image-Turbo",
        "/Users/entropy/.cache/modelscope/hub/models/",
    );
    let vocab = repo.get("tokenizer/vocab.json").await?;
    let merges = repo.get("tokenizer/merges.txt").await?;
    
    let bpe_builder = BPE::from_file(vocab.to_str().unwrap(), merges.to_str().unwrap());
    let bpe = bpe_builder.build().unwrap();
    let tokenizer = Tokenizer::new(bpe);
    let encoding = tokenizer.encode("Hello world!", false).unwrap();
    
    println!("Tokens: {:?}", encoding.get_tokens());
    println!("Ids: {:?}", encoding.get_ids());

    Ok(())
}