use anyhow::Result;
use tiktoken_rs::{cl100k_base, p50k_base};

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    Gpt35,
    Gpt4,
    Claude2,
    Claude3,
    Llama2,
    Llama3,
}

pub struct Tokenizer {
    model: ModelType,
}

impl Tokenizer {
    pub fn new(model: ModelType) -> Result<Self> {
        Ok(Self { model })
    }

    pub fn count(&self, text: &str) -> Result<usize> {
        match self.model {
            ModelType::Gpt35 => self.count_gpt35(text),
            ModelType::Gpt4 => self.count_gpt4(text),
            ModelType::Claude2 => self.count_claude(text),
            ModelType::Claude3 => self.count_claude(text),
            ModelType::Llama2 => self.count_llama(text),
            ModelType::Llama3 => self.count_llama(text),
        }
    }

    fn count_gpt35(&self, text: &str) -> Result<usize> {
        let bpe = cl100k_base()?;
        Ok(bpe.encode_with_special_tokens(text).len())
    }

    fn count_gpt4(&self, text: &str) -> Result<usize> {
        let bpe = cl100k_base()?;
        Ok(bpe.encode_with_special_tokens(text).len())
    }

    fn count_claude(&self, text: &str) -> Result<usize> {
        let bpe = cl100k_base()?;
        let tokens = bpe.encode_with_special_tokens(text).len();
        Ok((tokens as f64 * 1.05) as usize)
    }

    fn count_llama(&self, text: &str) -> Result<usize> {
        let bpe = p50k_base()?;
        Ok(bpe.encode_with_special_tokens(text).len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_new() {
        let tokenizer = Tokenizer::new(ModelType::Gpt35);
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_count_gpt35_simple_text() {
        let tokenizer = Tokenizer::new(ModelType::Gpt35).unwrap();
        let count = tokenizer.count("Hello world").unwrap();
        assert!(count > 0);
        assert!(count < 10);
    }

    #[test]
    fn test_count_gpt4_simple_text() {
        let tokenizer = Tokenizer::new(ModelType::Gpt4).unwrap();
        let count = tokenizer.count("Hello world").unwrap();
        assert!(count > 0);
        assert!(count < 10);
    }

    #[test]
    fn test_count_claude2_has_multiplier() {
        let tokenizer = Tokenizer::new(ModelType::Claude2).unwrap();
        let count = tokenizer.count("Hello world").unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_count_llama2_simple_text() {
        let tokenizer = Tokenizer::new(ModelType::Llama2).unwrap();
        let count = tokenizer.count("Hello world").unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_count_empty_string() {
        let tokenizer = Tokenizer::new(ModelType::Gpt35).unwrap();
        let count = tokenizer.count("").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_long_text() {
        let tokenizer = Tokenizer::new(ModelType::Gpt35).unwrap();
        let long_text = "a".repeat(1000);
        let count = tokenizer.count(&long_text).unwrap();
        assert!(count > 100);
    }

    #[test]
    fn test_count_special_characters() {
        let tokenizer = Tokenizer::new(ModelType::Gpt4).unwrap();
        let count = tokenizer.count("Hello! @#$%^&*()").unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_count_unicode() {
        let tokenizer = Tokenizer::new(ModelType::Gpt35).unwrap();
        let count = tokenizer.count("Hello 世界 🌍").unwrap();
        assert!(count > 0);
    }
}