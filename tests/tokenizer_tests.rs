use token_count::tokenizers::{ModelType, Tokenizer};
use token_count::cost::CostCalculator;

#[test]
fn test_gpt35_tokenization() {
    let tokenizer = Tokenizer::new(ModelType::Gpt35).unwrap();
    let text = "Hello, world!";
    let count = tokenizer.count(text).unwrap();
    assert!(count > 0);
    assert!(count < text.len());
}

#[test]
fn test_gpt4_tokenization() {
    let tokenizer = Tokenizer::new(ModelType::Gpt4).unwrap();
    let text = "The quick brown fox jumps over the lazy dog.";
    let count = tokenizer.count(text).unwrap();
    assert!(count > 0);
}

#[test]
fn test_cost_calculation() {
    let calculator = CostCalculator::new();
    let cost = calculator.estimate_cost(ModelType::Gpt35, 1000);
    assert_eq!(cost.input_per_1k, 0.0015);
    assert_eq!(cost.total_input, 0.0015);
}

#[test]
fn test_cost_scaling() {
    let calculator = CostCalculator::new();
    let cost = calculator.estimate_cost(ModelType::Gpt4, 5000);
    assert_eq!(cost.total_input, 0.15);
}

#[test]
fn test_empty_text() {
    let tokenizer = Tokenizer::new(ModelType::Gpt35).unwrap();
    let count = tokenizer.count("").unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_long_text() {
    let tokenizer = Tokenizer::new(ModelType::Claude3).unwrap();
    let text = "a".repeat(10000);
    let count = tokenizer.count(&text).unwrap();
    assert!(count > 1000);
}
