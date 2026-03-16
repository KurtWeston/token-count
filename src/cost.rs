use crate::tokenizers::ModelType;

#[derive(Debug)]
pub struct CostEstimate {
    pub input_per_1k: f64,
    pub output_per_1k: f64,
    pub total_input: f64,
}

pub struct CostCalculator;

impl CostCalculator {
    pub fn new() -> Self {
        Self
    }

    pub fn estimate_cost(&self, model: ModelType, tokens: usize) -> CostEstimate {
        let (input_per_1k, output_per_1k) = self.get_pricing(model);
        let total_input = (tokens as f64 / 1000.0) * input_per_1k;

        CostEstimate {
            input_per_1k,
            output_per_1k,
            total_input,
        }
    }

    fn get_pricing(&self, model: ModelType) -> (f64, f64) {
        match model {
            ModelType::Gpt35 => (0.0015, 0.002),
            ModelType::Gpt4 => (0.03, 0.06),
            ModelType::Claude2 => (0.008, 0.024),
            ModelType::Claude3 => (0.003, 0.015),
            ModelType::Llama2 => (0.0002, 0.0002),
            ModelType::Llama3 => (0.0003, 0.0003),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_new() {
        let calc = CostCalculator::new();
        assert!(std::mem::size_of_val(&calc) == 0);
    }

    #[test]
    fn test_estimate_cost_gpt35() {
        let calc = CostCalculator::new();
        let estimate = calc.estimate_cost(ModelType::Gpt35, 1000);
        assert_eq!(estimate.input_per_1k, 0.0015);
        assert_eq!(estimate.output_per_1k, 0.002);
        assert_eq!(estimate.total_input, 0.0015);
    }

    #[test]
    fn test_estimate_cost_gpt4() {
        let calc = CostCalculator::new();
        let estimate = calc.estimate_cost(ModelType::Gpt4, 2000);
        assert_eq!(estimate.input_per_1k, 0.03);
        assert_eq!(estimate.total_input, 0.06);
    }

    #[test]
    fn test_estimate_cost_zero_tokens() {
        let calc = CostCalculator::new();
        let estimate = calc.estimate_cost(ModelType::Gpt35, 0);
        assert_eq!(estimate.total_input, 0.0);
    }

    #[test]
    fn test_estimate_cost_claude2() {
        let calc = CostCalculator::new();
        let estimate = calc.estimate_cost(ModelType::Claude2, 1000);
        assert_eq!(estimate.input_per_1k, 0.008);
        assert_eq!(estimate.output_per_1k, 0.024);
    }

    #[test]
    fn test_estimate_cost_llama2() {
        let calc = CostCalculator::new();
        let estimate = calc.estimate_cost(ModelType::Llama2, 5000);
        assert_eq!(estimate.input_per_1k, 0.0002);
        assert_eq!(estimate.total_input, 0.001);
    }

    #[test]
    fn test_estimate_cost_large_token_count() {
        let calc = CostCalculator::new();
        let estimate = calc.estimate_cost(ModelType::Gpt4, 100000);
        assert_eq!(estimate.total_input, 3.0);
    }

    #[test]
    fn test_all_models_have_pricing() {
        let calc = CostCalculator::new();
        let models = vec![
            ModelType::Gpt35,
            ModelType::Gpt4,
            ModelType::Claude2,
            ModelType::Claude3,
            ModelType::Llama2,
            ModelType::Llama3,
        ];
        
        for model in models {
            let estimate = calc.estimate_cost(model, 1000);
            assert!(estimate.input_per_1k > 0.0);
            assert!(estimate.output_per_1k > 0.0);
        }
    }
}