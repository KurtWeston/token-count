use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::fs;
use std::io::{self, Read};

mod cost;
mod tokenizers;

use cost::CostCalculator;
use tokenizers::{ModelType, Tokenizer};

#[derive(Parser)]
#[command(name = "token-count")]
#[command(about = "Calculate token counts for LLM text using various tokenizers")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Count {
        #[arg(short, long, help = "Model to use (gpt3, gpt4, claude2, claude3, llama2, llama3)")]
        model: String,
        #[arg(short, long, help = "Input file path")]
        file: Option<String>,
        #[arg(short, long, help = "Text input")]
        text: Option<String>,
        #[arg(short, long, help = "Output as JSON")]
        json: bool,
    },
    Compare {
        #[arg(short, long, help = "Input file path")]
        file: Option<String>,
        #[arg(short, long, help = "Text input")]
        text: Option<String>,
        #[arg(short, long, help = "Output as JSON")]
        json: bool,
    },
    Batch {
        #[arg(help = "File paths to analyze")]
        files: Vec<String>,
        #[arg(short, long, help = "Model to use")]
        model: String,
        #[arg(short, long, help = "Output as JSON")]
        json: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Count { model, file, text, json } => {
            let input = get_input(file, text)?;
            let model_type = parse_model(&model)?;
            count_tokens(&input, model_type, json)?;
        }
        Commands::Compare { file, text, json } => {
            let input = get_input(file, text)?;
            compare_models(&input, json)?;
        }
        Commands::Batch { files, model, json } => {
            let model_type = parse_model(&model)?;
            batch_process(files, model_type, json)?;
        }
    }

    Ok(())
}

fn get_input(file: Option<String>, text: Option<String>) -> Result<String> {
    if let Some(f) = file {
        Ok(fs::read_to_string(f)?)
    } else if let Some(t) = text {
        Ok(t)
    } else {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        Ok(buffer)
    }
}

fn parse_model(model: &str) -> Result<ModelType> {
    match model.to_lowercase().as_str() {
        "gpt3" | "gpt-3.5" => Ok(ModelType::Gpt35),
        "gpt4" | "gpt-4" => Ok(ModelType::Gpt4),
        "claude2" => Ok(ModelType::Claude2),
        "claude3" => Ok(ModelType::Claude3),
        "llama2" => Ok(ModelType::Llama2),
        "llama3" => Ok(ModelType::Llama3),
        _ => Err(anyhow::anyhow!("Unknown model: {}", model)),
    }
}

fn count_tokens(text: &str, model: ModelType, json: bool) -> Result<()> {
    let tokenizer = Tokenizer::new(model)?;
    let count = tokenizer.count(text)?;
    let calculator = CostCalculator::new();
    let cost = calculator.estimate_cost(model, count);

    if json {
        let output = serde_json::json!({
            "model": format!("{:?}", model),
            "tokens": count,
            "characters": text.len(),
            "ratio": text.len() as f64 / count as f64,
            "cost_per_1k": cost.input_per_1k,
            "estimated_cost": cost.total_input
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("{}\n", "Token Analysis".bold().cyan());
        println!("Model:      {}", format!("{:?}", model).green());
        println!("Tokens:     {}", count.to_string().yellow());
        println!("Characters: {}", text.len());
        println!("Ratio:      {:.2} chars/token", text.len() as f64 / count as f64);
        println!("\n{}", "Cost Estimate".bold().cyan());
        println!("Input:  ${:.6} (${:.4}/1K tokens)", cost.total_input.to_string().green(), cost.input_per_1k);
    }

    Ok(())
}

fn compare_models(text: &str, json: bool) -> Result<()> {
    let models = vec![
        ModelType::Gpt35,
        ModelType::Gpt4,
        ModelType::Claude3,
        ModelType::Llama3,
    ];

    let calculator = CostCalculator::new();
    let mut results = Vec::new();

    for model in models {
        let tokenizer = Tokenizer::new(model)?;
        let count = tokenizer.count(text)?;
        let cost = calculator.estimate_cost(model, count);
        results.push((model, count, cost));
    }

    if json {
        let output: Vec<_> = results
            .iter()
            .map(|(model, count, cost)| {
                serde_json::json!({
                    "model": format!("{:?}", model),
                    "tokens": count,
                    "cost": cost.total_input
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("{}\n", "Model Comparison".bold().cyan());
        for (model, count, cost) in results {
            println!(
                "{:<12} {:>6} tokens  ${:.6}",
                format!("{:?}", model).green(),
                count.to_string().yellow(),
                cost.total_input.to_string().cyan()
            );
        }
    }

    Ok(())
}

fn batch_process(files: Vec<String>, model: ModelType, json: bool) -> Result<()> {
    let tokenizer = Tokenizer::new(model)?;
    let calculator = CostCalculator::new();
    let mut results = Vec::new();

    for file in &files {
        let text = fs::read_to_string(file)?;
        let count = tokenizer.count(&text)?;
        let cost = calculator.estimate_cost(model, count);
        results.push((file.clone(), count, cost));
    }

    if json {
        let output: Vec<_> = results
            .iter()
            .map(|(file, count, cost)| {
                serde_json::json!({
                    "file": file,
                    "tokens": count,
                    "cost": cost.total_input
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("{}\n", "Batch Analysis".bold().cyan());
        let total: usize = results.iter().map(|(_, count, _)| count).sum();
        let total_cost: f64 = results.iter().map(|(_, _, cost)| cost.total_input).sum();

        for (file, count, cost) in &results {
            println!(
                "{:<30} {:>6} tokens  ${:.6}",
                file.green(),
                count.to_string().yellow(),
                cost.total_input.to_string().cyan()
            );
        }
        println!("\n{}", "Total".bold());
        println!("Tokens: {}  Cost: ${:.6}", total.to_string().yellow(), total_cost.to_string().green());
    }

    Ok(())
}
