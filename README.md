# token-count

Calculate token counts for text using various LLM tokenizers to estimate API costs and context limits

## Features

- Support for GPT-3.5, GPT-4, Claude 2/3, and Llama 2/3 tokenizers
- Read input from files, stdin, or command-line arguments
- Display token count with model-specific encoding
- Calculate estimated API costs based on current pricing for each model
- Batch processing mode to analyze multiple files at once
- JSON output format for programmatic integration
- Interactive mode for testing prompts with real-time token counting
- Show token-to-character ratio and efficiency metrics
- Compare token counts across different models simultaneously
- Colorized output with clear cost breakdowns per model

## How to Use

Use this project when you need to:

- Quickly solve problems related to token-count
- Integrate rust functionality into your workflow
- Learn how rust handles common patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/KurtWeston/token-count.git
cd token-count

# Install dependencies
cargo build
```

## Usage

```bash
cargo run
```

## Built With

- rust

## Dependencies

- `clap`
- `tiktoken-rs`
- `serde`
- `serde_json`
- `colored`
- `anyhow`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
