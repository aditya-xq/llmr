# llmr

**Drop a GGUF in. Get an API out.** Zero config, auto-optimized.

A tiny CLI that runs GGUF models through llama.cpp Docker servers with automatic hardware detection, model discovery, and tuned inference profiles, all in one command. vLLM and SGLang support are planned and the backend boundary is being kept explicit so those adapters can land without changing the CLI shape.

## Quick Start

```bash
# Install
cargo install --path .

# Go (auto-finds GGUF models, detects hardware, optimizes settings)
llmr serve
```

That's it. Your model is live at `http://localhost:8080`.

## Why llmr

- **No config** — Detects CPU, GPU, VRAM, and RAM automatically
- **No hunting** — Scans your drives for GGUF files on first run
- **No guesswork** — Benchmarks your hardware and caches optimal profiles
- **No lock-in** — Runs in Docker, works on Linux, macOS, and Windows
- **No manual daemon step** — If Docker is installed but stopped, `serve` and `tune` try to start it before running containers or tuning benchmarks

## Commands

| Command | Description |
|---------|-------------|
| `llmr serve` | Start a llama.cpp server (auto-discovers GGUF models) |
| `llmr serve -m model.gguf` | Serve a specific GGUF model with llama.cpp |
| `llmr serve --public` | Bind to 0.0.0.0 |
| `llmr serve --no-gpu` | CPU-only mode |
| `llmr serve --auto` | Auto-tune on first run |
| `llmr serve --benchmark` | Run tuning benchmark immediately |
| `llmr serve --skip-hardware` | Skip hardware detection |
| `llmr serve --dry-run` | Print docker command without running |
| `llmr serve --quick` | Skip tuning, use defaults |
| `llmr status` | Show running containers |
| `llmr stop` | Stop all servers |
| `llmr stop -n <name>` | Stop specific container |
| `llmr profiles list` | List cached profiles |
| `llmr profiles show <key>` | Show profile details |
| `llmr profiles delete <key>` | Delete cached profile |
| `llmr profiles clear` | Clear all cached profiles |
| `llmr profiles --file <path>` | Use alternate profile storage path |
| `llmr tune` | Auto-tune a llama.cpp profile for a GGUF model |
| `llmr tune -m model.gguf` | Tune a specific GGUF model |
| `llmr doctor` | Run diagnostics |
| `llmr update` | Update to latest version |
| `llmr version` | Show version |

Common serve options: `-p` port, `-t` threads, `-c` ctx_size, `-g` gpu_layers, `-b` batch_size, `-u` ubatch_size, `--parallel`, `--split-mode`, `--cache-type-k`, `--cache-type-v`, `--dry-run`, `--debug`.

Run `llmr serve --help` for the full list.

## Tuning

| Command | Description |
|---------|-------------|
| `llmr tune` | Auto-tune a llama.cpp profile for a GGUF model |
| `llmr tune -m model.gguf` | Tune a specific GGUF model |
| `llmr tune --dry-run` | Show tuning result without saving |
| `llmr tune --quick` | Run fewer benchmark iterations |
| `llmr tune --max-rounds <n>` | Number of tuning rounds (default: 4) |
| `llmr tune --prompt-tokens <n>` | Prompt tokens for benchmarking (default: 512) |
| `llmr tune --generation-tokens <n>` | Generation tokens for benchmarking (default: 128) |

## Benchmarks

### llmr bench (CLI)

Run performance and quality benchmarks against a running server (or auto-start one).

| Command | Description |
|---------|-------------|
| `llmr bench` | Run performance benchmarks against a running server |
| `llmr bench --model model.gguf` | Auto-start server and run benchmarks |
| `llmr bench --tasks gsm8k` | Quality evaluation using lm-evaluation-harness |
| `llmr bench --base-url <url>` | Target server URL (default: http://127.0.0.1:8080) |
| `llmr bench --config <file>` | Use a custom benchmark config |
| `llmr bench --test-type <type>` | Test type: quality, latency, throughput |
| `llmr bench --dry-run` | Show planned benchmark without running |
| `llmr bench --quick` | Run reduced iterations |
| `llmr bench --parallel <n>` | Parallel requests (default: 1) |
| `llmr bench --retries <n>` | Retry failed requests (default: 3) |

Quality evaluation with lm-evaluation-harness:

```bash
llmr bench --model model.gguf --tasks gsm8k
```

The `--model` flag auto-starts the server if needed. Without it, `llmr bench` targets a running server at `--base-url`.

### llmr-bench (Standalone Binary)

Standalone performance benchmarking tool for detailed throughput and latency analysis.

```bash
llmr-bench --config config.yaml --output report.json
```

| Option | Description |
|--------|-------------|
| `-c, --config <CONFIG>` | Benchmark config file (default: config.yaml) |
| `-o, --output <OUTPUT>` | Output report file (JSON) |
| `--verbose` | Verbose output |

## Profiles

Settings are auto-cached per model + hardware combo. On first run, `llmr serve` asks whether to run tuning. If accepted, Docker is started if needed, benchmarks must complete successfully, and only then is the tuned profile saved. Subsequent starts reuse the cached profile.

Config lives at `~/.config/llmr/` (Linux), `~/Library/Application Support/llmr/` (macOS), or `%APPDATA%\llmr\` (Windows).

## Prerequisites

- [Rust](https://rustup.rs/) 1.75+
- [Docker](https://docs.docker.com/get-docker/)
- Python 3.10+ (for quality evaluation with lm-evaluation-harness)

`llmr serve` and `llmr tune` require Docker for real execution. Dry-run flows stay offline and only render the planned command/profile behavior.

## Building

```bash
cargo build --release
cargo test
```