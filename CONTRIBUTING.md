# Contributing

We welcome contributions! Please follow these guidelines.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/aditya-xq/llmr.git
cd llmr

# Build the project
cargo build

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run -- serve --model /path/to/model.gguf --dry-run
```

## Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Use clippy for linting (`cargo clippy -- -D warnings`)
- Write idiomatic Rust

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Make your changes
4. Run tests and formatting (`cargo fmt && cargo clippy -- -D warnings && cargo test`)
5. Commit with clear messages
6. Push to your fork
7. Submit a pull request

## CI

Every PR runs `cargo fmt --check`, `cargo clippy -- -D warnings`, compiles all test suites, executes Docker-free unit tests, runs a dependency security audit, and validates commit messages against conventional-commit format (see `.github/workflows/check.yml`). Integration/e2e tests that require Docker are not run in CI — run them locally with `cargo test`.

Commit subjects must follow `<type>(<scope>): <summary>` — releases derive semver versions from these messages. See the "Release & Commit Discipline" section in AGENTS.md for the full contract.

## Commit Messages

- Use clear, descriptive messages
- Start with a verb (Add, Fix, Update, Remove)
- Reference issues when applicable

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture
```
