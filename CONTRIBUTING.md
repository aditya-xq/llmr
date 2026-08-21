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
5. Commit with conventional-commit messages (below)
6. Push to your fork
7. Submit a pull request **targeting `develop`**

Only `develop` may be merged into `main`; releases ride that path automatically (see [docs/RELEASING.md](docs/RELEASING.md)).

## CI

Every PR runs `cargo fmt --check`, `cargo clippy -- -D warnings`, compiles all test suites, executes Docker-free unit tests, runs a dependency security audit, and validates commit messages against conventional-commit format (see `.github/workflows/check.yml`). Integration/e2e tests that require Docker are not run in CI — run them locally with `cargo test`.

## Commit Messages

Subjects must follow `<type>(<scope>): <summary>` — CI enforces this and release versions are derived from them:

```
feat(cli): add --port flag          # minor bump
fix(hardware): normalize MiB units  # patch bump
chore(deps): bump tokio to 1.52     # no release impact
```

Allowed types: `feat` `fix` `perf` `revert` `chore` `docs` `style` `refactor` `test` `build` `ci`. Breaking changes append `!` before the colon (`feat!:`) or add a `BREAKING CHANGE:` footer — either forces a major bump.

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture
```
