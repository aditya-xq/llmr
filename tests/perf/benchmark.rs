use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

#[test]
fn bench_help_shows_options() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("base-url"))
        .stdout(predicate::str::contains("model"));
}

#[test]
fn bench_default_base_url() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("127.0.0.1:8080"));
}

#[test]
fn bench_with_custom_base_url() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--base-url")
        .arg("http://localhost:9090")
        .arg("--dry-run")
        .assert()
        .success();
}

#[test]
fn bench_invalid_url_format() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--base-url")
        .arg("not-a-valid-url")
        .assert()
        .failure();
}

#[test]
fn bench_with_config_file_nonexistent() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("bench")
        .arg("--config")
        .arg("/nonexistent/config.yaml")
        .assert()
        .failure();
}

#[test]
fn bench_dry_run_quick() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--dry-run")
        .arg("--quick")
        .assert()
        .success();
}

#[test]
fn bench_with_model_flag() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--model")
        .arg("test-model.gguf")
        .arg("--dry-run")
        .assert()
        .success();
}

#[test]
fn bench_with_prompt_tokens() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--prompt-tokens")
        .arg("256")
        .arg("--dry-run")
        .assert()
        .success();
}

#[test]
fn bench_with_generation_tokens() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--generation-tokens")
        .arg("128")
        .arg("--dry-run")
        .assert()
        .success();
}

#[test]
fn bench_with_invalid_prompt_tokens_zero() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--prompt-tokens")
        .arg("0")
        .assert()
        .failure();
}

#[test]
fn bench_with_invalid_generation_tokens_zero() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--generation-tokens")
        .arg("0")
        .assert()
        .failure();
}

#[test]
fn bench_fewshot_default() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("fewshot"));
}

#[tokio::test]
async fn test_benchmark_config_parsing_valid() {
    use llmr::bench::config;

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.yaml");

    let yaml = r#"
server:
  endpoint: "/v1/chat/completions"
  health_endpoint: "/health"
  props_endpoint: "/props"
model:
  name: "test-model"
performance:
  prompts:
    - "Hello world"
  max_tokens: 100
  temperature: 0.7
  top_p: 0.9
  seed: 42
  warmup_runs: 1
  measured_runs: 3
  stream: false
quality:
  enabled: false
  tasks: []
"#;

    std::fs::write(&config_path, yaml).unwrap();

    let result = config::load_config(&config_path);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_benchmark_config_invalid_measured_runs() {
    use llmr::bench::config;

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.yaml");

    let yaml = r#"
server:
  endpoint: "/v1/chat/completions"
  health_endpoint: "/health"
model:
  name: "test-model"
performance:
  prompts:
    - "Hello world"
  max_tokens: 100
  temperature: 0.7
  top_p: 0.9
  seed: 42
  warmup_runs: 1
  measured_runs: 0
  stream: false
"#;

    std::fs::write(&config_path, yaml).unwrap();

    let result = config::load_config(&config_path);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_benchmark_config_empty_prompts() {
    use llmr::bench::config;

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.yaml");

    let yaml = r#"
server:
  endpoint: "/v1/chat/completions"
  health_endpoint: "/health"
model:
  name: "test-model"
performance:
  prompts: []
  max_tokens: 100
  temperature: 0.7
  top_p: 0.9
  seed: 42
  warmup_runs: 1
  measured_runs: 3
  stream: false
"#;

    std::fs::write(&config_path, yaml).unwrap();

    let result = config::load_config(&config_path);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_benchmark_config_temperature_out_of_range() {
    use llmr::bench::config;

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.yaml");

    let yaml = r#"
server:
  endpoint: "/v1/chat/completions"
  health_endpoint: "/health"
model:
  name: "test-model"
performance:
  prompts:
    - "Hello world"
  max_tokens: 100
  temperature: 3.0
  top_p: 0.9
  seed: 42
  warmup_runs: 1
  measured_runs: 3
  stream: false
"#;

    std::fs::write(&config_path, yaml).unwrap();

    let result = config::load_config(&config_path);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_benchmark_config_top_p_out_of_range() {
    use llmr::bench::config;

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("bench.yaml");

    let yaml = r#"
server:
  endpoint: "/v1/chat/completions"
  health_endpoint: "/health"
model:
  name: "test-model"
performance:
  prompts:
    - "Hello world"
  max_tokens: 100
  temperature: 0.7
  top_p: 0.0
  seed: 42
  warmup_runs: 1
  measured_runs: 3
  stream: false
"#;

    std::fs::write(&config_path, yaml).unwrap();

    let result = config::load_config(&config_path);
    assert!(result.is_err());
}

#[test]
fn test_server_config_defaults() {
    use llmr::bench::types::ServerConfig;

    let config = ServerConfig::default();
    assert_eq!(config.endpoint, "/v1/chat/completions");
    assert_eq!(config.health_endpoint, "/health");
    assert_eq!(config.props_endpoint, "/props");
    assert!(config.metrics_endpoint.is_none());
}

#[test]
fn test_performance_config_defaults() {
    use llmr::bench::types::PerformanceConfig;

    let config = PerformanceConfig::default();
    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.top_p, 1.0);
    assert_eq!(config.warmup_runs, 1);
    assert_eq!(config.measured_runs, 3);
    assert!(!config.prompts.is_empty());
}
