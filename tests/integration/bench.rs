use llmr::bench::config::load_config;
use llmr::bench::report::BenchmarkReport;
use llmr::bench::types::BenchmarkConfig;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_load_config_file_not_found() {
    let result = load_config(std::path::Path::new("/nonexistent/config.yaml"));
    assert!(result.is_err());
}

#[test]
fn test_load_config_invalid_yaml() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"invalid: yaml: content: [").unwrap();
    let result = load_config(file.path());
    assert!(result.is_err());
}

#[test]
fn test_load_config_valid() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(
        b"
server:
  endpoint: http://localhost:8080/v1/chat/completions
  health_endpoint: /health
  props_endpoint: /props
model:
  name: test-model
performance:
  prompts:
    - Hello world
  max_tokens: 256
  temperature: 1.0
  top_p: 1.0
  seed: 42
  warmup_runs: 1
  measured_runs: 3
  stream: false
quality:
  enabled: false
  tasks: []
",
    )
    .unwrap();
    let result = load_config(file.path());
    assert!(result.is_ok());
    let config = result.unwrap();
    assert_eq!(config.model.name, "test-model");
}

#[test]
fn test_load_config_invalid_validation() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(
        b"
server:
  endpoint: http://localhost:8080
model:
  name: test
performance:
  prompts:
    - test
  max_tokens: 0
  temperature: 1.0
  top_p: 1.0
  seed: 42
  warmup_runs: 1
  measured_runs: 1
  stream: false
quality:
  enabled: false
  tasks: []
",
    )
    .unwrap();
    let result = load_config(file.path());
    assert!(result.is_err());
}

#[test]
fn test_report_to_json() {
    let config = BenchmarkConfig::default();
    let report = BenchmarkReport::new("test", &config, None, None, None);
    let json = report.to_json().unwrap();
    assert!(json.contains("test"));
    assert!(json.contains("run_id"));
}

#[test]
fn test_report_to_console() {
    let config = BenchmarkConfig::default();
    let report = BenchmarkReport::new("test", &config, None, None, None);
    let console = report.to_console();
    assert!(console.contains("Benchmark Report"));
    assert!(console.contains("test"));
}

#[test]
fn test_config_with_invalid_temperature() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(
        b"
server:
  endpoint: http://localhost:8080
model:
  name: test
performance:
  prompts:
    - test
  max_tokens: 100
  temperature: 5.0
  top_p: 1.0
  seed: 42
  warmup_runs: 1
  measured_runs: 1
  stream: false
quality:
  enabled: false
  tasks: []
",
    )
    .unwrap();
    let result = load_config(file.path());
    assert!(result.is_err());
}
