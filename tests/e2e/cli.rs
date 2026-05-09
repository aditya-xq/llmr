use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("llmr"));
}

#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("llmr"))
        .stdout(predicate::str::contains("INFO").not())
        .stdout(predicate::str::contains("Starting llmr").not());
}

#[test]
fn test_cli_doctor() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("doctor")
        .assert()
        .code(predicate::ne(2))
        .stdout(predicate::str::contains("llmr").or(predicate::str::contains("Doctor")));
}

#[test]
fn test_cli_serve_no_model_interactive() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("serve")
        .assert()
        .success()
        .stdout(predicate::str::contains("Searching for GGUF models"));
}

#[test]
fn test_cli_serve_with_dry_run() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("serve")
        .arg("--model")
        .arg("nonexistent.gguf")
        .arg("--dry-run")
        .assert()
        .failure();
}

#[test]
fn test_cli_profiles_list() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("profiles").arg("list").assert().success();
}

#[test]
fn test_cli_profiles_list_alias() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("profiles").assert().success();
}

#[test]
fn test_cli_status() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("status").assert().success();
}

#[test]
fn test_cli_status_name() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("status")
        .arg("--name")
        .arg("nonexistent-container")
        .assert()
        .success();
}

#[test]
fn test_cli_stop() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("stop").assert().success();
}

#[test]
fn test_cli_stop_name() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("stop")
        .arg("--name")
        .arg("nonexistent-container")
        .assert()
        .success();
}

#[test]
fn test_cli_profiles_show_missing_key() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("profiles")
        .arg("show")
        .arg("nonexistent-key-for-testing")
        .assert()
        .success();
}

#[test]
fn test_cli_profiles_delete_missing_key() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("profiles")
        .arg("delete")
        .arg("nonexistent-key")
        .assert()
        .success();
}

#[test]
fn test_cli_subcommand_versions() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("serve"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("stop"))
        .stdout(predicate::str::contains("profiles"))
        .stdout(predicate::str::contains("doctor"));
}

#[test]
fn test_cli_verbose_flag() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("--verbose").arg("version").assert().success();
}

#[test]
fn test_cli_quiet_flag() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("--quiet").arg("version").assert().success();
}

#[test]
fn test_cli_serve_with_gpu_flag() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("serve")
        .arg("--model")
        .arg("test.gguf")
        .arg("--no-gpu")
        .arg("--dry-run")
        .assert()
        .failure();
}

#[test]
fn test_cli_serve_with_threads() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("serve")
        .arg("--model")
        .arg("test.gguf")
        .arg("--threads")
        .arg("4")
        .arg("--dry-run")
        .assert()
        .failure();
}

#[test]
fn test_cli_serve_with_port() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("serve")
        .arg("--model")
        .arg("test.gguf")
        .arg("--port")
        .arg("9999")
        .arg("--dry-run")
        .assert()
        .failure();
}

#[test]
fn test_cli_serve_dry_run_uses_runtime_overrides() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("override test.gguf");
    std::fs::write(&model_path, "dummy").unwrap();

    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("serve")
        .arg("--model")
        .arg(&model_path)
        .arg("--dry-run")
        .arg("--threads")
        .arg("6")
        .arg("--ctx-size")
        .arg("16384")
        .arg("--batch-size")
        .arg("1024")
        .arg("--ubatch-size")
        .arg("256")
        .arg("--parallel")
        .arg("3")
        .arg("--cache-type-k")
        .arg("q8_0")
        .arg("--cache-type-v")
        .arg("q8_0")
        .arg("--port")
        .arg("9090")
        .assert()
        .success()
        .stdout(predicate::str::contains("docker run"))
        .stdout(predicate::str::contains("127.0.0.1:9090:8080"))
        .stdout(predicate::str::contains("16384"))
        .stdout(predicate::str::contains("1024"))
        .stdout(predicate::str::contains("256"))
        .stdout(predicate::str::contains("q8_0"))
        .stdout(predicate::str::contains("3"));
}

#[test]
fn test_cli_tune_requires_model() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("tune")
        .assert()
        .success()
        .stdout(predicate::str::contains("Searching for GGUF models"));
}

#[test]
fn test_cli_tune_dry_run() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("tune")
        .arg("--model")
        .arg("nonexistent.gguf")
        .arg("--dry-run")
        .assert()
        .failure();
}
