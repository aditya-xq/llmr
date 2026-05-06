use assert_cmd::Command;

#[test]
fn serve_nonexistent_model() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("/nonexistent/path/model.gguf")
        .assert()
        .failure();
}

#[test]
fn serve_invalid_port_65536() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--port")
        .arg("65536")
        .arg("--dry-run")
        .assert()
        .failure();
}

#[test]
fn serve_invalid_split_mode() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--split-mode")
        .arg("invalid")
        .arg("--dry-run")
        .assert()
        .failure();
}

#[test]
fn serve_invalid_gpu_layers_negative() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--gpu-layers")
        .arg("-1")
        .arg("--dry-run")
        .assert()
        .failure();
}

#[test]
fn profiles_show_nonexistent() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("profiles")
        .arg("show")
        .arg("nonexistent/key")
        .assert()
        .success();
}

#[test]
fn profiles_delete_nonexistent() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("profiles")
        .arg("delete")
        .arg("nonexistent/key")
        .assert()
        .success();
}

#[test]
fn bench_invalid_base_url() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--base-url")
        .arg("not-a-url")
        .assert()
        .failure();
}

#[test]
fn serve_empty_model_argument() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("")
        .assert()
        .failure();
}

#[test]
fn tune_empty_model_argument() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("tune")
        .arg("--model")
        .arg("")
        .assert()
        .failure();
}

#[test]
fn unknown_subcommand() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("unknowncommand")
        .assert()
        .failure();
}

#[test]
fn invalid_global_flag() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("--invalid-flag")
        .assert()
        .failure();
}
