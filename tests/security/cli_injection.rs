use assert_cmd::Command;

#[test]
fn serve_null_bytes_not_spawned() {
    let result = std::panic::catch_unwind(|| {
        let _ = Command::cargo_bin("llmr")
            .unwrap()
            .arg("serve")
            .arg("--model")
            .arg("test\x00model")
            .assert();
    });
    assert!(result.is_err());
}

#[test]
fn serve_path_traversal_windows() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("..\\..\\..\\windows\\system32\\config\\sam")
        .assert()
        .failure();
}

#[test]
fn serve_shell_metachar_semicolon() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("test; whoami")
        .assert()
        .failure();
}

#[test]
fn serve_shell_metachar_pipe() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("test | cat /etc/passwd")
        .assert()
        .failure();
}

#[test]
fn serve_shell_metachar_backtick() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("test`ls`")
        .assert()
        .failure();
}

#[test]
fn serve_shell_metachar_dollar() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("test$(whoami)")
        .assert()
        .failure();
}

#[test]
fn serve_very_long_path() {
    let long_path = "a".repeat(10000);
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg(&long_path)
        .assert()
        .failure();
}

#[test]
fn tune_path_traversal() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("tune")
        .arg("--model")
        .arg("../../../root/.ssh/id_rsa")
        .assert()
        .failure();
}

#[test]
fn bench_path_traversal() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--model")
        .arg("../../../etc/shadow")
        .assert()
        .failure();
}

#[test]
fn serve_newline_injection() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("test\nrm -rf /")
        .assert()
        .failure();
}

#[test]
fn serve_carriage_return_injection() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("test\rrm -rf /")
        .assert()
        .failure();
}

#[test]
fn serve_tab_injection() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--model")
        .arg("test\twhoami")
        .assert()
        .failure();
}
