use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn smoke_help() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("llmr"))
        .stdout(predicate::str::contains("serve"));
}

#[test]
fn smoke_version() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("llmr"))
        .stdout(predicate::str::contains("1.0"));
}

#[test]
fn smoke_doctor() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains("llmr").or(predicate::str::contains("Doctor")));
}

#[test]
fn smoke_status() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("status")
        .assert()
        .success();
}

#[test]
fn smoke_profiles_list() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("profiles")
        .arg("list")
        .assert()
        .success();
}

#[test]
fn smoke_serve_help() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("serve")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("serve"));
}

#[test]
fn smoke_tune_help() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("tune")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("tune"));
}

#[test]
fn smoke_bench_help() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("bench")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("bench"));
}

#[test]
fn smoke_update_check() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("update")
        .arg("--check")
        .assert()
        .code(predicate::eq(0).or(predicate::eq(1)));
}

#[test]
fn smoke_update_help() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("update")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("update"));
}

#[test]
fn smoke_profiles_help() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("profiles")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("profiles"));
}

#[test]
fn smoke_stop_help() {
    Command::cargo_bin("llmr")
        .unwrap()
        .arg("stop")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("stop"));
}
