use llmr::diagnostics::{EnvCheck, EnvDiagnostic};

#[test]
fn test_env_diagnostic_new() {
    let diag = EnvDiagnostic::new();
    assert!(diag.config_dir.to_string_lossy().contains("llmr"));
}

#[test]
fn test_env_diagnostic_default() {
    let diag = EnvDiagnostic::default();
    assert!(diag.config_dir.to_string_lossy().contains("llmr"));
}

#[test]
fn test_env_check() {
    let check = EnvCheck::new();
    let _ = check.check();
}
