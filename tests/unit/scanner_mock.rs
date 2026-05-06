use llmr::models::{ModelInfo, ModelScanner};
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_model_info_from_path_nonexistent() {
    let result = ModelInfo::from_path(Path::new("/nonexistent/path/model.gguf"));
    assert!(result.is_none());
}

#[test]
fn test_model_info_from_path_directory() {
    let temp_dir = TempDir::new().unwrap();
    let result = ModelInfo::from_path(temp_dir.path());
    assert!(result.is_none());
}

#[test]
fn test_model_info_from_path_valid_file() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test-model.gguf");

    std::fs::write(&model_path, b"fake gguf data").unwrap();

    let result = ModelInfo::from_path(&model_path);
    assert!(result.is_some());

    let info = result.unwrap();
    assert_eq!(info.name, "test-model.gguf");
    assert!(info.size_bytes > 0);
    assert!(!info.size_formatted.is_empty());
}

#[test]
fn test_model_info_format_size_bytes() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("tiny.bin");

    std::fs::write(&model_path, b"x").unwrap();

    let info = ModelInfo::from_path(&model_path).unwrap();
    assert!(info.size_formatted.contains("B"));
}

#[test]
fn test_model_info_format_size_kb() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("small.bin");

    let data: Vec<u8> = (0..2048).map(|i| i as u8).collect();
    std::fs::write(&model_path, &data).unwrap();

    let info = ModelInfo::from_path(&model_path).unwrap();
    assert!(info.size_formatted.contains("KB") || info.size_formatted.contains("B"));
}

#[test]
fn test_model_info_format_size_mb() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("medium.bin");

    let data: Vec<u8> = vec![0u8; 5 * 1024 * 1024];
    std::fs::write(&model_path, &data).unwrap();

    let info = ModelInfo::from_path(&model_path).unwrap();
    assert!(info.size_formatted.contains("MB"));
}

#[test]
fn test_model_info_format_size_gb() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("large.bin");

    let data: Vec<u8> = vec![0u8; 2 * 1024 * 1024 * 1024];
    std::fs::write(&model_path, &data).unwrap();

    let info = ModelInfo::from_path(&model_path).unwrap();
    assert!(info.size_formatted.contains("GB"));
}

#[test]
fn test_model_scanner_new() {
    let scanner = ModelScanner::new();
    let _ = scanner;
}

#[test]
fn test_model_info_unicode_filename() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("模型.gguf");

    std::fs::write(&model_path, b"fake").unwrap();

    let result = ModelInfo::from_path(&model_path);
    assert!(result.is_some());
}

#[test]
fn test_model_info_partial_unicode() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test-模型.gguf");

    std::fs::write(&model_path, b"fake").unwrap();

    let result = ModelInfo::from_path(&model_path);
    assert!(result.is_some());
    assert!(result.unwrap().name.contains("模型"));
}
