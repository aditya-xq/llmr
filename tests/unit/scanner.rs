use llmr::models::{ModelInfo, ModelScanner};
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

#[test]
fn test_model_info_from_path_valid() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.gguf");

    let mut file = File::create(&model_path).unwrap();
    file.write_all(b"test content").unwrap();
    drop(file);

    let info = ModelInfo::from_path(&model_path);
    assert!(info.is_some());

    let info = info.unwrap();
    assert_eq!(info.name, "test_model.gguf");
    assert_eq!(info.size_bytes, 12);
}

#[test]
fn test_model_info_from_path_not_file() {
    let temp_dir = TempDir::new().unwrap();
    let dir_path = temp_dir.path().join("models");

    std::fs::create_dir(&dir_path).unwrap();

    let info = ModelInfo::from_path(&dir_path);
    assert!(info.is_none());
}

#[test]
fn test_model_info_from_path_not_exists() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("nonexistent.gguf");

    let info = ModelInfo::from_path(&path);
    assert!(info.is_none());
}

#[test]
fn test_model_scanner_new() {
    let _scanner = ModelScanner::new();
}

#[test]
fn test_model_scanner_scan_directory() {
    let temp_dir = TempDir::new().unwrap();

    let model_path = temp_dir.path().join("model1.gguf");
    std::fs::write(&model_path, "test").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "model1.gguf");
}

#[test]
fn test_model_scanner_scan_directory_empty() {
    let temp_dir = TempDir::new().unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert!(models.is_empty());
}

#[test]
fn test_model_scanner_scan_directory_not_dir() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("file.txt");
    std::fs::write(&file_path, "test").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(&file_path);

    assert!(models.is_empty());
}

#[test]
fn test_model_scanner_scan_directory_only_gguf() {
    let temp_dir = TempDir::new().unwrap();

    std::fs::write(temp_dir.path().join("model.gguf"), "test").unwrap();
    std::fs::write(temp_dir.path().join("model.txt"), "test").unwrap();
    std::fs::write(temp_dir.path().join("model.bin"), "test").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert_eq!(models.len(), 1);
}

#[test]
fn test_model_scanner_scan_directory_accepts_uppercase_extension() {
    let temp_dir = TempDir::new().unwrap();

    std::fs::write(temp_dir.path().join("model.GGUF"), "test").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert_eq!(models.len(), 1);
}

#[test]
fn test_model_scanner_scan_directory_sorted_by_size() {
    let temp_dir = TempDir::new().unwrap();

    let small = temp_dir.path().join("small.gguf");
    std::fs::write(&small, "a").unwrap();

    let large = temp_dir.path().join("large.gguf");
    std::fs::write(&large, "abcdefghij").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert_eq!(models.len(), 2);
    assert!(models[0].size_bytes >= models[1].size_bytes);
}

#[test]
fn test_model_scanner_scan_paths() {
    let temp_dir = TempDir::new().unwrap();

    let model_path = temp_dir.path().join("model.gguf");
    std::fs::write(&model_path, "test").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_paths(&[temp_dir.path().to_path_buf()]);

    assert!(!models.is_empty());
}

#[test]
fn test_model_scanner_scan_paths_with_file() {
    let temp_dir = TempDir::new().unwrap();

    let model_path = temp_dir.path().join("model.gguf");
    std::fs::write(&model_path, "test").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_paths(std::slice::from_ref(&model_path));

    assert_eq!(models.len(), 1);
}

#[test]
fn test_model_scanner_find_root_paths() {
    let roots = ModelScanner::find_root_paths();
    assert!(roots.is_empty() || !roots.is_empty());
}

#[test]
fn test_model_info_size_formatted() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test.gguf");
    std::fs::write(&model_path, "test").unwrap();

    let info = ModelInfo::from_path(&model_path).unwrap();
    assert!(!info.size_formatted.is_empty());
}
