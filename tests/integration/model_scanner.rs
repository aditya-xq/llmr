use llmr::models::{ModelInfo, ModelScanner};
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

#[test]
fn test_model_scanner_single_file() {
    let temp_dir = TempDir::new().unwrap();

    let model_path = temp_dir.path().join("test.gguf");
    let mut file = File::create(&model_path).unwrap();
    file.write_all(b"test model content").unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "test.gguf");
}

#[test]
fn test_model_scanner_multiple_files() {
    let temp_dir = TempDir::new().unwrap();

    File::create(temp_dir.path().join("model1.gguf"))
        .unwrap()
        .write_all(b"content1")
        .unwrap();
    File::create(temp_dir.path().join("model2.gguf"))
        .unwrap()
        .write_all(b"content2")
        .unwrap();
    File::create(temp_dir.path().join("model3.gguf"))
        .unwrap()
        .write_all(b"content3")
        .unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert_eq!(models.len(), 3);
}

#[test]
fn test_model_scanner_excludes_non_gguf() {
    let temp_dir = TempDir::new().unwrap();

    File::create(temp_dir.path().join("model.gguf"))
        .unwrap()
        .write_all(b"content")
        .unwrap();
    File::create(temp_dir.path().join("readme.txt"))
        .unwrap()
        .write_all(b"text")
        .unwrap();
    File::create(temp_dir.path().join("data.bin"))
        .unwrap()
        .write_all(b"binary")
        .unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "model.gguf");
}

#[test]
fn test_model_scanner_nested_directories() {
    let temp_dir = TempDir::new().unwrap();

    let subdir = temp_dir.path().join("models");
    std::fs::create_dir(&subdir).unwrap();

    File::create(temp_dir.path().join("root.gguf"))
        .unwrap()
        .write_all(b"root")
        .unwrap();
    File::create(subdir.join("nested.gguf"))
        .unwrap()
        .write_all(b"nested")
        .unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert!(models.iter().any(|m| m.name == "root.gguf"));
}

#[test]
fn test_model_scanner_sorting() {
    let temp_dir = TempDir::new().unwrap();

    File::create(temp_dir.path().join("small.gguf"))
        .unwrap()
        .write_all(b"a")
        .unwrap();
    File::create(temp_dir.path().join("large.gguf"))
        .unwrap()
        .write_all(b"abcdefghijklmnop")
        .unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert!(models[0].size_bytes >= models[1].size_bytes);
}

#[test]
fn test_model_info_size_calculation() {
    let temp_dir = TempDir::new().unwrap();

    let model_path = temp_dir.path().join("test.gguf");
    let content = b"test content for size";
    File::create(&model_path)
        .unwrap()
        .write_all(content)
        .unwrap();

    let info = ModelInfo::from_path(&model_path).unwrap();
    assert_eq!(info.size_bytes, content.len() as u64);
}

#[test]
fn test_model_info_format_size_various() {
    let temp_dir = TempDir::new().unwrap();

    let path = temp_dir.path().join("test.gguf");
    File::create(&path).unwrap().write_all(b"a").unwrap();

    let info = ModelInfo::from_path(&path).unwrap();

    assert!(
        info.size_formatted.contains("B")
            || info.size_formatted.contains("KB")
            || info.size_formatted.contains("MB")
            || info.size_formatted.contains("GB")
    );
}

#[test]
fn test_model_scanner_paths_mixed() {
    let temp_dir = TempDir::new().unwrap();

    let model_path = temp_dir.path().join("model.gguf");
    File::create(&model_path)
        .unwrap()
        .write_all(b"content")
        .unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_paths(&[model_path]);

    assert!(!models.is_empty());
}

#[test]
fn test_model_scanner_empty_directory() {
    let temp_dir = TempDir::new().unwrap();

    let scanner = ModelScanner::new();
    let models = scanner.scan_directory(temp_dir.path());

    assert!(models.is_empty());
}

#[test]
fn test_model_info_name_extraction() {
    let temp_dir = TempDir::new().unwrap();

    let model_path = temp_dir.path().join("MyModel-Q4_K_M.gguf");
    File::create(&model_path)
        .unwrap()
        .write_all(b"content")
        .unwrap();

    let info = ModelInfo::from_path(&model_path).unwrap();
    assert_eq!(info.name, "MyModel-Q4_K_M.gguf");
}
