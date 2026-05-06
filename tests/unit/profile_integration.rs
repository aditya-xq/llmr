use llmr::models::{Profile, ProfileManager};
use tempfile::TempDir;

fn create_test_hardware() -> llmr::hardware::HardwareInfo {
    llmr::hardware::HardwareInfo {
        cpu: llmr::hardware::CpuInfo {
            cores: 8,
            threads: 16,
            name: "Test CPU".to_string(),
            architecture: "x86_64".to_string(),
            frequency: Some(3600),
        },
        gpu: Some(llmr::hardware::GpuInfo {
            names: vec!["NVIDIA RTX 3080".to_string()],
            vram_mb: vec![10240],
            vram_free_mb: vec![8000],
            type_: "nvidia".to_string(),
        }),
        ram: llmr::hardware::RamInfo {
            total: 32 * 1024 * 1024 * 1024,
            total_gb: 32,
            free_gb: 16,
        },
        has_nvlink: false,
    }
}

#[test]
fn test_profile_manager_new() {
    let _pm = ProfileManager::new();
    let config_dir = ProfileManager::config_dir();
    assert!(config_dir.to_string_lossy().contains("llmr"));
}

#[tokio::test]
async fn test_profile_save_load() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let hardware = create_test_hardware();
    let profile = Profile::new("test.gguf".to_string(), 1_000_000_000, &hardware);

    pm.save(&profile).await.unwrap();

    let loaded = pm.load(&profile.key()).await.unwrap();
    assert!(loaded.is_some());

    let loaded = loaded.unwrap();
    assert_eq!(loaded.model_file, "test.gguf");
}

#[tokio::test]
async fn test_profile_load_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let loaded = pm.load("nonexistent").await.unwrap();
    assert!(loaded.is_none());
}

#[tokio::test]
async fn test_profile_delete() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let hardware = create_test_hardware();
    let profile = Profile::new("test.gguf".to_string(), 1_000_000_000, &hardware);

    pm.save(&profile).await.unwrap();
    pm.delete(&profile.key()).await.unwrap();

    let loaded = pm.load(&profile.key()).await.unwrap();
    assert!(loaded.is_none());
}

#[tokio::test]
async fn test_profile_list_all() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let hardware = create_test_hardware();

    for i in 0..3 {
        let profile = Profile::new(
            format!("model{}.gguf", i),
            (i + 1) as u64 * 1_000_000,
            &hardware,
        );
        pm.save(&profile).await.unwrap();
    }

    let profiles = pm.list_all().await.unwrap();
    assert_eq!(profiles.len(), 3);
}

#[tokio::test]
async fn test_profile_clear_all() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let hardware = create_test_hardware();
    let profile = Profile::new("test.gguf".to_string(), 1_000_000_000, &hardware);
    pm.save(&profile).await.unwrap();

    pm.clear_all().await.unwrap();

    let profiles = pm.list_all().await.unwrap();
    assert!(profiles.is_empty());
}

#[tokio::test]
async fn test_profile_load_or_create() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let model_file = temp_dir.path().join("model.gguf");
    std::fs::write(&model_file, "dummy").unwrap();

    let hardware = create_test_hardware();
    let profile = pm
        .load_or_create(model_file.to_str().unwrap(), &hardware)
        .await
        .unwrap();

    assert!(profile.model_file.contains("model.gguf"));
    assert!(profile.threads > 0);
}

#[test]
fn test_profile_key_generation() {
    let hardware = create_test_hardware();
    let profile = Profile::new("test.gguf".to_string(), 4_000_000_000, &hardware);

    let key = profile.key();
    assert!(key.contains("test"));
    assert!(key.contains("cpu8"));
}

#[test]
fn test_profile_validate() {
    let hardware = create_test_hardware();
    let mut profile = Profile::new("test.gguf".to_string(), 4_000_000_000, &hardware);

    assert!(profile.validate().is_ok());

    profile.model_file = "".to_string();
    assert!(profile.validate().is_err());
}
