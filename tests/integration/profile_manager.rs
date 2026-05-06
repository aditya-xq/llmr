use llmr::hardware::{CpuInfo, GpuInfo, HardwareInfo, RamInfo};
use llmr::models::{Profile, ProfileManager};
use tempfile::TempDir;

fn create_test_hardware() -> HardwareInfo {
    HardwareInfo {
        cpu: CpuInfo {
            cores: 8,
            threads: 16,
            name: "Test CPU".to_string(),
            architecture: "x86_64".to_string(),
            frequency: Some(3600),
        },
        gpu: Some(GpuInfo {
            names: vec!["NVIDIA RTX 3080".to_string()],
            vram_mb: vec![10240],
            vram_free_mb: vec![8000],
            type_: "nvidia".to_string(),
        }),
        ram: RamInfo {
            total: 32 * 1024 * 1024 * 1024,
            total_gb: 32,
            free_gb: 16,
        },
        has_nvlink: false,
    }
}

#[tokio::test]
async fn test_profile_manager_config_dir() {
    let config_dir = ProfileManager::config_dir();
    assert!(config_dir.to_string_lossy().contains("llmr"));
}

#[tokio::test]
async fn test_profile_save_and_retrieve() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let hardware = create_test_hardware();
    let profile = Profile::new("test.gguf".to_string(), 1_000_000_000, &hardware);

    pm.save(&profile).await.unwrap();

    let loaded = pm.load(&profile.key()).await.unwrap();
    assert!(loaded.is_some());

    let loaded = loaded.unwrap();
    assert_eq!(loaded.model_file, profile.model_file);
    assert_eq!(loaded.threads, profile.threads);
}

#[tokio::test]
async fn test_profile_overwrite() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let hardware = create_test_hardware();
    let profile1 = Profile::new("test.gguf".to_string(), 1_000_000_000, &hardware);

    pm.save(&profile1).await.unwrap();

    let mut profile2 = Profile::new("test.gguf".to_string(), 1_000_000_000, &hardware);
    profile2.threads = 16;

    pm.save(&profile2).await.unwrap();

    let loaded = pm.load(&profile2.key()).await.unwrap().unwrap();
    assert_eq!(loaded.threads, 16);
}

#[tokio::test]
async fn test_profile_delete_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let result = pm.delete("nonexistent").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_profile_list_all_empty() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let profiles = pm.list_all().await.unwrap();
    assert!(profiles.is_empty());
}

#[tokio::test]
async fn test_profile_list_all_multiple() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let hardware = create_test_hardware();

    for i in 0..5 {
        let profile = Profile::new(format!("model{}.gguf", i), i as u64 * 1_000_000, &hardware);
        pm.save(&profile).await.unwrap();
    }

    let profiles = pm.list_all().await.unwrap();
    assert_eq!(profiles.len(), 5);
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
async fn test_profile_manager_load_or_create_new_profile() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let model_file = temp_dir.path().join("newmodel.gguf");
    std::fs::write(&model_file, "dummy").unwrap();

    let hardware = create_test_hardware();

    let profile = pm
        .load_or_create(model_file.to_str().unwrap(), &hardware)
        .await
        .unwrap();

    assert!(profile.model_file.contains("newmodel.gguf"));
}

#[tokio::test]
async fn test_profile_load_or_create_existing() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let model_path = temp_dir.path().join("existingmodel.gguf");
    std::fs::write(&model_path, "dummy").unwrap();

    let hardware = create_test_hardware();

    let _ = pm
        .load_or_create(model_path.to_str().unwrap(), &hardware)
        .await
        .unwrap();
    let profile2 = pm
        .load_or_create(model_path.to_str().unwrap(), &hardware)
        .await
        .unwrap();

    assert!(profile2.model_file.contains("existingmodel.gguf"));
}

#[tokio::test]
async fn test_profile_key_generation_deterministic() {
    let hardware = create_test_hardware();

    let profile1 = Profile::new("model.gguf".to_string(), 1_000_000_000, &hardware);
    let profile2 = Profile::new("model.gguf".to_string(), 1_000_000_000, &hardware);

    let key1 = profile1.key();
    let key2 = profile2.key();

    assert_eq!(key1, key2);
}

#[tokio::test]
async fn test_profile_recompute() {
    let temp_dir = TempDir::new().unwrap();
    let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

    let model_path = temp_dir.path().join("testmodel.gguf");
    std::fs::write(&model_path, "dummy").unwrap();

    let hardware = create_test_hardware();

    let profile = pm
        .recompute_profile(model_path.to_str().unwrap(), &hardware)
        .await
        .unwrap();

    assert!(profile.model_file.contains("testmodel.gguf"));
    assert!(profile.threads > 0);
}
