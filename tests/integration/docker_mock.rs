use llmr::errors::Error;

#[tokio::test]
async fn test_docker_not_installed_handling() {
    use llmr::docker::DockerClient;

    let status = DockerClient::check_installed();
    match status {
        llmr::docker::DockerInstallStatus::NotInstalled => {
            let result = DockerClient::new();
            assert!(matches!(result, Err(Error::DockerCliNotFound)));
        }
        _ => {
            let result = DockerClient::new();
            assert!(result.is_ok());
        }
    }
}

#[test]
fn test_docker_install_status_not_installed() {
    use llmr::docker::DockerInstallStatus;

    let status = DockerInstallStatus::NotInstalled;
    assert!(!status.is_installed());
}

#[test]
fn test_docker_install_status_installed() {
    use llmr::docker::DockerInstallStatus;

    let status = DockerInstallStatus::Installed;
    assert!(status.is_installed());
}

#[test]
fn test_docker_install_status_not_running() {
    use llmr::docker::DockerInstallStatus;

    let status = DockerInstallStatus::InstalledButNotRunning;
    assert!(status.is_installed());
}

#[test]
fn test_docker_install_status_default() {
    use llmr::docker::DockerInstallStatus;

    let status = DockerInstallStatus::default();
    assert!(!status.is_installed());
}

#[test]
fn test_docker_install_status_equality() {
    use llmr::docker::DockerInstallStatus;

    assert_eq!(
        DockerInstallStatus::NotInstalled,
        DockerInstallStatus::NotInstalled
    );
    assert_eq!(
        DockerInstallStatus::Installed,
        DockerInstallStatus::Installed
    );
    assert_ne!(
        DockerInstallStatus::NotInstalled,
        DockerInstallStatus::Installed
    );
}

#[tokio::test]
async fn test_docker_diagnose_when_not_installed() {
    use llmr::docker::DockerClient;

    let status = DockerClient::check_installed();
    if !status.is_installed() {
        let result = DockerClient::new();
        assert!(matches!(result, Err(Error::DockerCliNotFound)));
    }
}

#[test]
fn test_docker_error_display() {
    let err = Error::DockerError {
        message: "test error".to_string(),
    };
    assert!(err.to_string().contains("test error"));
    assert!(err.to_string().contains("Docker"));
}

#[test]
fn test_docker_cli_not_found_error() {
    let err = Error::DockerCliNotFound;
    assert!(err.to_string().contains("CLI not found"));
}

#[tokio::test]
async fn test_profile_docker_image_selection_cpu() {
    use llmr::hardware::{CpuInfo, HardwareInfo, RamInfo};
    use llmr::models::Profile;

    let hardware = HardwareInfo {
        cpu: CpuInfo {
            cores: 8,
            threads: 16,
            name: "Test CPU".to_string(),
            architecture: "x86_64".to_string(),
            frequency: None,
        },
        gpu: None,
        ram: RamInfo {
            total: 32 * 1024 * 1024 * 1024,
            total_gb: 32,
            free_gb: 16,
        },
        has_nvlink: false,
    };
    let backend = llmr::tuning::Backend::LlamaCpp;
    let image = Profile::select_docker_image(&hardware.gpu, backend);
    assert!(image.contains("server"));
}

#[tokio::test]
async fn test_profile_docker_image_selection_nvidia() {
    use llmr::hardware::{CpuInfo, GpuInfo, HardwareInfo, RamInfo};
    use llmr::models::Profile;
    use llmr::tuning::Backend;

    let hardware = HardwareInfo {
        cpu: CpuInfo {
            cores: 8,
            threads: 16,
            name: "Test CPU".to_string(),
            architecture: "x86_64".to_string(),
            frequency: None,
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
    };

    let backend = Backend::LlamaCpp;
    let image = Profile::select_docker_image(&hardware.gpu, backend);
    assert!(image.contains("cuda"));
}

#[tokio::test]
async fn test_profile_docker_image_selection_amd() {
    use llmr::hardware::{CpuInfo, GpuInfo, HardwareInfo, RamInfo};
    use llmr::models::Profile;
    use llmr::tuning::Backend;

    let hardware = HardwareInfo {
        cpu: CpuInfo {
            cores: 8,
            threads: 16,
            name: "Test CPU".to_string(),
            architecture: "x86_64".to_string(),
            frequency: None,
        },
        gpu: Some(GpuInfo {
            names: vec!["AMD Radeon RX 6800".to_string()],
            vram_mb: vec![16384],
            vram_free_mb: vec![12000],
            type_: "amd".to_string(),
        }),
        ram: RamInfo {
            total: 32 * 1024 * 1024 * 1024,
            total_gb: 32,
            free_gb: 16,
        },
        has_nvlink: false,
    };

    let backend = Backend::LlamaCpp;
    let image = Profile::select_docker_image(&hardware.gpu, backend);
    assert!(image.contains("rocm") || image.contains("amd"));
}

#[test]
fn test_profile_container_name_format() {
    use llmr::models::Profile;
    use llmr::tuning::Backend;

    let profile = Profile {
        model_file: "test-model.gguf".to_string(),
        model_size_bytes: 4_000_000_000,
        cpu_cores: 8,
        gpu_count: 1,
        gpu_vram_total_mb: 10240,
        docker_image: "ghcr.io/ngxson/llama.cpp:b40".to_string(),
        backend: Backend::LlamaCpp,
        threads: 4,
        batch_size: 512,
        ubatch_size: 512,
        gpu_layers: 32,
        split_mode: "layer".to_string(),
        context_size: 2048,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        parallel_slots: 1,
        gpu_type: "nvidia".to_string(),
        has_nvlink: false,
        created_at: "2024-01-01".to_string(),
        best_tps: None,
    };

    let name = profile.container_name();
    assert!(name.starts_with("llmr_"));
    assert!(name.contains("test-model"));
}
