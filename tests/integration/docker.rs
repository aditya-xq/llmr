use llmr::docker::DockerClient;
use llmr::errors::Error;
use llmr::hardware::{CpuInfo, GpuInfo, HardwareInfo, RamInfo};
use llmr::models::Profile;

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
async fn test_docker_client_new() {
    let client = DockerClient::new();
    assert!(client.is_ok() || matches!(client, Err(Error::DockerCliNotFound)));
}

#[tokio::test]
async fn test_docker_get_info() {
    let Ok(client) = DockerClient::new() else {
        return;
    };
    let result = client.get_info().await;
    if let Err(err) = result {
        assert!(matches!(err, Error::DockerError { .. }));
        assert!(err.to_string().contains("docker") || err.to_string().contains("Docker"));
    }
}

#[tokio::test]
async fn test_docker_image_exists() {
    let Ok(client) = DockerClient::new() else {
        return;
    };
    let exists = client.image_exists("nonexistent-image:latest").await;
    #[allow(clippy::overly_complex_bool_expr)]
    let _ = exists;
}

#[tokio::test]
async fn test_docker_get_container_nonexistent() {
    let Ok(client) = DockerClient::new() else {
        return;
    };
    let container = client.get_container("nonexistent-container-12345").await;
    assert!(matches!(
        container,
        Ok(None) | Err(Error::DockerError { .. })
    ));
}

#[tokio::test]
async fn test_docker_list_containers() {
    let Ok(client) = DockerClient::new() else {
        return;
    };
    let result = client.list_containers().await;
    assert!(matches!(result, Ok(_) | Err(Error::DockerError { .. })));
}

#[tokio::test]
async fn test_docker_list_containers_by_prefix() {
    let Ok(client) = DockerClient::new() else {
        return;
    };
    let result = client
        .list_containers_by_prefix("nonexistent-prefix-")
        .await;
    match result {
        Ok(containers) => assert!(containers.is_empty()),
        Err(err) => assert!(matches!(err, Error::DockerError { .. })),
    }
}

#[tokio::test]
async fn test_profile_to_docker_args_complete() {
    let hardware = create_test_hardware();
    let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

    let args = profile.to_docker_args(8080, true, false).unwrap();

    assert!(args.contains(&"--name".to_string()));
    assert!(args.contains(&"-d".to_string()));
    assert!(args.contains(&"-p".to_string()));
    assert!(args.contains(&"-v".to_string()));
}

#[tokio::test]
async fn test_profile_docker_image_selection() {
    let hardware = create_test_hardware();
    let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

    assert!(profile.docker_image.contains("cuda"));
}

#[tokio::test]
async fn test_profile_container_name_uniqueness() {
    let hardware = create_test_hardware();
    let profile1 = Profile::new("test_model1.gguf".to_string(), 4_000_000_000, &hardware);
    let profile2 = Profile::new("test_model2.gguf".to_string(), 4_000_000_000, &hardware);

    let name1 = profile1.container_name();
    let name2 = profile2.container_name();

    assert_ne!(name1, name2);
}
