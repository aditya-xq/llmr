use llmr::hardware::{
    convert_gpu_devices, detect, CpuInfo, GpuDevice, GpuInfo, GpuMemoryType, HardwareInfo, RamInfo,
};

#[test]
fn test_hardware_info_default() {
    let info = HardwareInfo::default();
    assert!(info.cpu.cores > 0);
}

#[test]
fn test_cpu_info_default() {
    let cpu = CpuInfo::default();
    assert!(cpu.cores > 0);
}

#[test]
fn test_gpu_info_default() {
    let gpu = GpuInfo::default();
    assert!(!gpu.names.is_empty());
}

#[test]
fn test_ram_info_default() {
    let ram = RamInfo::default();
    assert!(ram.total_gb > 0);
}

#[tokio::test]
async fn test_detect() {
    let info = detect().await.unwrap();
    assert!(info.cpu.cores > 0);
    assert!(info.cpu.threads > 0);
}

#[test]
fn test_default_config() {
    let config = HardwareInfo::default();
    assert_eq!(config.cpu.cores, 4);
    assert_eq!(config.cpu.threads, 4);
    assert!(config.gpu.is_none());
    assert_eq!(config.ram.total_gb, 8);
    assert_eq!(config.ram.free_gb, 4);
}

#[test]
fn test_convert_gpu_devices() {
    let devices = vec![
        GpuDevice {
            name: "NVIDIA RTX 3080".to_string(),
            vram_mb: 10240,
            vram_free_mb: 8000,
            memory_type: GpuMemoryType::Dedicated,
        },
        GpuDevice {
            name: "NVIDIA RTX 3090".to_string(),
            vram_mb: 24576,
            vram_free_mb: 20000,
            memory_type: GpuMemoryType::Dedicated,
        },
    ];
    let gpu_info = convert_gpu_devices(&devices, None);
    assert_eq!(gpu_info.names.len(), 2);
    assert_eq!(gpu_info.type_, "nvidia");
    assert_eq!(gpu_info.vram_mb.iter().sum::<u64>(), 34816);
}

#[test]
fn test_convert_gpu_devices_amd() {
    let devices = vec![GpuDevice {
        name: "AMD Radeon RX 6800".to_string(),
        vram_mb: 16384,
        vram_free_mb: 12000,
        memory_type: GpuMemoryType::Dedicated,
    }];
    let gpu_info = convert_gpu_devices(&devices, None);
    assert_eq!(gpu_info.type_, "amd");
}

#[test]
fn test_cpu_info_clone() {
    let cpu = CpuInfo {
        cores: 8,
        threads: 16,
        name: "Test CPU".to_string(),
        architecture: "x86_64".to_string(),
        frequency: Some(3600),
    };
    let cloned = cpu.clone();
    assert_eq!(cpu.cores, cloned.cores);
    assert_eq!(cpu.threads, cloned.threads);
}

#[test]
fn test_gpu_info_clone() {
    let gpu = GpuInfo {
        names: vec!["NVIDIA RTX 3080".to_string()],
        vram_mb: vec![10240],
        vram_free_mb: vec![8000],
        type_: "nvidia".to_string(),
    };
    let cloned = gpu.clone();
    assert_eq!(gpu.names, cloned.names);
    assert_eq!(gpu.names.len(), cloned.names.len());
}
