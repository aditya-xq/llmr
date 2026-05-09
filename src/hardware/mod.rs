use crate::errors::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use sysinfo::System;
use tokio::time::timeout;
use tracing::info;

/// Indicates whether GPU memory is dedicated or shared/unified (e.g., Apple Silicon).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum GpuMemoryType {
    #[default]
    Dedicated,
    Shared,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuDevice {
    pub name: String,
    pub vram_mb: u64,
    pub vram_free_mb: u64,
    pub memory_type: GpuMemoryType,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub gpu: Option<GpuInfo>,
    pub ram: RamInfo,
    pub has_nvlink: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub cores: u32,
    pub threads: u32,
    pub name: String,
    pub architecture: String,
    pub frequency: Option<u64>,
}

impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            cores: 4,
            threads: 4,
            name: "Unknown".to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            frequency: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub names: Vec<String>,
    pub vram_mb: Vec<u64>,
    pub vram_free_mb: Vec<u64>,
    pub type_: String,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            names: vec!["Unknown GPU".to_string()],
            vram_mb: vec![0],
            vram_free_mb: vec![0],
            type_: "unknown".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamInfo {
    // total RAM in bytes (as reported by sysinfo)
    pub total: u64,
    pub total_gb: u32,
    pub free_gb: u32,
}

impl Default for RamInfo {
    fn default() -> Self {
        Self {
            // 8 GiB expressed in bytes for internal consistency
            total: 8 * 1024 * 1024 * 1024,
            total_gb: 8,
            free_gb: 4,
        }
    }
}

/// Common hardware info returned by platform-specific detection
#[derive(Debug, Default)]
pub struct PlatformHardwareInfo {
    pub cpu: CpuInfo,
    pub gpu: Option<Vec<GpuDevice>>,
    pub ram: RamInfo,
}

/// Detects hardware capabilities (CPU, GPU, RAM) on the current platform.
/// Runs platform detection and GPU detection concurrently.
/// Returns a `HardwareInfo` with detected specs or default values if detection fails.
pub async fn detect() -> Result<HardwareInfo> {
    info!("Detecting hardware...");

    let (info_result, has_nvlink) = tokio::join!(detect_platform(), detect_nvlink());

    let info = info_result?;

    #[cfg(target_os = "macos")]
    let gpu = info
        .gpu
        .as_ref()
        .map(|d| convert_gpu_devices(d, Some("apple")));
    #[cfg(not(target_os = "macos"))]
    let gpu = info
        .gpu
        .as_ref()
        .map(|d| convert_gpu_devices(d, None::<&str>));

    let gpu_count = gpu.as_ref().map_or(0, |g| g.names.len() as u32);

    info!(
        "Hardware detected: {} CPU cores ({} threads), {} GPU(s), {} GB RAM",
        info.cpu.cores, info.cpu.threads, gpu_count, info.ram.total_gb
    );

    if let Some(ref gpu_info) = gpu {
        let vram_total: u64 = gpu_info.vram_mb.iter().sum();
        for (i, name) in gpu_info.names.iter().enumerate() {
            let vram = gpu_info.vram_mb.get(i).copied().unwrap_or(0);
            info!("  GPU[{}]: {} - {}MB VRAM", i, name, vram);
        }
        info!("  Total VRAM: {}MB", vram_total);
    }

    Ok(HardwareInfo {
        cpu: info.cpu,
        gpu,
        ram: info.ram,
        has_nvlink,
    })
}

async fn detect_platform() -> Result<PlatformHardwareInfo> {
    let result = timeout(Duration::from_secs(5), async { detect_hardware().await }).await;

    match result {
        Ok(Ok(info)) => Ok(info),
        _ => Ok(PlatformHardwareInfo::default()),
    }
}

async fn detect_hardware() -> Result<PlatformHardwareInfo> {
    let mut system = System::new_all();
    let cpu = detect_cpu(&mut system);
    let ram = detect_ram(&system);
    let gpu = detect_gpu().await;

    Ok(PlatformHardwareInfo { cpu, gpu, ram })
}

fn detect_cpu(system: &mut System) -> CpuInfo {
    system.refresh_cpu_specifics(sysinfo::CpuRefreshKind::everything());

    let cpus = system.cpus();
    let threads = cpus.len() as u32;

    // Try to get physical core count
    let cores = system
        .physical_core_count()
        .map(|c| c as u32)
        .unwrap_or(threads);

    // Get CPU name from the first CPU
    let name = cpus
        .first()
        .map(|cpu| {
            let brand = cpu.brand();
            if !brand.is_empty() && brand != "Unknown" {
                brand.to_string()
            } else {
                cpu.vendor_id().to_string()
            }
        })
        .unwrap_or_else(|| "Unknown".to_string());

    let frequency = cpus
        .first()
        .map(|cpu| cpu.frequency())
        .filter(|&freq| freq > 0)
        .map(|freq| freq * 1_000_000); // Convert MHz to Hz

    CpuInfo {
        cores,
        threads,
        name,
        architecture: std::env::consts::ARCH.to_string(),
        frequency,
    }
}

fn detect_ram(system: &System) -> RamInfo {
    // sysinfo reports memory in bytes
    let total_bytes = system.total_memory();
    let available_bytes = system.available_memory();

    let total_gb = (total_bytes / 1024 / 1024 / 1024) as u32;
    let free_gb = (available_bytes / 1024 / 1024 / 1024) as u32;

    RamInfo {
        total: total_bytes,
        total_gb,
        free_gb,
    }
}

async fn detect_gpu() -> Option<Vec<GpuDevice>> {
    let mut devices = Vec::new();

    // Try NVIDIA detection (works cross-platform)
    if let Some(nvidia_devices) = detect_nvidia_gpu().await {
        devices.extend(nvidia_devices);
    }

    // Platform-specific detection for non-NVIDIA GPUs
    #[cfg(target_os = "linux")]
    {
        if let Some(amd_devices) = detect_amd_gpu_linux().await {
            devices.extend(amd_devices);
        }
        if let Some(intel_devices) = detect_intel_gpu_linux() {
            devices.extend(intel_devices);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(apple_devices) = detect_apple_gpu_macos().await {
            devices.extend(apple_devices);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(other_devices) = detect_gpu_windows_powershell().await {
            devices.extend(other_devices);
        }
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

async fn detect_nvidia_gpu() -> Option<Vec<GpuDevice>> {
    if which::which("nvidia-smi").is_err() {
        return None;
    }

    let output = match timeout(Duration::from_secs(5), async {
        tokio::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await
    })
    .await
    {
        Ok(Ok(output)) => output,
        _ => return None,
    };

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            let name = parts[0].to_string();
            let vram_mb = parts[1].parse::<u64>().unwrap_or(0);
            let vram_free_mb = parts[2].parse::<u64>().unwrap_or(0);

            devices.push(GpuDevice {
                name,
                vram_mb,
                vram_free_mb,
                memory_type: GpuMemoryType::Dedicated,
            });
        }
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

#[cfg(target_os = "linux")]
async fn detect_amd_gpu_linux() -> Option<Vec<GpuDevice>> {
    if which::which("rocm-smi").is_err() {
        return None;
    }

    let output = match timeout(Duration::from_secs(5), async {
        tokio::process::Command::new("rocm-smi")
            .args(["--showmeminfo", "vram", "--csv"])
            .output()
            .await
    })
    .await
    {
        Ok(Ok(output)) => output,
        _ => return None,
    };

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in stdout.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 2 {
            let name = parts[0].to_string();
            let vram_kb = parts[1].parse::<u64>().unwrap_or(0);
            let vram_mb = vram_kb / 1024;

            // rocm-smi --showmeminfo vram reports total VRAM only, not free
            devices.push(GpuDevice {
                name,
                vram_mb,
                vram_free_mb: 0,
                memory_type: GpuMemoryType::Dedicated,
            });
        }
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

#[cfg(target_os = "linux")]
fn detect_intel_gpu_linux() -> Option<Vec<GpuDevice>> {
    // Use lspci to check for Intel GPU vendor ID (0x8086)
    // instead of relying on /dev/dri which could be AMD or other vendors
    let output = std::process::Command::new("lspci")
        .arg("-vmm")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for section in stdout.split("\n\n") {
        if (section.contains("Class: VGA compatible controller")
            || section.contains("Class: 3D controller"))
            && section.contains("0x8086")
        {
            let name = section
                .lines()
                .find(|line| line.starts_with("Device:"))
                .and_then(|line| line.strip_prefix("Device:").map(|s| s.trim().to_string()))
                .unwrap_or_else(|| "Intel Graphics".to_string());

            return Some(vec![GpuDevice {
                name,
                vram_mb: 0,
                vram_free_mb: 0,
                memory_type: GpuMemoryType::Shared,
            }]);
        }
    }

    None
}

#[cfg(target_os = "macos")]
async fn detect_apple_gpu_macos() -> Option<Vec<GpuDevice>> {
    let output = match timeout(Duration::from_secs(5), async {
        tokio::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .await
    })
    .await
    {
        Ok(Ok(output)) => output,
        _ => return None,
    };

    if !output.status.success() {
        return None;
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let mut devices = Vec::new();

    if let Some(items) = json.as_array() {
        for item in items {
            if let Some(gpus) = item.get("spdisplays_ndrvs").and_then(|v| v.as_array()) {
                for gpu in gpus {
                    if let Some(name) = gpu.get("sppci_model").and_then(|v| v.as_str()) {
                        devices.push(GpuDevice {
                            name: name.to_string(),
                            vram_mb: gpu
                                .get("sppci_vram")
                                .and_then(|v| {
                                    v.as_str()?.split_whitespace().next()?.parse::<u64>().ok()
                                })
                                .unwrap_or(0),
                            vram_free_mb: 0,
                            memory_type: GpuMemoryType::Shared,
                        });
                    }
                }
            }
        }
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

#[cfg(target_os = "windows")]
async fn detect_gpu_windows_powershell() -> Option<Vec<GpuDevice>> {
    let output = match timeout(Duration::from_secs(5), async {
        tokio::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"
            ])
            .output()
            .await
    }).await {
        Ok(Ok(output)) => output,
        _ => return None,
    };

    if !output.status.success() {
        return None;
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let mut devices = Vec::new();

    let items = if json.is_array() {
        json.as_array().unwrap().clone()
    } else {
        vec![json]
    };

    for item in items {
        if let Some(name) = item.get("Name").and_then(|v| v.as_str()) {
            // Skip NVIDIA GPUs - they are already detected via nvidia-smi
            if name.to_lowercase().contains("nvidia") {
                continue;
            }

            let ram_bytes = item.get("AdapterRAM").and_then(|v| v.as_u64()).unwrap_or(0);
            let vram_mb = ram_bytes / 1024 / 1024;

            devices.push(GpuDevice {
                name: name.to_string(),
                vram_mb,
                vram_free_mb: 0,
                memory_type: GpuMemoryType::Dedicated,
            });
        }
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

async fn detect_nvlink() -> bool {
    #[cfg(target_os = "linux")]
    {
        timeout(Duration::from_secs(5), async {
            if which::which("nvidia-smi").is_err() {
                return false;
            }
            tokio::process::Command::new("nvidia-smi")
                .args(["nvlink", "--status"])
                .output()
                .await
                .ok()
                .map(|output| {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    stderr.contains("active")
                        || String::from_utf8_lossy(&output.stdout).contains("active")
                })
                .unwrap_or(false)
        })
        .await
        .unwrap_or(false)
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Converts a slice of `GpuDevice` into a `GpuInfo` struct.
/// If `custom_type` is provided, uses it as the GPU type; otherwise infers from device names.
pub fn convert_gpu_devices(devices: &[GpuDevice], custom_type: Option<&str>) -> GpuInfo {
    let names: Vec<String> = devices.iter().map(|d| d.name.clone()).collect();
    let vram_mb: Vec<_> = devices.iter().map(|d| d.vram_mb).collect();
    let vram_free_mb: Vec<_> = devices.iter().map(|d| d.vram_free_mb).collect();

    let type_ = custom_type.map(|s| s.to_string()).unwrap_or_else(|| {
        devices
            .first()
            .map(|d| {
                let name = d.name.to_lowercase();
                if name.contains("nvidia") {
                    "nvidia"
                } else if name.contains("amd") || name.contains("radeon") {
                    "amd"
                } else if name.contains("intel") {
                    "intel"
                } else {
                    "unknown"
                }
            })
            .unwrap_or("unknown")
            .to_string()
    });

    GpuInfo {
        names,
        vram_mb,
        vram_free_mb,
        type_,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
