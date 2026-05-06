use crate::errors::{Error, Result};
use crate::hardware::HardwareInfo;
use crate::tuning::Backend;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tracing::{info, warn};

const FULL_GPU_OFFLOAD_LAYERS: i32 = 999;
const ESTIMATED_MODEL_LAYERS: f64 = 32.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub model_file: String,
    pub model_size_bytes: u64,
    pub cpu_cores: u32,
    pub gpu_count: u32,
    pub gpu_vram_total_mb: u64,
    pub docker_image: String,
    #[serde(default)]
    pub backend: Backend,
    pub threads: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub gpu_layers: i32,
    pub split_mode: String,
    pub context_size: u32,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub parallel_slots: u32,
    pub gpu_type: String,
    pub has_nvlink: bool,
    pub created_at: String,
    pub best_tps: Option<f64>,
}

impl Profile {
    pub fn new(model_file: String, model_size_bytes: u64, hardware: &HardwareInfo) -> Self {
        Self::with_backend(model_file, model_size_bytes, hardware, Backend::default())
    }

    pub fn with_backend(
        model_file: String,
        model_size_bytes: u64,
        hardware: &HardwareInfo,
        backend: Backend,
    ) -> Self {
        let cpu_cores = hardware.cpu.cores;
        let threads = Self::compute_threads(cpu_cores, hardware.cpu.threads);
        let batch_size = Self::compute_batch_size(&hardware.gpu);
        let ubatch_size = Self::compute_ubatch_size(batch_size);
        let gpu_layers = Self::compute_gpu_layers(&hardware.gpu, model_size_bytes);
        let split_mode = Self::compute_split_mode(&hardware.gpu, hardware.has_nvlink);
        let context_size = Self::compute_context_size(hardware.ram.free_gb);
        let (cache_type_k, cache_type_v) = Self::compute_cache_types(&hardware.gpu);
        let docker_image = Self::select_docker_image(&hardware.gpu, backend);
        let gpu_type = Self::hardware_gpu_type(hardware);

        Self {
            model_file,
            model_size_bytes,
            cpu_cores,
            gpu_count: hardware
                .gpu
                .as_ref()
                .map(|g| g.names.len() as u32)
                .unwrap_or(0),
            gpu_vram_total_mb: hardware
                .gpu
                .as_ref()
                .map(|g| g.vram_mb.iter().sum())
                .unwrap_or(0),
            docker_image,
            backend,
            threads,
            batch_size,
            ubatch_size,
            gpu_layers,
            split_mode,
            context_size,
            cache_type_k,
            cache_type_v,
            parallel_slots: 1,
            gpu_type,
            has_nvlink: hardware.has_nvlink,
            created_at: chrono::Utc::now().to_rfc3339(),
            best_tps: None,
        }
    }

    fn compute_threads(cpu_cores: u32, cpu_threads: u32) -> u32 {
        let available_threads = cpu_threads.max(1);
        let threads = (cpu_cores.max(1) * 9) / 10;
        threads.max(1).min(available_threads)
    }

    fn compute_batch_size(gpu: &Option<crate::hardware::GpuInfo>) -> u32 {
        match gpu {
            Some(g) if Self::supports_gpu_offload(&g.type_) => 2048,
            _ => 512,
        }
    }

    fn compute_ubatch_size(batch_size: u32) -> u32 {
        let ubatch = batch_size / 4;
        ubatch.clamp(64, 512)
    }

    #[allow(dead_code)]
    fn compute_gpu_layers(gpu: &Option<crate::hardware::GpuInfo>, model_size_bytes: u64) -> i32 {
        let Some(gpu) = gpu
            .as_ref()
            .filter(|gpu| Self::supports_gpu_offload(&gpu.type_))
        else {
            return 0;
        };

        Self::estimate_gpu_layers(gpu.vram_mb.iter().sum(), model_size_bytes)
    }

    #[allow(dead_code)]
    fn estimate_gpu_layers(total_vram_mb: u64, model_size_bytes: u64) -> i32 {
        if total_vram_mb == 0 || model_size_bytes == 0 {
            return FULL_GPU_OFFLOAD_LAYERS;
        }

        let model_mb = model_size_bytes as f64 / 1_048_576.0;
        let usable_vram = total_vram_mb as f64 * 0.9;

        if model_mb <= usable_vram {
            FULL_GPU_OFFLOAD_LAYERS
        } else {
            let layers = (usable_vram * ESTIMATED_MODEL_LAYERS / model_mb).ceil() as i32;
            layers.max(1)
        }
    }

    fn compute_split_mode(gpu: &Option<crate::hardware::GpuInfo>, has_nvlink: bool) -> String {
        match gpu {
            Some(g) if g.names.len() > 1 => {
                if has_nvlink {
                    "row".to_string()
                } else {
                    "layer".to_string()
                }
            }
            _ => "none".to_string(),
        }
    }

    #[allow(dead_code)]
    fn compute_context_size(free_ram_gb: u32) -> u32 {
        if free_ram_gb >= 32 {
            65536
        } else if free_ram_gb >= 16 {
            32768
        } else if free_ram_gb >= 8 {
            8192
        } else {
            4096
        }
    }

    fn compute_cache_types(gpu: &Option<crate::hardware::GpuInfo>) -> (String, String) {
        match gpu {
            Some(g) if Self::supports_gpu_offload(&g.type_) => {
                ("q4_0".to_string(), "q4_0".to_string())
            }
            _ => ("f16".to_string(), "f16".to_string()),
        }
    }

    pub fn select_docker_image(gpu: &Option<crate::hardware::GpuInfo>, backend: Backend) -> String {
        let registry = backend.docker_registry();
        match backend {
            Backend::LlamaCpp => Self::select_llama_cpp_image(gpu, registry),
            Backend::Vllm => Self::select_vllm_image(gpu, registry),
            Backend::Sglang => Self::select_sglang_image(gpu, registry),
        }
    }

    fn select_llama_cpp_image(gpu: &Option<crate::hardware::GpuInfo>, registry: &str) -> String {
        let base = format!("{}/llama.cpp", registry);
        match gpu {
            Some(g) if g.type_ == "nvidia" => {
                let driver_ver = Self::get_nvidia_driver_version();
                if driver_ver >= 550 {
                    format!("{}:server-cuda13", base)
                } else {
                    format!("{}:server-cuda", base)
                }
            }
            Some(g) if g.type_ == "amd" => format!("{}:server-rocm", base),
            Some(g) if g.type_ == "intel" => format!("{}:server-intel", base),
            Some(g) if g.type_ == "vulkan" => format!("{}:server-vulkan", base),
            _ => format!("{}:server", base),
        }
    }

    fn select_vllm_image(gpu: &Option<crate::hardware::GpuInfo>, registry: &str) -> String {
        let base = format!("{}/vllm", registry);
        match gpu {
            Some(g) if g.type_ == "nvidia" => {
                let driver_ver = Self::get_nvidia_driver_version();
                if driver_ver >= 550 {
                    format!("{}:latest-cuda13", base)
                } else {
                    format!("{}:latest-cuda12", base)
                }
            }
            Some(g) if g.type_ == "amd" => format!("{}:latest-rocm", base),
            _ => format!("{}:latest", base),
        }
    }

    fn select_sglang_image(gpu: &Option<crate::hardware::GpuInfo>, registry: &str) -> String {
        let base = format!("{}/sglang", registry);
        match gpu {
            Some(g) if g.type_ == "nvidia" => {
                let driver_ver = Self::get_nvidia_driver_version();
                if driver_ver >= 550 {
                    format!("{}:latest-cuda13", base)
                } else {
                    format!("{}:latest-cuda12", base)
                }
            }
            Some(g) if g.type_ == "amd" => format!("{}:latest-rocm", base),
            _ => format!("{}:latest", base),
        }
    }

    fn get_nvidia_driver_version() -> u32 {
        let output = match StdCommand::new("nvidia-smi")
            .args(["--query-gpu=driver_version", "--format=csv,noheader"])
            .output()
        {
            Ok(o) => o,
            Err(_) => return 0,
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let first_line = match stdout.lines().next() {
            Some(l) => l,
            None => return 0,
        };
        let version = match first_line.split('.').next() {
            Some(v) => v.trim(),
            None => return 0,
        };
        version.parse().unwrap_or(0)
    }

    fn hardware_gpu_type(hardware: &HardwareInfo) -> String {
        hardware
            .gpu
            .as_ref()
            .map(|g| g.type_.clone())
            .unwrap_or_else(|| "none".to_string())
    }

    fn supports_gpu_offload(gpu_type: &str) -> bool {
        matches!(gpu_type, "nvidia" | "amd" | "intel" | "vulkan")
    }

    pub fn uses_gpu_image(&self) -> bool {
        Self::is_gpu_image(&self.docker_image)
    }

    pub(crate) fn is_gpu_image(image: &str) -> bool {
        let tag = image.rsplit(':').next().unwrap_or(image);
        matches!(
            tag,
            "server-cuda13" | "server-cuda" | "server-rocm" | "server-intel" | "server-vulkan"
        )
    }

    fn sanitize_identifier(input: &str, max_len: usize, fallback: &str) -> String {
        let sanitized: String = input
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .take(max_len)
            .collect();

        if sanitized.is_empty() {
            fallback.to_string()
        } else {
            sanitized
        }
    }

    pub fn key(&self) -> String {
        self.generate_key()
    }

    pub fn container_name(&self) -> String {
        let safe_model = Path::new(&self.model_file)
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("model");
        let sanitized = Self::sanitize_identifier(safe_model, 40, "model");
        format!("llmr_{}", sanitized)
    }

    pub fn generate_key(&self) -> String {
        let model_basename = Path::new(&self.model_file)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&self.model_file);
        let sanitized = Self::sanitize_identifier(model_basename, 64, "model");
        format!(
            "{sanitized}__cpu{}_gpu{}_{}mb",
            self.cpu_cores, self.gpu_count, self.gpu_vram_total_mb
        )
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_file.is_empty() {
            return Err(Error::InvalidProfile {
                reason: "model_file is empty".to_string(),
            });
        }
        if self.cpu_cores == 0 {
            return Err(Error::InvalidProfile {
                reason: "cpu_cores is zero".to_string(),
            });
        }
        Ok(())
    }

    pub fn to_docker_args(
        &self,
        port: u16,
        enable_metrics: bool,
        public: bool,
    ) -> Result<Vec<String>> {
        self.ensure_backend_supported()?;
        let mut args = Vec::new();
        let container_port = self.container_port()?;

        args.push("--name".to_string());
        args.push(self.container_name());
        args.push("-d".to_string());

        if public {
            args.push("-p".to_string());
            args.push(format!("{}:{}", port, container_port));
        } else {
            args.push("-p".to_string());
            args.push(format!("127.0.0.1:{}:{}", port, container_port));
        }

        if self.uses_gpu_image() {
            args.push("--gpus".to_string());
            args.push("all".to_string());
        }

        let model_dir = Path::new(&self.model_file)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or(".");

        args.push("-v".to_string());
        args.push(format!("{}:/models:ro", model_dir));

        args.push(self.docker_image.clone());

        args.extend(self.server_args(enable_metrics)?);

        Ok(args)
    }

    pub fn container_port(&self) -> Result<u16> {
        self.ensure_backend_supported()?;
        match self.backend {
            Backend::LlamaCpp => Ok(8080),
            Backend::Vllm | Backend::Sglang => Err(Error::InvalidProfile {
                reason: self.backend.unsupported_message(),
            }),
        }
    }

    pub fn server_args(&self, enable_metrics: bool) -> Result<Vec<String>> {
        self.ensure_backend_supported()?;
        match self.backend {
            Backend::LlamaCpp => Ok(self.llama_server_args(enable_metrics)),
            Backend::Vllm | Backend::Sglang => Err(Error::InvalidProfile {
                reason: self.backend.unsupported_message(),
            }),
        }
    }

    pub fn server_args_for_mode(
        &self,
        enable_metrics: bool,
        enable_gpu: bool,
    ) -> Result<Vec<String>> {
        let mut server_args = self.server_args(enable_metrics)?;
        if enable_gpu {
            return Ok(server_args);
        }

        if let Some(pos) = server_args.iter().position(|arg| arg == "--n-gpu-layers") {
            server_args.remove(pos);
            server_args.remove(pos);
        }
        server_args.extend(["--n-gpu-layers".to_string(), "0".to_string()]);
        Ok(server_args)
    }

    fn ensure_backend_supported(&self) -> Result<()> {
        if self.backend.supports_serving() {
            return Ok(());
        }

        Err(Error::InvalidProfile {
            reason: self.backend.unsupported_message(),
        })
    }

    pub fn llama_server_args(&self, enable_metrics: bool) -> Vec<String> {
        let mut args = Vec::new();

        args.push("-m".to_string());
        args.push(format!(
            "/models/{}",
            Path::new(&self.model_file)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model.gguf")
        ));
        args.push("--host".to_string());
        args.push("0.0.0.0".to_string());
        args.push("--port".to_string());
        args.push("8080".to_string());
        args.push("-t".to_string());
        args.push(self.threads.to_string());
        args.push("--threads-batch".to_string());
        args.push(self.threads.to_string());
        args.push("-b".to_string());
        args.push(self.batch_size.to_string());
        args.push("--ubatch-size".to_string());
        args.push(self.ubatch_size.to_string());
        args.push("--parallel".to_string());
        args.push(self.parallel_slots.to_string());
        args.push("--cont-batching".to_string());
        args.push("--cache-type-k".to_string());
        args.push(self.cache_type_k.clone());
        args.push("--cache-type-v".to_string());
        args.push(self.cache_type_v.clone());

        if self.context_size > 0 {
            args.push("-c".to_string());
            args.push(self.context_size.to_string());
        }

        if self.gpu_layers > 0 && Self::supports_gpu_offload(&self.gpu_type) {
            args.push("--n-gpu-layers".to_string());
            args.push(self.gpu_layers.to_string());
            if self.gpu_count > 1 && self.split_mode != "none" && self.split_mode != "auto" {
                args.push("--split-mode".to_string());
                args.push(self.split_mode.clone());
                args.push("--main-gpu".to_string());
                args.push("0".to_string());
            }
        }

        if enable_metrics {
            args.push("--metrics".to_string());
        }

        args
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCache {
    pub model_paths: Vec<String>,
    pub scan_timestamp: String,
    pub scanned_folders: Vec<String>,
}

pub struct ProfileManager {
    config_dir: PathBuf,
}

impl ProfileManager {
    pub fn new() -> Self {
        let config_dir = Self::config_dir();
        std::fs::create_dir_all(&config_dir).ok();
        Self { config_dir }
    }

    pub fn config_dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("llmr")
    }

    pub fn new_with_dir(config_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&config_dir).ok();
        Self { config_dir }
    }

    pub async fn load_or_create(
        &self,
        model_path: &str,
        hardware: &HardwareInfo,
    ) -> Result<Profile> {
        let model_size = Self::get_file_size(model_path)?;
        let key = self.generate_key(model_path, hardware);

        if let Some(profile) = self.load(&key).await? {
            if profile.gpu_type != Profile::hardware_gpu_type(hardware) {
                warn!("Cached GPU type differs, recomputing profile");
                let profile = Profile::new(model_path.to_string(), model_size, hardware);
                self.save(&profile).await?;
                return Ok(profile);
            }
            info!("Loaded existing profile: {}", key);
            Ok(profile)
        } else {
            info!("Creating new profile: {}", key);
            let profile = Profile::new(model_path.to_string(), model_size, hardware);
            self.save(&profile).await?;
            Ok(profile)
        }
    }

    fn get_file_size(path: &str) -> Result<u64> {
        std::fs::metadata(path)
            .map(|m| m.len())
            .map_err(Error::from)
    }

    pub async fn load(&self, key: &str) -> Result<Option<Profile>> {
        let profile_path = self.profile_path(key);

        if !profile_path.exists() {
            return Ok(None);
        }

        let content = tokio::fs::read_to_string(&profile_path).await?;
        let profile: Profile = toml::from_str(&content)?;

        Ok(Some(profile))
    }

    pub async fn save(&self, profile: &Profile) -> Result<()> {
        let key = profile.generate_key();
        let profile_path = self.profile_path(&key);

        let content = toml::to_string_pretty(profile)?;
        tokio::fs::write(&profile_path, content).await?;

        info!("Profile saved: {}", key);
        Ok(())
    }

    fn profile_path(&self, key: &str) -> PathBuf {
        self.config_dir.join(format!("{}.toml", key))
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        let profile_path = self.profile_path(key);

        if profile_path.exists() {
            tokio::fs::remove_file(&profile_path).await?;
            info!("Profile deleted: {}", key);
        }

        Ok(())
    }

    pub async fn clear_all(&self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.config_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let is_profile = path.extension().is_some_and(|ext| ext == "toml")
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| stem != "hardware");
            if is_profile {
                tokio::fs::remove_file(path).await.ok();
            }
        }

        info!("All profiles cleared");
        Ok(())
    }

    fn hardware_cache_path(&self) -> PathBuf {
        self.config_dir.join("hardware.toml")
    }

    pub async fn save_hardware(&self, hardware: &HardwareInfo) -> Result<()> {
        let cache_path = self.hardware_cache_path();
        let content = toml::to_string_pretty(hardware)?;
        tokio::fs::write(&cache_path, content).await?;
        info!("Hardware info cached");
        Ok(())
    }

    pub async fn load_hardware(&self) -> Result<Option<HardwareInfo>> {
        let cache_path = self.hardware_cache_path();
        if !cache_path.exists() {
            return Ok(None);
        }
        let content = tokio::fs::read_to_string(&cache_path).await?;
        let hardware: HardwareInfo = toml::from_str(&content)?;
        Ok(Some(hardware))
    }

    fn model_cache_path(&self) -> PathBuf {
        self.config_dir.join("models.toml")
    }

    pub async fn save_model_cache(&self, model_paths: &[String]) -> Result<()> {
        let cache_path = self.model_cache_path();
        let folders: Vec<String> = model_paths
            .iter()
            .filter_map(|p| {
                std::path::Path::new(p)
                    .parent()
                    .and_then(|p| p.to_str())
                    .map(String::from)
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let cache = ModelCache {
            model_paths: model_paths.to_vec(),
            scan_timestamp: chrono::Utc::now().to_rfc3339(),
            scanned_folders: folders,
        };
        let content = toml::to_string_pretty(&cache)?;
        tokio::fs::write(&cache_path, content).await?;
        info!("Model cache saved with {} models", model_paths.len());
        Ok(())
    }

    pub async fn load_model_cache(&self) -> Result<Option<ModelCache>> {
        let cache_path = self.model_cache_path();
        if !cache_path.exists() {
            return Ok(None);
        }
        let content = tokio::fs::read_to_string(&cache_path).await?;
        let cache: ModelCache = toml::from_str(&content)?;
        Ok(Some(cache))
    }

    pub fn get_cached_model_folders(&self) -> Vec<PathBuf> {
        let cache_path = self.model_cache_path();
        if !cache_path.exists() {
            return Vec::new();
        }
        if let Ok(content) = std::fs::read_to_string(&cache_path) {
            if let Ok(cache) = toml::from_str::<ModelCache>(&content) {
                return cache
                    .scanned_folders
                    .into_iter()
                    .map(PathBuf::from)
                    .filter(|p| p.exists())
                    .collect();
            }
        }
        Vec::new()
    }

    pub async fn get_model_cache(&self) -> Result<Option<ModelCache>> {
        self.load_model_cache().await
    }

    pub async fn list_all(&self) -> Result<Vec<(String, Profile)>> {
        let mut profiles = Vec::new();
        let hardware_cache = "hardware";
        let model_cache = "models";

        let mut entries = tokio::fs::read_dir(&self.config_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension().is_some_and(|ext| ext == "toml") {
                let stem = entry
                    .file_name()
                    .to_string_lossy()
                    .trim_end_matches(".toml")
                    .to_string();
                if stem == hardware_cache || stem == model_cache {
                    continue;
                }
                if let Some(profile) = self.load(&stem).await? {
                    profiles.push((stem, profile));
                }
            }
        }

        profiles.sort_by(|a, b| a.1.created_at.cmp(&b.1.created_at));
        Ok(profiles)
    }

    pub async fn recompute_profile(
        &self,
        model_path: &str,
        hardware: &HardwareInfo,
    ) -> Result<Profile> {
        let model_size = Self::get_file_size(model_path)?;
        let mut profile = Profile::new(model_path.to_string(), model_size, hardware);

        if let Some(gpu) = &hardware.gpu {
            if Profile::supports_gpu_offload(&gpu.type_) {
                profile.gpu_layers =
                    Profile::estimate_gpu_layers(gpu.vram_mb.iter().sum(), model_size);
            }
        }

        self.save(&profile).await?;
        info!("Profile recomputed: {}", profile.generate_key());
        Ok(profile)
    }

    pub fn find_free_port(start: u16) -> u16 {
        for port in start..start + 50 {
            if std::net::TcpListener::bind(format!("127.0.0.1:{}", port)).is_ok() {
                return port;
            }
        }
        start
    }

    fn generate_key(&self, model_path: &str, hardware: &HardwareInfo) -> String {
        let model_basename = Path::new(model_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let gpu_count = hardware
            .gpu
            .as_ref()
            .map(|g| g.names.len() as u32)
            .unwrap_or(0);
        let gpu_vram = hardware
            .gpu
            .as_ref()
            .map(|g| g.vram_mb.iter().sum())
            .unwrap_or(0);
        let sanitized = Profile::sanitize_identifier(model_basename, 64, "model");
        format!(
            "{sanitized}__cpu{}_gpu{}_{}mb",
            hardware.cpu.cores, gpu_count, gpu_vram
        )
    }

    pub async fn profile_exists(&self, model_path: &str, hardware: &HardwareInfo) -> Result<bool> {
        let key = self.generate_key(model_path, hardware);
        Ok(self.load(&key).await?.is_some())
    }

    pub async fn find_profile_key(&self, model_path: &str) -> Result<Option<String>> {
        let model_basename = Path::new(model_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let sanitized = Profile::sanitize_identifier(model_basename, 64, "model");

        let mut entries = tokio::fs::read_dir(&self.config_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension().is_some_and(|ext| ext == "toml") {
                let entry_path = entry.path();
                let Some(key) = entry_path.file_stem().and_then(|stem| stem.to_str()) else {
                    continue;
                };
                if key == "hardware" {
                    continue;
                }
                let expected_prefix = format!("{sanitized}__");
                if key.starts_with(&expected_prefix) {
                    return Ok(Some(key.to_string()));
                }
            }
        }
        Ok(None)
    }
}

impl Default for ProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{CpuInfo, GpuInfo, HardwareInfo, RamInfo};
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

    fn create_cpu_only_hardware() -> HardwareInfo {
        HardwareInfo {
            cpu: CpuInfo {
                cores: 4,
                threads: 8,
                name: "Test CPU".to_string(),
                architecture: "x86_64".to_string(),
                frequency: Some(3600),
            },
            gpu: None,
            ram: RamInfo {
                total: 16 * 1024 * 1024 * 1024,
                total_gb: 16,
                free_gb: 8,
            },
            has_nvlink: false,
        }
    }

    fn create_multi_gpu_hardware() -> HardwareInfo {
        HardwareInfo {
            cpu: CpuInfo {
                cores: 16,
                threads: 32,
                name: "Test CPU".to_string(),
                architecture: "x86_64".to_string(),
                frequency: Some(3600),
            },
            gpu: Some(GpuInfo {
                names: vec!["NVIDIA GPU 0".to_string(), "NVIDIA GPU 1".to_string()],
                vram_mb: vec![24000, 24000],
                vram_free_mb: vec![20000, 20000],
                type_: "nvidia".to_string(),
            }),
            ram: RamInfo {
                total: 64 * 1024 * 1024 * 1024,
                total_gb: 64,
                free_gb: 32,
            },
            has_nvlink: true,
        }
    }

    fn create_low_ram_hardware() -> HardwareInfo {
        HardwareInfo {
            cpu: CpuInfo {
                cores: 2,
                threads: 4,
                name: "Test CPU".to_string(),
                architecture: "x86_64".to_string(),
                frequency: Some(2000),
            },
            gpu: None,
            ram: RamInfo {
                total: 4 * 1024 * 1024 * 1024,
                total_gb: 4,
                free_gb: 2,
            },
            has_nvlink: false,
        }
    }

    #[test]
    fn test_profile_new() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        assert_eq!(profile.model_file, "test_model.gguf");
        assert_eq!(profile.model_size_bytes, 4_000_000_000);
        assert_eq!(profile.cpu_cores, 8);
        assert_eq!(profile.gpu_count, 1);
        assert_eq!(profile.gpu_vram_total_mb, 10240);
        assert!(profile.threads > 0);
        assert!(profile.batch_size > 0);
        assert!(profile.context_size > 0);
    }

    #[test]
    fn test_profile_compute_threads() {
        let _hardware = create_test_hardware();

        let result = Profile::compute_threads(4, 8);
        assert_eq!(result, 3);

        let result = Profile::compute_threads(1, 2);
        assert_eq!(result, 1);

        let result = Profile::compute_threads(10, 20);
        assert_eq!(result, 9);

        let result = Profile::compute_threads(100, 200);
        assert_eq!(result, 90);

        let result = Profile::compute_threads(0, 0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_profile_compute_batch_size() {
        let nvidia_gpu = Some(GpuInfo {
            names: vec!["NVIDIA RTX 3080".to_string()],
            vram_mb: vec![10240],
            vram_free_mb: vec![8000],
            type_: "nvidia".to_string(),
        });

        let amd_gpu = Some(GpuInfo {
            names: vec!["AMD RX 6800".to_string()],
            vram_mb: vec![16384],
            vram_free_mb: vec![14000],
            type_: "amd".to_string(),
        });

        let no_gpu: Option<GpuInfo> = None;

        let apple_gpu = Some(GpuInfo {
            names: vec!["Apple M1".to_string()],
            vram_mb: vec![16000],
            vram_free_mb: vec![12000],
            type_: "apple".to_string(),
        });
        let unknown_gpu = Some(GpuInfo {
            names: vec!["Unknown GPU".to_string()],
            vram_mb: vec![10240],
            vram_free_mb: vec![8000],
            type_: "unknown".to_string(),
        });

        assert_eq!(Profile::compute_batch_size(&nvidia_gpu), 2048);
        assert_eq!(Profile::compute_batch_size(&amd_gpu), 2048);
        assert_eq!(Profile::compute_batch_size(&no_gpu), 512);
        assert_eq!(Profile::compute_batch_size(&apple_gpu), 512);
        assert_eq!(Profile::compute_batch_size(&unknown_gpu), 512);
    }

    #[test]
    fn test_profile_compute_ubatch_size() {
        assert_eq!(Profile::compute_ubatch_size(2048), 512);
        assert_eq!(Profile::compute_ubatch_size(512), 128);
        assert_eq!(Profile::compute_ubatch_size(1024), 256);
        assert_eq!(Profile::compute_ubatch_size(100), 64);
        assert_eq!(Profile::compute_ubatch_size(10000), 512);
    }

    #[test]
    fn test_profile_compute_gpu_layers() {
        let large_gpu = Some(GpuInfo {
            names: vec!["NVIDIA RTX 3090".to_string()],
            vram_mb: vec![24000],
            vram_free_mb: vec![20000],
            type_: "nvidia".to_string(),
        });

        let small_gpu = Some(GpuInfo {
            names: vec!["NVIDIA GTX 1660".to_string()],
            vram_mb: vec![6000],
            vram_free_mb: vec![5000],
            type_: "nvidia".to_string(),
        });

        let no_gpu: Option<GpuInfo> = None;
        let unknown_gpu = Some(GpuInfo {
            names: vec!["Unknown GPU".to_string()],
            vram_mb: vec![12000],
            vram_free_mb: vec![10000],
            type_: "unknown".to_string(),
        });

        assert_eq!(Profile::compute_gpu_layers(&large_gpu, 4_000_000_000), 999);
        assert!(Profile::compute_gpu_layers(&small_gpu, 10_000_000_000) > 0);
        assert!(Profile::compute_gpu_layers(&small_gpu, 10_000_000_000) < 32);
        assert_eq!(Profile::compute_gpu_layers(&no_gpu, 4_000_000_000), 0);
        assert_eq!(Profile::compute_gpu_layers(&unknown_gpu, 4_000_000_000), 0);
    }

    #[test]
    fn test_profile_compute_split_mode() {
        let multi_gpu = Some(GpuInfo {
            names: vec!["NVIDIA GPU 0".to_string(), "NVIDIA GPU 1".to_string()],
            vram_mb: vec![24000, 24000],
            vram_free_mb: vec![20000, 20000],
            type_: "nvidia".to_string(),
        });

        let single_gpu: Option<GpuInfo> = None;

        let result_with_nvlink = Profile::compute_split_mode(&multi_gpu, true);
        assert_eq!(result_with_nvlink, "row");

        let result_without_nvlink = Profile::compute_split_mode(&multi_gpu, false);
        assert_eq!(result_without_nvlink, "layer");

        let result_no_gpu = Profile::compute_split_mode(&single_gpu, false);
        assert_eq!(result_no_gpu, "none");
    }

    #[test]
    fn test_profile_compute_context_size() {
        assert_eq!(Profile::compute_context_size(64), 65536);
        assert_eq!(Profile::compute_context_size(32), 65536);
        assert_eq!(Profile::compute_context_size(16), 32768);
        assert_eq!(Profile::compute_context_size(8), 8192);
        assert_eq!(Profile::compute_context_size(4), 4096);
    }

    #[test]
    fn test_profile_compute_cache_types() {
        let nvidia_gpu = Some(GpuInfo {
            names: vec!["NVIDIA RTX 3080".to_string()],
            vram_mb: vec![10240],
            vram_free_mb: vec![8000],
            type_: "nvidia".to_string(),
        });

        let no_gpu: Option<GpuInfo> = None;

        let (k, v) = Profile::compute_cache_types(&nvidia_gpu);
        assert_eq!(k, "q4_0");
        assert_eq!(v, "q4_0");

        let (k, v) = Profile::compute_cache_types(&no_gpu);
        assert_eq!(k, "f16");
        assert_eq!(v, "f16");
    }

    #[test]
    fn test_profile_select_docker_image() {
        let nvidia_gpu = Some(GpuInfo {
            names: vec!["NVIDIA RTX 3080".to_string()],
            vram_mb: vec![10240],
            vram_free_mb: vec![8000],
            type_: "nvidia".to_string(),
        });

        let amd_gpu = Some(GpuInfo {
            names: vec!["AMD RX 6800".to_string()],
            vram_mb: vec![16384],
            vram_free_mb: vec![14000],
            type_: "amd".to_string(),
        });

        let intel_gpu = Some(GpuInfo {
            names: vec!["Intel Iris".to_string()],
            vram_mb: vec![4000],
            vram_free_mb: vec![3000],
            type_: "intel".to_string(),
        });

        let vulkan_gpu = Some(GpuInfo {
            names: vec!["AMD RX 6800".to_string()],
            vram_mb: vec![16384],
            vram_free_mb: vec![14000],
            type_: "vulkan".to_string(),
        });

        let no_gpu: Option<GpuInfo> = None;
        let backend = Backend::LlamaCpp;

        let nvidia_img = Profile::select_docker_image(&nvidia_gpu, backend);
        assert!(nvidia_img.contains("cuda"));

        let amd_img = Profile::select_docker_image(&amd_gpu, backend);
        assert!(amd_img.contains("rocm"));

        let intel_img = Profile::select_docker_image(&intel_gpu, backend);
        assert!(intel_img.contains("intel"));

        let vulkan_img = Profile::select_docker_image(&vulkan_gpu, backend);
        assert!(vulkan_img.contains("vulkan"));

        let cpu_img = Profile::select_docker_image(&no_gpu, backend);
        assert!(cpu_img.contains("server"));
        assert!(!cpu_img.contains("cuda"));
    }

    #[test]
    fn test_profile_is_gpu_image_recognizes_supported_llama_cpp_tags() {
        assert!(Profile::is_gpu_image(
            "ghcr.io/ggml-org/llama.cpp:server-cuda13"
        ));
        assert!(Profile::is_gpu_image(
            "ghcr.io/ggml-org/llama.cpp:server-cuda"
        ));
        assert!(Profile::is_gpu_image(
            "ghcr.io/ggml-org/llama.cpp:server-rocm"
        ));
        assert!(Profile::is_gpu_image(
            "ghcr.io/ggml-org/llama.cpp:server-vulkan"
        ));
        assert!(!Profile::is_gpu_image("ghcr.io/ggml-org/llama.cpp:server"));
    }

    #[test]
    fn test_profile_key_generation() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let key = profile.key();
        assert!(key.contains("test_model"));
        assert!(key.contains("cpu8"));
        assert!(key.contains("gpu1"));
    }

    #[test]
    fn test_profile_container_name() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let name = profile.container_name();
        assert!(name.starts_with("llmr_"));
        assert!(name.len() <= 50);
    }

    #[test]
    fn test_profile_container_name_special_chars() {
        let hardware = create_test_hardware();
        let profile = Profile::new(
            "model with spaces @#$.gguf".to_string(),
            4_000_000_000,
            &hardware,
        );

        let name = profile.container_name();
        assert!(name.starts_with("llmr_"));
        assert!(name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn test_profile_container_name_fallback_when_sanitized_empty() {
        let hardware = create_test_hardware();
        let profile = Profile::new("@#$%.gguf".to_string(), 4_000_000_000, &hardware);

        assert_eq!(profile.container_name(), "llmr_model");
    }

    #[test]
    fn test_profile_generate_key() {
        let hardware = create_test_hardware();
        let profile = Profile::new("mymodel.gguf".to_string(), 4_000_000_000, &hardware);

        let key = profile.generate_key();
        assert!(key.contains("mymodel"));
        assert!(key.contains("cpu8"));
    }

    #[test]
    fn test_profile_validate() {
        let hardware = create_test_hardware();
        let mut profile = Profile::new("test.gguf".to_string(), 4_000_000_000, &hardware);

        assert!(profile.validate().is_ok());

        profile.model_file = "".to_string();
        assert!(profile.validate().is_err());

        let hardware2 = create_cpu_only_hardware();
        let mut profile2 = Profile::new("test.gguf".to_string(), 4_000_000_000, &hardware2);
        profile2.cpu_cores = 0;
        assert!(profile2.validate().is_err());
    }

    #[test]
    fn test_profile_to_docker_args() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let args = profile.to_docker_args(8080, true, false).unwrap();

        assert!(args.contains(&"--name".to_string()));
        assert!(args.contains(&"-d".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"127.0.0.1:8080:8080".to_string()));
    }

    #[test]
    fn test_profile_to_docker_args_includes_gpu_passthrough_for_gpu_image() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let args = profile.to_docker_args(8080, false, false).unwrap();

        assert!(args.windows(2).any(|pair| pair == ["--gpus", "all"]));
    }

    #[test]
    fn test_profile_to_docker_args_public() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let args = profile.to_docker_args(9000, false, true).unwrap();

        assert!(args.contains(&"9000:8080".to_string()));
    }

    #[test]
    fn test_profile_llama_server_args() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let args = profile.llama_server_args(false);

        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"--host".to_string()));
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8080".to_string()));
        assert!(args.contains(&"-t".to_string()));
        assert!(args.contains(&"-b".to_string()));
        assert!(args.contains(&"--ubatch-size".to_string()));
        assert!(args.contains(&"--parallel".to_string()));
        assert!(args.contains(&"--cont-batching".to_string()));
    }

    #[test]
    fn test_planned_backend_server_args_are_explicitly_unsupported() {
        let hardware = create_test_hardware();
        let profile = Profile::with_backend(
            "test_model.gguf".to_string(),
            4_000_000_000,
            &hardware,
            Backend::Vllm,
        );

        let err = profile.to_docker_args(8080, false, false).unwrap_err();
        assert!(err.to_string().contains("planned"));
        assert!(err.to_string().contains("llama.cpp"));
    }

    #[test]
    fn test_profile_llama_server_args_with_metrics() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let args = profile.llama_server_args(true);

        assert!(args.contains(&"--metrics".to_string()));
    }

    #[test]
    fn test_profile_llama_server_args_gpu_layers() {
        let hardware = create_test_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        let args = profile.llama_server_args(false);

        assert!(args.contains(&"--n-gpu-layers".to_string()));
    }

    #[test]
    fn test_profile_cpu_only() {
        let hardware = create_cpu_only_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        assert_eq!(profile.gpu_count, 0);
        assert_eq!(profile.gpu_layers, 0);
        assert!(profile.gpu_type == "none" || profile.gpu_type.is_empty());
    }

    #[test]
    fn test_profile_multi_gpu() {
        let hardware = create_multi_gpu_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        assert_eq!(profile.gpu_count, 2);
        assert!(profile.split_mode == "row" || profile.split_mode == "layer");
    }

    #[test]
    fn test_profile_low_ram() {
        let hardware = create_low_ram_hardware();
        let profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);

        assert!(profile.context_size <= 4096);
    }

    #[test]
    fn test_profile_context_size_zero() {
        let hardware = create_low_ram_hardware();
        let mut profile = Profile::new("test_model.gguf".to_string(), 4_000_000_000, &hardware);
        profile.context_size = 0;

        let args = profile.llama_server_args(false);
        assert!(!args.contains(&"-c".to_string()));
    }

    #[test]
    fn test_profile_manager_new() {
        let _pm = ProfileManager::new();
        let config_dir = ProfileManager::config_dir();
        assert!(config_dir.to_string_lossy().contains("llmr"));
    }

    #[test]
    fn test_profile_manager_with_temp_dir() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let profile_path = pm.profile_path("test_key");
        assert!(profile_path.to_string_lossy().contains("test_key.toml"));
    }

    #[tokio::test]
    async fn test_profile_manager_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let hardware = create_test_hardware();
        let profile = Profile::new("test.gguf".to_string(), 1_000_000_000, &hardware);

        pm.save(&profile).await.unwrap();

        let loaded = pm.load(&profile.key()).await.unwrap();
        assert!(loaded.is_some());

        let loaded_profile = loaded.unwrap();
        assert_eq!(loaded_profile.model_file, "test.gguf");
        assert_eq!(loaded_profile.threads, profile.threads);
    }

    #[tokio::test]
    async fn test_profile_manager_load_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let loaded = pm.load("nonexistent_key").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_profile_manager_delete() {
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
    async fn test_profile_manager_list_all() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let hardware = create_test_hardware();
        let profile1 = Profile::new("model1.gguf".to_string(), 1_000_000_000, &hardware);
        let profile2 = Profile::new("model2.gguf".to_string(), 2_000_000_000, &hardware);

        pm.save(&profile1).await.unwrap();
        pm.save(&profile2).await.unwrap();

        let profiles = pm.list_all().await.unwrap();
        assert_eq!(profiles.len(), 2);
    }

    #[tokio::test]
    async fn test_profile_manager_list_all_excludes_model_cache() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let hardware = create_test_hardware();
        let profile = Profile::new("model1.gguf".to_string(), 1_000_000_000, &hardware);
        pm.save(&profile).await.unwrap();

        let model_cache = ModelCache {
            model_paths: vec!["test.gguf".to_string()],
            scan_timestamp: "2026-05-02T00:00:00Z".to_string(),
            scanned_folders: vec!["Z:\\AI\\llms".to_string()],
        };
        let cache_path = temp_dir.path().join("models.toml");
        let content = toml::to_string_pretty(&model_cache).unwrap();
        tokio::fs::write(&cache_path, content).await.unwrap();

        let profiles = pm.list_all().await.unwrap();
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0].1.model_file.contains("model1.gguf"));
    }

    #[tokio::test]
    async fn test_profile_manager_clear_all() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let hardware = create_test_hardware();
        let profile1 = Profile::new("model1.gguf".to_string(), 1_000_000_000, &hardware);
        let profile2 = Profile::new("model2.gguf".to_string(), 2_000_000_000, &hardware);

        pm.save(&profile1).await.unwrap();
        pm.save(&profile2).await.unwrap();

        pm.clear_all().await.unwrap();

        let profiles = pm.list_all().await.unwrap();
        assert!(profiles.is_empty());
    }

    #[tokio::test]
    async fn test_profile_manager_clear_all_preserves_hardware_cache() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let hardware = create_test_hardware();
        let profile = Profile::new("model.gguf".to_string(), 1_000_000_000, &hardware);
        pm.save(&profile).await.unwrap();
        pm.save_hardware(&hardware).await.unwrap();

        pm.clear_all().await.unwrap();

        assert!(pm.list_all().await.unwrap().is_empty());
        assert!(pm.load_hardware().await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_profile_manager_load_or_create_new() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        // Create a dummy model file
        let model_path = temp_dir.path().join("new_model.gguf");
        std::fs::write(&model_path, "dummy").unwrap();

        let hardware = create_test_hardware();

        let profile = pm
            .load_or_create(model_path.to_str().unwrap(), &hardware)
            .await
            .unwrap();

        assert!(profile.model_file.contains("new_model.gguf"));
    }

    #[tokio::test]
    async fn test_profile_manager_load_or_create_existing() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        // Create a dummy model file
        let model_path = temp_dir.path().join("existing_model.gguf");
        std::fs::write(&model_path, "dummy").unwrap();

        let hardware = create_test_hardware();
        let profile1 = pm
            .load_or_create(model_path.to_str().unwrap(), &hardware)
            .await
            .unwrap();

        let profile2 = pm
            .load_or_create(model_path.to_str().unwrap(), &hardware)
            .await
            .unwrap();

        assert_eq!(profile1.threads, profile2.threads);
    }

    #[tokio::test]
    async fn test_profile_manager_find_profile_key() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        // Create a model file
        let model_path = temp_dir.path().join("test_model.gguf");
        std::fs::write(&model_path, "dummy").unwrap();

        let hardware = create_test_hardware();
        let profile = Profile::new(
            model_path.to_str().unwrap().to_string(),
            1_000_000_000,
            &hardware,
        );

        pm.save(&profile).await.unwrap();

        let key = pm.find_profile_key("test_model.gguf").await.unwrap();
        assert!(key.is_some());
    }

    #[tokio::test]
    async fn test_profile_manager_find_profile_key_requires_key_separator() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let hardware = create_test_hardware();
        let colliding_profile =
            Profile::new("model.gguf-extra".to_string(), 1_000_000_000, &hardware);
        pm.save(&colliding_profile).await.unwrap();

        let key = pm.find_profile_key("model.gguf").await.unwrap();

        assert!(key.is_none());
    }

    #[tokio::test]
    async fn test_profile_manager_recompute() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        // Create a model file
        let model_path = temp_dir.path().join("test_model.gguf");
        std::fs::write(&model_path, "dummy").unwrap();

        let hardware = create_test_hardware();
        let profile = pm
            .recompute_profile(model_path.to_str().unwrap(), &hardware)
            .await
            .unwrap();

        assert!(profile.model_file.contains("test_model.gguf"));
        assert!(profile.threads > 0);
    }

    #[test]
    fn test_find_free_port() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let free_port = ProfileManager::find_free_port(port);
        assert_ne!(free_port, port);
    }

    #[test]
    fn test_generate_key() {
        let temp_dir = TempDir::new().unwrap();
        let pm = ProfileManager::new_with_dir(temp_dir.path().to_path_buf());

        let hardware = create_test_hardware();
        let key = pm.generate_key("test_model.gguf", &hardware);

        assert!(key.contains("test_model"));
        assert!(key.contains("cpu8"));
    }
}
