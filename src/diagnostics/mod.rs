use crate::docker::{DockerClient, DockerDiagnostic};
use crate::hardware::HardwareInfo;
use crate::models::ProfileManager;
use crate::utils::{gpu_style, Style};
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::timeout;

#[derive(Debug, Clone)]
pub struct EnvDiagnostic {
    pub config_dir_exists: bool,
    pub config_dir: PathBuf,
}

impl Default for EnvDiagnostic {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvDiagnostic {
    pub fn new() -> Self {
        let config_dir = ProfileManager::config_dir();
        Self {
            config_dir_exists: config_dir.exists(),
            config_dir,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HardwareDiagnostic {
    pub from_cache: bool,
    pub hardware: Option<HardwareInfo>,
    pub error: Option<String>,
}

impl HardwareDiagnostic {
    fn from_err(msg: String) -> Self {
        Self {
            from_cache: false,
            hardware: None,
            error: Some(msg),
        }
    }
}

pub struct EnvCheck;

impl Default for EnvCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvCheck {
    pub fn new() -> Self {
        Self
    }

    pub fn check(&self) -> EnvDiagnostic {
        EnvDiagnostic::new()
    }

    pub fn run(&self, style: &Style) {
        print_env_diagnostic(style, &self.check());
    }
}

pub struct DockerCheck;

impl Default for DockerCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl DockerCheck {
    pub fn new() -> Self {
        Self
    }

    pub async fn check(&self) -> DockerDiagnostic {
        DockerClient::diagnose().await
    }
}

pub struct HardwareCheck;

impl Default for HardwareCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareCheck {
    pub fn new() -> Self {
        Self
    }

    pub async fn check(&self) -> HardwareDiagnostic {
        let profile_manager = ProfileManager::new();

        let load_result = timeout(Duration::from_secs(10), profile_manager.load_hardware()).await;

        match load_result {
            Ok(Ok(Some(hw))) => {
                return HardwareDiagnostic {
                    from_cache: true,
                    hardware: Some(hw),
                    error: None,
                };
            }
            Ok(Ok(None)) => {}
            Ok(Err(e)) => return HardwareDiagnostic::from_err(e.to_string()),
            Err(_) => {
                return HardwareDiagnostic::from_err("Loading hardware cache timed out".to_string())
            }
        }

        let detect_result = timeout(Duration::from_secs(30), crate::hardware::detect()).await;

        match detect_result {
            Ok(Ok(hw)) => {
                let _ = profile_manager.save_hardware(&hw).await;
                HardwareDiagnostic {
                    from_cache: false,
                    hardware: Some(hw),
                    error: None,
                }
            }
            Ok(Err(e)) => HardwareDiagnostic::from_err(e.to_string()),
            Err(_) => HardwareDiagnostic::from_err("Hardware detection timed out".to_string()),
        }
    }

    pub async fn run(&self, style: &Style) {
        print_hardware_diagnostic(style, &self.check().await);
    }
}

pub fn print_env_diagnostic(style: &Style, diag: &EnvDiagnostic) {
    println!();
    println!("  {} {}", style.info("→"), style.accent("Env"));

    if diag.config_dir_exists {
        println!(
            "    {} Config: {}",
            style.check(),
            style.muted(diag.config_dir.display().to_string())
        );
    } else {
        println!(
            "    {} Config: {} (will be created)",
            style.dash(),
            style.muted(diag.config_dir.display().to_string())
        );
    }
}

pub fn print_docker_diagnostic(style: &Style, diag: &DockerDiagnostic) {
    println!();
    println!("  {} {}", style.info("→"), style.accent("Docker"));

    if diag.daemon_running {
        println!("    {} Daemon running", style.check());
        if let Some(version) = &diag.server_version {
            println!("      v{}", version);
        }
    } else if diag.install_status.is_installed() {
        println!("    {} Installed but not running", style.warning("!"));
        if let Some(err) = &diag.error {
            println!("      {}", style.muted(err));
        }
    } else {
        println!("    {} Not installed", style.warning("!"));
        if let Some(err) = &diag.error {
            println!("      {}", style.muted(err));
        }
    }
}

pub fn print_hardware_diagnostic(style: &Style, diag: &HardwareDiagnostic) {
    println!();
    println!("  {} {}", style.info("→"), style.accent("Hardware"));

    match &diag.hardware {
        Some(hw) => {
            if diag.from_cache {
                println!("    {} Loaded from cache", style.info("→"));
            }

            println!(
                "    {} CPU: {} cores / {} threads · {}",
                style.check(),
                hw.cpu.cores,
                hw.cpu.threads,
                style.muted(&hw.cpu.architecture)
            );
            println!("    {} RAM: {:.0} GB", style.check(), hw.ram.total_gb);

            if let Some(gpu) = &hw.gpu {
                for (i, name) in gpu.names.iter().enumerate() {
                    let vram = gpu.vram_mb.get(i).copied().unwrap_or(0);
                    let colored = gpu_style(style, name);
                    println!(
                        "    {} GPU {}: {} · {}",
                        style.check(),
                        i,
                        colored,
                        style.vram(vram)
                    );
                }
            } else {
                println!("    {} No GPU detected (CPU mode)", style.dash());
            }
        }
        None => {
            println!("    {} Hardware detection failed", style.warning("!"));
            if let Some(err) = &diag.error {
                println!("      {}", style.muted(err));
            }
        }
    }
}

pub fn print_diagnostic_results(
    style: &Style,
    docker_diag: &DockerDiagnostic,
    hardware_diag: &HardwareDiagnostic,
) {
    print_docker_diagnostic(style, docker_diag);
    print_hardware_diagnostic(style, hardware_diag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_diagnostic_new() {
        let diag = EnvDiagnostic::new();
        assert!(diag.config_dir.to_string_lossy().contains("llmr"));
    }

    #[test]
    fn test_env_diagnostic_default() {
        let diag = EnvDiagnostic::default();
        assert!(diag.config_dir.to_string_lossy().contains("llmr"));
    }

    #[test]
    fn test_env_check() {
        let check = EnvCheck::new();
        let _ = check.check();
    }

    #[tokio::test]
    async fn test_docker_check() {
        let check = DockerCheck::new();
        let _ = check.check().await;
    }

    #[tokio::test]
    async fn test_docker_diagnostic_default() {
        let diag = DockerDiagnostic::default();
        assert!(!diag.available);
    }

    #[tokio::test]
    async fn test_hardware_check() {
        let check = HardwareCheck::new();
        let _ = check.check().await;
    }

    #[tokio::test]
    async fn test_hardware_diagnostic_default() {
        let diag = HardwareDiagnostic::default();
        assert!(!diag.from_cache);
    }
}
