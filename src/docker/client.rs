use crate::errors::{Error, Result};
use crate::models::Profile;
use crate::utils::Style;
use std::process::Output;
use std::time::Duration;
use tokio::time::{timeout, Instant};
use tracing::info;
use tracing::warn;

const DOCKER_FORMAT: &str = "{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}";
const DOCKER_COMMAND_TIMEOUT_SECS: u64 = 30;
const DOCKER_PULL_TIMEOUT_SECS: u64 = 600;
const DOCKER_RUN_TIMEOUT_SECS: u64 = 60;
const HEALTH_CHECK_TIMEOUT_MS: u64 = 750;
const HEALTH_CHECK_FAST_INTERVAL_MS: u64 = 250;
const HEALTH_CHECK_SLOW_INTERVAL_MS: u64 = 1_000;
const HEALTH_CHECK_FAST_WINDOW_SECS: u64 = 10;
const HEALTH_CHECK_MAX_WAIT_SECS: u64 = 240;

#[derive(Debug, Clone, Default, PartialEq)]
pub enum DockerInstallStatus {
    #[default]
    NotInstalled,
    Installed,
    InstalledButNotRunning,
}

impl DockerInstallStatus {
    pub fn is_installed(&self) -> bool {
        !matches!(self, DockerInstallStatus::NotInstalled)
    }
}

struct ContainerConfig<'a> {
    name: &'a str,
    image: &'a str,
    model_dir: &'a str,
    port: u16,
    container_port: u16,
    profile: &'a Profile,
    enable_metrics: bool,
    public: bool,
    debug: bool,
    enable_gpu: bool,
}

pub struct DockerClient;

impl DockerClient {
    pub fn new() -> Result<Self> {
        if which::which("docker").is_err() {
            return Err(Error::DockerCliNotFound);
        }
        Ok(Self)
    }

    pub fn check_installed() -> DockerInstallStatus {
        if which::which("docker").is_ok() {
            DockerInstallStatus::Installed
        } else {
            DockerInstallStatus::NotInstalled
        }
    }

    pub async fn start_daemon() -> Result<()> {
        Self::start_daemon_internal().await
    }

    async fn start_daemon_internal() -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            let desktop_path = std::env::var("ProgramFiles")
                .map(|p| format!("{}\\Docker\\Docker\\Docker Desktop.exe", p))
                .ok();

            if let Some(path) = desktop_path {
                if std::path::Path::new(&path).exists() {
                    let output = tokio::process::Command::new(&path).spawn();

                    if output.is_ok() {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        return Ok(());
                    }
                }
            }

            let start_paths = [
                "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe",
                "C:\\Program Files (x86)\\Docker\\Docker\\Docker Desktop.exe",
            ];

            for path in start_paths.iter() {
                if std::path::Path::new(path).exists() {
                    let output = tokio::process::Command::new(path).spawn();

                    if output.is_ok() {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        return Ok(());
                    }
                }
            }

            let output = tokio::process::Command::new("powershell")
                .args(["-NoProfile", "-Command", "Start-Service docker"])
                .output()
                .await?;

            if output.status.success() {
                return Ok(());
            }

            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("NoServiceFoundForGivenName")
                || stderr.contains("Cannot find any service")
            {
                return Err(Error::DockerError {
                    message: "Docker Desktop is not running. Please start Docker Desktop from the Start menu or system tray.".to_string(),
                });
            }

            Err(Error::DockerError {
                message: format!("Failed to start Docker daemon: {}", stderr),
            })
        }

        #[cfg(target_os = "macos")]
        {
            let output = tokio::process::Command::new("open")
                .args(["-a", "Docker"])
                .output()
                .await?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(Error::DockerError {
                    message: format!("Failed to start Docker: {}", stderr),
                });
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
            return Ok(());
        }

        #[cfg(target_os = "linux")]
        {
            let dockerd_check = tokio::process::Command::new("which")
                .arg("dockerd")
                .output()
                .await?;

            if dockerd_check.status.success() {
                let output = tokio::process::Command::new("sudo")
                    .args(["service", "docker", "start"])
                    .output()
                    .await?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let output2 = tokio::process::Command::new("dockerd")
                        .arg("--version")
                        .output()
                        .await?;

                    if !output2.status.success() {
                        return Err(Error::DockerError {
                            message: format!("Failed to start Docker daemon: {}", stderr),
                        });
                    }
                }
                return Ok(());
            }

            let output = tokio::process::Command::new("systemctl")
                .args(["start", "docker"])
                .output()
                .await?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(Error::DockerError {
                    message: format!("Failed to start Docker daemon: {}", stderr),
                });
            }
            Ok(())
        }

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            Err(Error::DockerError {
                message: "Unsupported platform for auto-starting Docker".to_string(),
            })
        }
    }

    pub async fn get_info(&self) -> Result<DockerInfo> {
        let output = Self::docker_version().await?;
        Ok(DockerInfo {
            server_version: output,
        })
    }

    async fn docker_version() -> Result<String> {
        let output = Self::run_docker_command(
            ["version", "--format", "{{.Server.Version}}"],
            "Failed to query Docker version",
        )
        .await?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    pub async fn image_exists(&self, image: &str) -> bool {
        let mut command = tokio::process::Command::new("docker");
        command.args(["image", "inspect", image]);

        let output = Self::command_output_with_timeout(
            command,
            &format!("Failed to inspect Docker image '{}'", image),
            Duration::from_secs(DOCKER_COMMAND_TIMEOUT_SECS),
        )
        .await;

        match output {
            Ok(o) => o.status.success(),
            Err(_) => false,
        }
    }

    pub async fn pull_image(&self, image: &str) -> Result<()> {
        info!("Pulling Docker image: {}", image);

        Self::run_docker_command_with_timeout(
            ["pull", image],
            &format!("Failed to pull Docker image '{}'", image),
            Duration::from_secs(DOCKER_PULL_TIMEOUT_SECS),
        )
        .await?;

        info!("Image pulled successfully: {}", image);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "CLI container startup currently passes explicit request fields at this API boundary"
    )]
    pub async fn run_container(
        &self,
        name: &str,
        image: &str,
        model_path: &str,
        port: u16,
        profile: &Profile,
        enable_metrics: bool,
        public: bool,
        debug: bool,
    ) -> Result<()> {
        info!("Running container: {} with image: {}", name, image);

        let model_dir = std::path::Path::new(model_path)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or(".");

        let is_gpu_image = profile.uses_gpu_image();
        let config = ContainerConfig {
            name,
            image,
            model_dir,
            port,
            container_port: profile.container_port()?,
            profile,
            enable_metrics,
            public,
            debug,
            enable_gpu: is_gpu_image,
        };

        let output = self.run_container_once(&config).await?;

        if output.status.success() {
            info!("Container '{}' started", name);
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if is_gpu_image && Self::should_retry_without_gpu(&stderr) {
            warn!("GPU passthrough unavailable, retrying in CPU mode");
            let cpu_image = Self::cpu_fallback_image(image);
            let cpu_config = ContainerConfig {
                name,
                image: &cpu_image,
                model_dir,
                port,
                container_port: profile.container_port()?,
                profile,
                enable_metrics,
                public,
                debug,
                enable_gpu: false,
            };
            let fallback = self.run_container_once(&cpu_config).await?;

            if fallback.status.success() {
                info!("Container '{}' started in CPU mode", name);
                return Ok(());
            }

            let fallback_stderr = String::from_utf8_lossy(&fallback.stderr);
            return Err(Error::DockerError {
                message: format!("Failed to start container: {}", fallback_stderr.trim()),
            });
        }

        Err(Error::DockerError {
            message: format!("Failed to start container: {}", stderr),
        })
    }

    async fn run_container_once(
        &self,
        config: &ContainerConfig<'_>,
    ) -> Result<std::process::Output> {
        let mut cmd = tokio::process::Command::new("docker");
        cmd.args(["run", "--name", config.name, "-d"]);

        if config.public {
            cmd.args(["-p", &format!("{}:{}", config.port, config.container_port)]);
        } else {
            cmd.args([
                "-p",
                &format!("127.0.0.1:{}:{}", config.port, config.container_port),
            ]);
        }

        if config.enable_gpu {
            cmd.args(["--gpus", "all"]);
        }

        cmd.args(["-v", &format!("{}:/models:ro", config.model_dir)]);
        cmd.arg(config.image);
        cmd.args(
            config
                .profile
                .server_args_for_mode(config.enable_metrics, config.enable_gpu)?,
        );

        if !config.debug {
            cmd.arg("--log-disable");
        }

        Self::command_output_with_timeout(
            cmd,
            &format!("Failed to start container '{}'", config.name),
            Duration::from_secs(DOCKER_RUN_TIMEOUT_SECS),
        )
        .await
    }

    fn cpu_fallback_image(image: &str) -> String {
        let gpu_variants = [
            "server-cuda13",
            "server-cuda",
            "server-rocm",
            "server-vulkan",
            "server-intel",
        ];
        gpu_variants
            .iter()
            .fold(image.to_string(), |acc, &variant| {
                acc.replace(variant, "server")
            })
    }

    fn should_retry_without_gpu(stderr: &str) -> bool {
        let stderr = stderr.to_lowercase();
        stderr.contains("could not select device driver")
            || stderr.contains("capabilities: [[gpu]]")
            || stderr.contains("unknown runtime")
            || stderr.contains("nvidia-container-runtime")
            || stderr.contains("could not load plugin")
    }

    pub async fn get_container(&self, name: &str) -> Result<Option<ContainerInfo>> {
        let filter = format!("name={}", name);
        let output = Self::run_docker_command(
            ["ps", "-a", "--filter", &filter, "--format", DOCKER_FORMAT],
            &format!("Failed to inspect container '{}'", name),
        )
        .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
            let (id, container_name, image, status) = Self::parse_container_line(line, "ID")?;
            if container_name != name {
                continue;
            }

            let ports = self.get_container_ports(&container_name).await?;
            let created_at = self.get_container_created_at(&id).await?;
            return Ok(Some(ContainerInfo {
                id,
                name: container_name,
                image,
                status,
                ports,
                created_at,
            }));
        }

        Ok(None)
    }

    pub async fn container_exists(&self, name: &str) -> Result<bool> {
        let filter = format!("name=^/{}$", name);
        let output = Self::run_docker_command(
            ["ps", "-a", "--filter", &filter, "--format", "{{.Names}}"],
            &format!("Failed to inspect container '{}'", name),
        )
        .await?;

        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|container_name| container_name.trim() == name))
    }

    async fn get_container_ports(&self, name: &str) -> Result<Vec<(u16, String)>> {
        let output = Self::run_docker_command(
            ["port", name],
            &format!("Failed to inspect ports for container '{}'", name),
        )
        .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut ports = Vec::new();

        for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
            let (container_port, host_port) = Self::parse_port_mapping(line, name)?;
            ports.push((container_port, host_port));
        }

        Ok(ports)
    }

    async fn get_container_created_at(&self, id: &str) -> Result<chrono::DateTime<chrono::Utc>> {
        let output = Self::run_docker_command(
            ["inspect", "--format", "{{.Created}}", id],
            &format!("Failed to get creation time for container '{}'", id),
        )
        .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        chrono::DateTime::parse_from_rfc3339(stdout.trim())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .or_else(|_| {
                chrono::DateTime::parse_from_str(stdout.trim(), "%Y-%m-%d %H:%M:%S %z")
                    .map(|dt| dt.with_timezone(&chrono::Utc))
            })
            .map_err(|e| Error::DockerError {
                message: format!("Failed to parse container creation time: {}", e),
            })
    }

    pub async fn list_containers(&self) -> Result<Vec<ContainerInfo>> {
        self.list_containers_internal(None).await
    }

    pub async fn list_containers_by_prefix(&self, prefix: &str) -> Result<Vec<ContainerInfo>> {
        self.list_containers_internal(Some(prefix)).await
    }

    async fn list_containers_internal(&self, prefix: Option<&str>) -> Result<Vec<ContainerInfo>> {
        let filter = prefix.map(|p| format!("name={}", p));
        let mut args: Vec<&str> = vec!["ps", "--format", DOCKER_FORMAT];
        if let Some(ref f) = filter {
            args.push("--filter");
            args.push(f);
        }

        let output = Self::run_docker_command(args, "Failed to list Docker containers").await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut containers = Vec::new();

        for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
            let (id, name, image, status) = Self::parse_container_line(line, "ID")?;
            if prefix.is_some_and(|prefix| !name.starts_with(prefix)) {
                continue;
            }

            containers.push(ContainerInfo {
                id,
                name,
                image,
                status,
                ports: vec![],
                created_at: chrono::Utc::now(),
            });
        }

        Ok(containers)
    }

    pub async fn stop_container(&self, name: &str) -> Result<()> {
        info!("Stopping container: {}", name);

        Self::run_docker_command(
            ["stop", name],
            &format!("Failed to stop container '{}'", name),
        )
        .await?;

        info!("Container '{}' stopped", name);
        Ok(())
    }

    pub async fn remove_container(&self, name: &str) -> Result<()> {
        info!("Removing container: {}", name);

        Self::run_docker_command(
            ["rm", "-f", name],
            &format!("Failed to remove container '{}'", name),
        )
        .await?;

        info!("Container '{}' removed", name);
        Ok(())
    }

    pub async fn wait_for_health(&self, name: &str, port: u16) -> Result<()> {
        info!("Waiting for container '{}' to be healthy", name);

        let client = reqwest::Client::new();
        let started = Instant::now();
        let max_wait = Duration::from_secs(HEALTH_CHECK_MAX_WAIT_SECS);
        let mut last_log_second = 0;

        loop {
            if self.is_port_accessible(&client, port).await {
                info!("Container '{}' is healthy", name);
                return Ok(());
            }

            let elapsed = started.elapsed();
            if elapsed >= max_wait {
                break;
            }

            let elapsed_secs = elapsed.as_secs();
            if elapsed_secs >= last_log_second + 10 {
                last_log_second = elapsed_secs;
                info!(
                    "  Waiting... {}/{}s",
                    elapsed_secs, HEALTH_CHECK_MAX_WAIT_SECS
                );
            }

            let interval = if elapsed_secs < HEALTH_CHECK_FAST_WINDOW_SECS {
                Duration::from_millis(HEALTH_CHECK_FAST_INTERVAL_MS)
            } else {
                Duration::from_millis(HEALTH_CHECK_SLOW_INTERVAL_MS)
            };
            tokio::time::sleep(interval).await;
        }

        Err(Error::Timeout {
            message: format!(
                "Container '{}' did not become healthy within {} seconds",
                name, HEALTH_CHECK_MAX_WAIT_SECS
            ),
        })
    }

    async fn is_port_accessible(&self, client: &reqwest::Client, port: u16) -> bool {
        let url = format!("http://localhost:{}/health", port);
        match client
            .get(&url)
            .timeout(Duration::from_millis(HEALTH_CHECK_TIMEOUT_MS))
            .send()
            .await
        {
            Ok(resp) => {
                resp.status().is_success()
                    || resp.text().await.map(|t| t.contains("ok")).unwrap_or(false)
            }
            Err(_) => false,
        }
    }

    async fn run_docker_command(args: impl AsRef<[&str]>, context: &str) -> Result<Output> {
        Self::run_docker_command_with_timeout(
            args,
            context,
            Duration::from_secs(DOCKER_COMMAND_TIMEOUT_SECS),
        )
        .await
    }

    async fn run_docker_command_with_timeout(
        args: impl AsRef<[&str]>,
        context: &str,
        duration: Duration,
    ) -> Result<Output> {
        let mut command = tokio::process::Command::new("docker");
        command.args(args.as_ref());
        let output = Self::command_output_with_timeout(command, context, duration).await?;

        if !output.status.success() {
            return Err(Self::command_output_error(context, &output));
        }

        Ok(output)
    }

    async fn command_output_with_timeout(
        mut command: tokio::process::Command,
        context: &str,
        duration: Duration,
    ) -> Result<Output> {
        match timeout(duration, command.output()).await {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(err)) => Err(Self::command_spawn_error(context, err)),
            Err(_) => Err(Error::Timeout {
                message: format!("{} timed out after {} seconds", context, duration.as_secs()),
            }),
        }
    }

    fn command_spawn_error(context: &str, err: std::io::Error) -> Error {
        let err_msg = err.to_string().to_lowercase();
        let message = if err_msg.contains("no such file")
            || err_msg.contains("cannot find")
            || err_msg.contains("the system cannot find")
        {
            "Docker is not running. Ensure Docker Desktop is started.".to_string()
        } else {
            format!("{}: {}", context, err)
        };
        Error::DockerError { message }
    }

    fn command_output_error(context: &str, output: &Output) -> Error {
        let detail = Self::command_output_detail(output);
        let message = if detail.is_empty() {
            context.to_string()
        } else {
            format!("{}: {}", context, detail)
        };

        Error::DockerError { message }
    }

    fn command_output_detail(output: &Output) -> String {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();

        let combined = if !stderr.is_empty() {
            format!("{}\n{}", stderr, stdout)
        } else {
            stdout.clone()
        };

        let combined_lower = combined.to_lowercase();
        if combined_lower.contains("cannot find")
            || combined_lower.contains("no such file")
            || combined_lower.contains("daemon not running")
            || combined_lower.contains("docker daemon")
            || combined_lower.contains("connection refused")
            || combined_lower.contains("pipe")
            || combined_lower.contains("//./pipe/docker")
            || combined_lower.contains("failed to connect")
        {
            return "Docker is not running. Ensure Docker Desktop is started.".to_string();
        }

        if !stderr.is_empty() {
            return stderr;
        }

        stdout
    }

    fn parse_container_line(
        line: &str,
        field_context: &str,
    ) -> Result<(String, String, String, String)> {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 4 {
            return Err(Error::DockerError {
                message: format!(
                    "Failed to parse container {}: expected 4 fields but got {} from '{}'",
                    field_context,
                    parts.len(),
                    line
                ),
            });
        }

        Ok((
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
            parts[3].to_string(),
        ))
    }

    fn parse_port_mapping(line: &str, container_name: &str) -> Result<(u16, String)> {
        let Some((container_port, host_binding)) = line.split_once(" -> ") else {
            return Err(Error::DockerError {
                message: format!(
                    "Failed to parse ports for container '{}': unexpected docker output '{}'",
                    container_name, line
                ),
            });
        };

        let container_port = container_port
            .trim()
            .split('/')
            .next()
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| Error::DockerError {
                message: format!(
                    "Failed to parse ports for container '{}': unexpected docker output '{}'",
                    container_name, line
                ),
            })?;

        let host_port = host_binding
            .split(':')
            .next_back()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Error::DockerError {
                message: format!(
                    "Failed to parse ports for container '{}': unexpected docker output '{}'",
                    container_name, line
                ),
            })?
            .to_string();

        Ok((container_port, host_port))
    }
}

#[derive(Debug, Clone)]
pub struct DockerInfo {
    pub server_version: String,
}

#[derive(Debug, Clone, Default)]
pub struct DockerDiagnostic {
    pub available: bool,
    pub daemon_running: bool,
    pub server_version: Option<String>,
    pub error: Option<String>,
    pub install_status: DockerInstallStatus,
}

impl DockerClient {
    pub async fn diagnose() -> DockerDiagnostic {
        let install_status = Self::check_installed();

        if !install_status.is_installed() {
            return DockerDiagnostic {
                available: false,
                daemon_running: false,
                server_version: None,
                error: Some("Docker not installed".to_string()),
                install_status: DockerInstallStatus::NotInstalled,
            };
        }

        let client = match Self::new() {
            Ok(c) => c,
            Err(_) => {
                return DockerDiagnostic {
                    available: false,
                    daemon_running: false,
                    server_version: None,
                    error: Some("Docker CLI not found".to_string()),
                    install_status: DockerInstallStatus::Installed,
                };
            }
        };

        match timeout(Duration::from_secs(5), client.get_info()).await {
            Ok(Ok(info)) => DockerDiagnostic {
                available: true,
                daemon_running: true,
                server_version: Some(info.server_version),
                error: None,
                install_status: DockerInstallStatus::Installed,
            },
            Ok(Err(_)) => DockerDiagnostic {
                available: true,
                daemon_running: false,
                server_version: None,
                error: Some("Daemon not running".to_string()),
                install_status: DockerInstallStatus::Installed,
            },
            Err(_) => DockerDiagnostic {
                available: true,
                daemon_running: false,
                server_version: None,
                error: Some("Docker check timed out".to_string()),
                install_status: DockerInstallStatus::Installed,
            },
        }
    }

    pub fn print_diagnostic(style: &Style, diag: &DockerDiagnostic) {
        println!();
        println!("  {} {}", style.info("→"), style.accent("Docker"));

        if diag.daemon_running {
            println!("    {} Daemon running", style.check());
            if let Some(version) = &diag.server_version {
                println!("      v{}", version);
            }
        } else if diag.install_status == DockerInstallStatus::Installed {
            println!("    {} Installed but not running", style.warning("!"));
            if let Some(err) = &diag.error {
                println!("      {}", style.muted(err));
            }
        } else if diag.available {
            println!("    {} Failed to connect", style.warning("!"));
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
}

#[derive(Debug, Clone)]
pub struct ContainerInfo {
    pub id: String,
    pub name: String,
    pub image: String,
    pub status: String,
    pub ports: Vec<(u16, String)>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fallback_image_should_replace_gpu_tags() {
        assert_eq!(
            DockerClient::cpu_fallback_image("ghcr.io/ggml-org/llama.cpp:server-cuda13"),
            "ghcr.io/ggml-org/llama.cpp:server"
        );
        assert_eq!(
            DockerClient::cpu_fallback_image("ghcr.io/ggml-org/llama.cpp:server-rocm"),
            "ghcr.io/ggml-org/llama.cpp:server"
        );
    }

    #[test]
    fn parse_container_line_should_require_four_fields() {
        let parsed =
            DockerClient::parse_container_line("abc123|llama_test|image:tag|Up 1 second", "ID")
                .unwrap();

        assert_eq!(parsed.1, "llama_test");
        assert!(DockerClient::parse_container_line("abc123|llama_test", "ID").is_err());
    }

    #[test]
    fn parse_port_mapping_should_extract_container_and_host_ports() {
        let parsed =
            DockerClient::parse_port_mapping("8080/tcp -> 127.0.0.1:9090", "llama_test").unwrap();

        assert_eq!(parsed, (8080, "9090".to_string()));
    }

    #[test]
    fn should_retry_without_gpu_should_match_runtime_errors() {
        assert!(DockerClient::should_retry_without_gpu(
            "could not select device driver \"\" with capabilities: [[gpu]]"
        ));
        assert!(!DockerClient::should_retry_without_gpu(
            "port is already allocated"
        ));
    }
}
