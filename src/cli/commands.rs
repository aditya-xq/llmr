use super::args::*;
use crate::bench::config;
use crate::bench::types::BenchmarkConfig;
use crate::diagnostics::{print_diagnostic_results, DockerCheck, EnvCheck, HardwareCheck};
use crate::docker::DockerClient;
use crate::errors::{Error, Result};
use crate::hardware::HardwareInfo;
use crate::models::{ModelScanner, Profile, ProfileManager};
use crate::tuning::{
    Backend, GgufExtractor, GgufFactsExtractor, HardwareProfile, LlamaCppProfile, OptimizeError,
    OptimizeOptions,
};
use crate::utils::Style;
use std::io::Write;
use std::path::Path;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

async fn ensure_docker(style: &Style) -> Result<DockerClient> {
    let install_status = DockerClient::check_installed();

    if !install_status.is_installed() {
        return Err(Error::Other {
            message: "Docker is not installed. Please install Docker to continue.".to_string(),
        });
    }

    let diag = DockerClient::diagnose().await;

    if diag.daemon_running {
        return DockerClient::new();
    }

    println!(
        "  {} Docker installed but daemon not running, attempting to start...",
        style.info("→")
    );
    std::io::stdout().flush()?;

    if let Err(e) = DockerClient::start_daemon().await {
        return Err(Error::Other {
            message: format!(
                "Docker is installed but not running. Please start Docker manually: {}",
                e
            ),
        });
    }

    println!(
        "  {} Waiting for Docker daemon to be ready...",
        style.info("→")
    );
    std::io::stdout().flush()?;

    for _ in 0..30 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let diag = DockerClient::diagnose().await;
        if diag.daemon_running {
            println!("  {} Docker daemon is ready", style.success("✓"));
            return DockerClient::new();
        }
    }

    Err(Error::Other {
        message: "Docker daemon did not start in time. Please start Docker manually.".to_string(),
    })
}

pub struct ServeCommand {
    args: ServeArgs,
    style: Style,
}

impl ServeCommand {
    pub fn new(args: ServeArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let style = &self.style;

        let model_path: String = if let Some(model) = &self.args.model {
            self.validate_model_path(model)?
        } else {
            match self.interactive_model_select().await? {
                Some(p) => p,
                None => {
                    println!("{}", style.error("No model selected"));
                    return Ok(());
                }
            }
        };

        info!("Starting llmr server");

        let hardware = if !self.args.skip_hardware && !self.args.no_gpu {
            crate::hardware::detect().await?
        } else {
            HardwareInfo::default()
        };

        let profile_manager = ProfileManager::new();
        let is_new_profile = !profile_manager
            .profile_exists(&model_path, &hardware)
            .await?;
        let mut docker_client = None;

        let mut profile = if is_new_profile
            && !self.args.dry_run
            && !self.args.no_benchmark
            && !self.args.quick
        {
            let run_tuning = if self.args.benchmark {
                true
            } else {
                println!();
                println!("  {} First run for this model", style.info("→"));
                print!(
                    "  {} Would you like to tune for optimal performance? [Y/n]: ",
                    style.info("→")
                );
                std::io::stdout().flush()?;
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let answer = input.trim().to_lowercase();
                answer.is_empty() || answer == "y" || answer == "yes"
            };

            if run_tuning {
                docker_client = Some(ensure_docker(style).await?);
                println!("  {} Running tuning optimization...", style.info("→"));
                std::io::stdout().flush()?;

                let extractor = GgufExtractor;
                let facts = match extractor.extract(Path::new(&model_path)) {
                    Ok(f) => {
                        println!("    Architecture: {}", f.architecture);
                        if let Some(ctx) = f.context_length {
                            println!("    Context: {}", ctx);
                        }
                        f
                    }
                    Err(e) => {
                        println!(
                            "  {} Could not parse GGUF metadata: {}",
                            style.warning("!"),
                            e
                        );
                        println!("  {} Proceeding with default tuning", style.info("→"));
                        crate::tuning::GgufFacts {
                            path: Path::new(&model_path).to_path_buf(),
                            architecture: "llama".to_string(),
                            model_name: None,
                            size_label: None,
                            quantization_version: None,
                            file_type: None,
                            alignment: None,
                            context_length: Some(4096),
                            embedding_length: None,
                            block_count: Some(32),
                            feed_forward_length: None,
                            attention_head_count: None,
                            attention_head_count_kv: None,
                            rope_dimension_count: None,
                            rope_scaling_type: None,
                            rope_scaling_factor: None,
                            rope_scaling_original_context_length: None,
                            chat_template: None,
                            tensor_count: 0,
                            weight_bytes: 0,
                        }
                    }
                };

                let hardware_profile: HardwareProfile = hardware.clone().into();

                let opts = OptimizeOptions {
                    benchmark_ctx_size: Some(facts.context_length.unwrap_or(4096).min(4096)),
                    prompt_tokens: 512,
                    generation_tokens: 128,
                    parallel_requests: 1,
                    max_rounds: 4,
                    min_relative_gain: 0.01,
                    warmup_samples: 2,
                    benchmark_samples: 5,
                    early_stop_samples: 2,
                    racing_keep_fraction: 0.25,
                    coarse_budget_ms: 500,
                    fine_budget_ms: 3000,
                    search_strategy: crate::tuning::SearchStrategy::Racing,
                };

                let backend = Backend::ACTIVE;
                let docker_image = Profile::select_docker_image(&hardware.gpu, backend);
                let model_path_for_tune = model_path.clone();

                let tune_result = crate::tuning::optimize_llama_cpp_profile(
                    &model_path,
                    hardware_profile,
                    &extractor,
                    move |llama_profile| {
                        let model_path = model_path_for_tune.clone();
                        let docker_image = docker_image.clone();
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                run_benchmark(
                                    &model_path,
                                    &docker_image,
                                    llama_profile,
                                    opts.warmup_samples,
                                    opts.benchmark_samples,
                                    opts.prompt_tokens,
                                    opts.generation_tokens,
                                )
                                .await
                            })
                        })
                    },
                    opts,
                )
                .map_err(|e| {
                    println!("  {} Tuning failed: {}", style.warning("!"), e);
                    Error::Other {
                        message: e.to_string(),
                    }
                })?;

                let tuned_profile =
                    convert_to_profile(&tune_result.profile, &model_path, &hardware);
                let stats = tune_result.stats;
                profile_manager.save(&tuned_profile).await?;
                println!("{}  Tuning complete - profile saved", style.success("✓"));
                println!(
                    "    Tested: {} candidates | Cache: {} entries",
                    stats.candidates_tested, stats.cache_size
                );
                if let Some(best_result) = tune_result.profile.estimated_result.as_ref() {
                    println!("    Result: prompt_tps={:.1} | decode_tps={:.1} | latency={:.0}ms | stability={:.0}%",
                        best_result.prompt_tps,
                        best_result.decode_tps,
                        best_result.latency_ms,
                        best_result.stability * 100.0);
                }
                tuned_profile
            } else {
                let base = profile_manager
                    .load_or_create(&model_path, &hardware)
                    .await?;
                println!("  {} Using default configuration", style.info("→"));
                base
            }
        } else if is_new_profile {
            println!("  {} Creating default profile", style.info("→"));
            profile_manager
                .load_or_create(&model_path, &hardware)
                .await?
        } else {
            profile_manager
                .load_or_create(&model_path, &hardware)
                .await?
        };

        self.apply_serve_overrides(&mut profile);

        if self.args.dry_run {
            self.print_docker_command(&profile)?;
            return Ok(());
        }

        let docker_client = match docker_client {
            Some(client) => client,
            None => ensure_docker(style).await?,
        };

        println!("  {} Starting container...", style.info("→"));
        std::io::stdout().flush()?;
        self.run_container(&docker_client, &model_path, &profile)
            .await?;

        println!();
        println!("  {} Server ready", style.success("✓"));
        println!(
            "    {}",
            style.accent(&format!(
                "http://localhost:{}/v1/chat/completions",
                self.args.port
            ))
        );
        if self.args.metrics {
            println!(
                "    {} Metrics at http://localhost:{}/metrics",
                style.info("→"),
                self.args.port
            );
        }
        println!(
            "    {} Health at http://localhost:{}/health",
            style.info("→"),
            self.args.port
        );
        println!(
            "  {} Stop with: {}",
            style.info("→"),
            style.muted("llmr stop")
        );

        Ok(())
    }

    fn apply_serve_overrides(&self, profile: &mut Profile) {
        if let Some(threads) = self.args.threads.filter(|threads| *threads > 0) {
            profile.threads = threads;
        }
        if let Some(ctx_size) = self.args.ctx_size.filter(|ctx_size| *ctx_size > 0) {
            profile.context_size = ctx_size;
        }
        if let Some(batch_size) = self.args.batch_size.filter(|batch_size| *batch_size > 0) {
            profile.batch_size = batch_size;
        }
        if let Some(ubatch_size) = self.args.ubatch_size.filter(|ubatch_size| *ubatch_size > 0) {
            profile.ubatch_size = ubatch_size;
        }
        if let Some(parallel) = self.args.parallel.filter(|parallel| *parallel > 0) {
            profile.parallel_slots = parallel;
        }
        if let Some(cache_type_k) = self
            .args
            .cache_type_k
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            profile.cache_type_k = cache_type_k.clone();
        }
        if let Some(cache_type_v) = self
            .args
            .cache_type_v
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            profile.cache_type_v = cache_type_v.clone();
        }

        if self.args.no_gpu {
            profile.gpu_layers = 0;
            profile.split_mode = "none".to_string();
            return;
        }

        if let Some(gpu_layers) = self.args.gpu_layers {
            profile.gpu_layers = i32::try_from(gpu_layers).unwrap_or(i32::MAX);
        }
        if let Some(split_mode) = self.args.split_mode {
            profile.split_mode = split_mode.as_str().to_string();
        }
    }

    fn validate_model_path(&self, model_path: &str) -> Result<String> {
        let path = std::path::Path::new(model_path);
        if !path.exists() {
            return Err(Error::ModelNotFound {
                path: model_path.to_string(),
            })?;
        }
        if !path.is_file() {
            return Err(Error::InvalidModelPath {
                path: model_path.to_string(),
            })?;
        }

        Ok(model_path.to_string())
    }

    async fn interactive_model_select(&self) -> Result<Option<String>> {
        let style = &self.style;
        let scanner = ModelScanner::new();
        let profile_manager = ProfileManager::new();

        println!();
        println!("  {}", style.title("Searching for GGUF models..."));
        println!();

        let cached_folders = profile_manager.get_cached_model_folders();
        let quick_scan_results = if !cached_folders.is_empty() {
            println!("  {} Checking cached locations...", style.info("→"));
            scanner.scan_paths(&cached_folders)
        } else {
            Vec::new()
        };

        let all = if !quick_scan_results.is_empty() {
            quick_scan_results
        } else {
            println!("  {} Scanning disks for GGUF files...", style.info("→"));
            scanner.scan_disks()
        };

        let model_paths: Vec<String> = all
            .iter()
            .map(|m| m.path.to_string_lossy().to_string())
            .collect();
        if !model_paths.is_empty() {
            let _ = profile_manager.save_model_cache(&model_paths).await;
        }

        if all.is_empty() {
            println!(
                "  {} No GGUF models found on local disks",
                style.warning("!")
            );
            println!();
            println!("  {} Common model locations checked:", style.info("→"));
            for root in ModelScanner::find_root_paths() {
                let root_str = root.to_string_lossy();
                println!("    {}", style.muted(root_str.as_ref()));
            }
            println!();
            print!("  {} Enter model path manually: ", style.info("→"));
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let path = input.trim().to_string();
            if path.is_empty() {
                return Ok(None);
            }
            return Ok(Some(path));
        }

        let all: Vec<_> = all;

        println!(
            "  {} Found {} model(s)",
            style.success("✓"),
            style.accent(&all.len().to_string())
        );
        println!();

        for (i, model) in all.iter().take(20).enumerate() {
            println!(
                "  {}.  {}  {}",
                i + 1,
                style.accent(&model.name),
                style.muted(&model.size_formatted)
            );
        }

        if all.len() > 20 {
            println!(
                "  {} ... and {} more",
                style.muted("→"),
                style.muted((all.len() - 20).to_string())
            );
        }

        println!();
        print!("  {} Select model (1-{}): ", style.info("→"), all.len());
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        let choice: usize = input.trim().parse().unwrap_or(0);
        if choice == 0 || choice > all.len() {
            println!("{}", style.error("Invalid selection"));
            return Ok(None);
        }

        let selected = &all[choice - 1];
        Ok(Some(selected.path.to_string_lossy().to_string()))
    }

    fn print_docker_command(&self, profile: &crate::models::Profile) -> Result<()> {
        let style = &self.style;
        println!();
        println!("  {}", style.title("Docker Command"));
        println!();
        println!("docker run \\");
        for arg in profile.to_docker_args(self.args.port, self.args.metrics, self.args.public)? {
            if arg.contains(' ') {
                println!("  '{}' \\", arg);
            } else {
                println!("  {} \\", arg);
            }
        }
        Ok(())
    }

    async fn run_container(
        &self,
        docker_client: &DockerClient,
        model_path: &str,
        profile: &crate::models::Profile,
    ) -> Result<()> {
        let container_name = profile.container_name();
        let style = &self.style;

        if let Some(_existing) = docker_client.get_container(&container_name).await? {
            println!("  {} Removing existing container", style.info("→"));
            docker_client.remove_container(&container_name).await?;
        }

        if !docker_client.image_exists(&profile.docker_image).await {
            println!(
                "  {} Pulling Docker image ({})...",
                style.info("→"),
                profile.docker_image
            );
            std::io::stdout().flush()?;
            docker_client.pull_image(&profile.docker_image).await?;
            println!("  {} Image pulled", style.success("✓"));
        }

        println!(
            "  {} Starting {} server...",
            style.info("→"),
            profile.backend.display_name()
        );
        std::io::stdout().flush()?;
        docker_client
            .run_container(
                &container_name,
                &profile.docker_image,
                model_path,
                self.args.port,
                profile,
                self.args.metrics,
                self.args.public,
                self.args.debug,
            )
            .await?;

        println!("  {} Waiting for server to be ready...", style.info("→"));
        std::io::stdout().flush()?;
        docker_client
            .wait_for_health(&container_name, self.args.port)
            .await?;

        Ok(())
    }
}

pub struct StatusCommand {
    args: StatusArgs,
    style: Style,
}

impl StatusCommand {
    pub fn new(args: StatusArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let docker_client = match DockerClient::new() {
            Ok(client) => client,
            Err(err) => {
                println!(
                    "{}",
                    self.style.warning(&format!("Docker unavailable: {err}"))
                );
                return Ok(());
            }
        };

        if let Some(name) = &self.args.name {
            self.show_container(&docker_client, name).await?;
        } else {
            self.show_all_containers(&docker_client).await?;
        }

        Ok(())
    }

    async fn show_container(&self, docker_client: &DockerClient, name: &str) -> Result<()> {
        let style = &self.style;

        println!();
        println!("  {}", style.title("Container Status"));
        println!();

        let container = match docker_client.get_container(name).await {
            Ok(container) => container,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if let Some(container) = container {
            println!("  {} {}", style.info("→"), style.accent("Name"));
            println!("    {}", container.name);

            println!("  {} {}", style.info("→"), style.accent("Image"));
            println!("    {}", container.image);

            println!("  {} {}", style.info("→"), style.accent("Status"));
            let status_lower = container.status.to_lowercase();
            if status_lower.contains("running") {
                println!(
                    "    {} {}",
                    style.success("running"),
                    style.muted(&container.status)
                );
            } else {
                println!(
                    "    {} {}",
                    style.warning("!"),
                    style.muted(&container.status)
                );
            }

            if !container.ports.is_empty() {
                println!("  {} {}", style.info("→"), style.accent("Ports"));
                for (port, host_ip) in &container.ports {
                    println!(
                        "    {}:{}",
                        style.accent(&port.to_string()),
                        style.muted(host_ip)
                    );
                }
            }

            println!("  {} {}", style.info("→"), style.accent("Started"));
            println!("    {}", container.created_at);

            println!();
            println!(
                "  {} Use `llmr stop` to stop this container",
                style.success("✓")
            );
        } else {
            println!(
                "  {} Container '{}' not found (not running)",
                style.warning("!"),
                name
            );
            println!();
            println!(
                "  {} Use `llmr serve` to start a container",
                style.info("→")
            );
        }

        Ok(())
    }

    async fn show_all_containers(&self, docker_client: &DockerClient) -> Result<()> {
        let style = &self.style;

        println!();
        println!("  {}", style.title("Containers"));
        println!();

        let containers = match docker_client.list_containers_by_prefix("llmr_").await {
            Ok(containers) => containers,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if containers.is_empty() {
            println!(
                "  {} No llmr containers are running at the moment",
                style.dash()
            );
            println!();
            println!("  {}", style.muted("Run `llmr serve` to start a container"));
            return Ok(());
        }

        println!("  {} {}", style.info("→"), style.accent("Running"));
        for container in containers {
            let status_lower = container.status.to_lowercase();
            let status_icon = if status_lower.contains("running") {
                style.success("●")
            } else {
                style.warning("●")
            };
            println!("    {} {}", status_icon, container.name);
            println!(
                "      {} · {}",
                container.image,
                style.muted(&container.status)
            );
        }

        println!();
        println!(
            "  {} Run `llmr status -n <name>` for details",
            style.info("→")
        );

        Ok(())
    }
}

pub struct StopCommand {
    args: StopArgs,
    style: Style,
}

impl StopCommand {
    pub fn new(args: StopArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let docker_client = match DockerClient::new() {
            Ok(client) => client,
            Err(err) => {
                println!(
                    "{}",
                    self.style.warning(&format!("Docker unavailable: {err}"))
                );
                return Ok(());
            }
        };

        if let Some(name) = &self.args.name {
            self.stop_container(&docker_client, name).await?;
        } else {
            self.stop_all_containers(&docker_client).await?;
        }

        Ok(())
    }

    async fn stop_container(&self, docker_client: &DockerClient, name: &str) -> Result<()> {
        let style = &self.style;
        let container = match docker_client.get_container(name).await {
            Ok(container) => container,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if container.is_some() {
            docker_client.remove_container(name).await?;
            println!(
                "{}",
                style.success(&format!("Container '{}' stopped and removed", name))
            );
        } else {
            warn!("Container '{}' not found", name);
        }

        Ok(())
    }

    async fn stop_all_containers(&self, docker_client: &DockerClient) -> Result<()> {
        let style = &self.style;
        let containers = match docker_client.list_containers_by_prefix("llmr_").await {
            Ok(containers) => containers,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if containers.is_empty() {
            println!("{}", style.muted("No running llmr_* containers found."));
            return Ok(());
        }

        for container in containers {
            docker_client.remove_container(&container.name).await?;
            println!(
                "{}",
                style.success(&format!(
                    "Container '{}' stopped and removed",
                    container.name
                ))
            );
        }

        Ok(())
    }
}

pub struct ProfilesCommand {
    args: ProfilesArgs,
    style: Style,
}

impl ProfilesCommand {
    pub fn new(args: ProfilesArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let profile_manager = ProfileManager::new();

        match &self.args.subcommand {
            Some(ProfilesSubcommand::List) => {
                self.list_profiles(&profile_manager).await?;
            }
            Some(ProfilesSubcommand::Delete { key }) => {
                self.delete_profile(&profile_manager, key).await?;
            }
            Some(ProfilesSubcommand::Clear) => {
                self.clear_profiles(&profile_manager).await?;
            }
            Some(ProfilesSubcommand::Show { key }) => {
                self.show_profile(&profile_manager, key).await?;
            }
            None => {
                self.list_profiles(&profile_manager).await?;
            }
        }

        Ok(())
    }

    async fn list_profiles(&self, profile_manager: &ProfileManager) -> Result<()> {
        let style = &self.style;
        let profiles = profile_manager.list_all().await?;

        if profiles.is_empty() {
            println!("{}", style.muted("No saved profiles."));
            return Ok(());
        }

        println!(
            "Saved profiles ({}):",
            style.accent(&profiles.len().to_string())
        );
        for (key, profile) in profiles {
            println!("\n  Profile: {}", style.accent(&key));
            println!("    Docker Image: {}", profile.docker_image);
            println!(
                "    Threads: {} | Batch: {} | GPU Layers: {}",
                profile.threads, profile.batch_size, profile.gpu_layers
            );
            println!(
                "    Split Mode: {} | Context: {} | Cache K/V: {}/{}",
                profile.split_mode,
                profile.context_size,
                profile.cache_type_k,
                profile.cache_type_v
            );
            println!(
                "    GPU Type: {} | GPU Count: {}",
                profile.gpu_type, profile.gpu_count
            );
            if let Some(tps) = profile.best_tps {
                println!("    Best TPS: {:.2}", tps);
            }
        }

        Ok(())
    }

    async fn delete_profile(&self, profile_manager: &ProfileManager, key: &str) -> Result<()> {
        let style = &self.style;
        profile_manager.delete(key).await?;
        println!("{}", style.success(&format!("Profile '{}' deleted", key)));
        Ok(())
    }

    async fn clear_profiles(&self, profile_manager: &ProfileManager) -> Result<()> {
        let style = &self.style;
        println!(
            "{}",
            style.warning("This will delete ALL saved profiles in the config directory.")
        );
        print!("Are you sure? [y/N] ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let confirm = input.trim().to_lowercase();

        if confirm == "y" || confirm == "yes" {
            profile_manager.clear_all().await?;
            println!("{}", style.success("All profiles cleared."));
        } else {
            println!("{}", style.muted("Aborted."));
        }

        Ok(())
    }

    async fn show_profile(&self, profile_manager: &ProfileManager, key: &str) -> Result<()> {
        let style = &self.style;
        if let Some(profile) = profile_manager.load(key).await? {
            println!("Profile: {}", style.accent(key));
            println!("Model: {}", profile.model_file);
            println!("Docker Image: {}", profile.docker_image);
            println!("Threads: {}", profile.threads);
            println!("GPU Layers: {}", profile.gpu_layers);
            println!("Context Size: {}", profile.context_size);
            println!("Batch Size: {}", profile.batch_size);
            println!("Ubatch Size: {}", profile.ubatch_size);
            println!("Split Mode: {}", profile.split_mode);
            println!("GPU Type: {}", profile.gpu_type);
            println!("Cache Type K: {}", profile.cache_type_k);
            println!("Cache Type V: {}", profile.cache_type_v);
            println!("Parallel Slots: {}", profile.parallel_slots);
            println!("Created: {}", profile.created_at);
            if let Some(tps) = profile.best_tps {
                println!("Best TPS: {:.2}", tps);
            }
        } else {
            println!("{}", style.warning(&format!("Profile '{}' not found", key)));
        }

        Ok(())
    }
}

pub struct DoctorCommand {
    style: Style,
}

impl DoctorCommand {
    pub fn new(style: Style) -> Self {
        Self { style }
    }

    pub async fn execute(&self) -> Result<()> {
        let style = &self.style;

        println!();
        println!("  {}", style.title("llmr Doctor"));
        println!();

        let env_check = EnvCheck::new();

        let (docker_result, hardware_result) = tokio::join!(
            async {
                let check = DockerCheck::new();
                check.check().await
            },
            async {
                let check = HardwareCheck::new();
                check.check().await
            }
        );

        env_check.run(style);

        print_diagnostic_results(style, &docker_result, &hardware_result);

        println!();
        println!("  {} Diagnostics complete", style.success("✓"));

        Ok(())
    }
}

pub struct VersionCommand;

impl VersionCommand {
    pub async fn execute() -> Result<()> {
        println!("llmr {}", env!("CARGO_PKG_VERSION"));
        println!("Currently supports llama.cpp; vLLM and SGLang adapters are planned.");
        Ok(())
    }
}

pub struct UpdateCommand {
    args: UpdateArgs,
    #[allow(dead_code)]
    style: Style,
}

impl UpdateCommand {
    pub fn new(args: UpdateArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let current = env!("CARGO_PKG_VERSION");

        if self.args.check {
            println!("{}", self.style.title("Checking for updates..."));
            let latest = self.fetch_latest_version().await?;

            if current == latest {
                println!(
                    "  {} Already on latest version {}",
                    self.style.success("✓"),
                    current
                );
                return Ok(());
            }

            println!(
                "  {} Version {} available (you have {})",
                self.style.info("→"),
                latest,
                current
            );
            println!();
            println!("  Run {} to update", self.style.accent("llmr update"));
            return Ok(());
        }

        println!("{}", self.style.title("Updating llmr..."));
        println!("  Current: {}", current);

        let latest = self.fetch_latest_version().await?;
        println!("  Latest:  {}", latest);

        if current == latest {
            println!();
            println!("  {} Already on latest version", self.style.success("✓"));
            return Ok(());
        }

        println!();
        self.perform_update().await
    }

    async fn fetch_latest_version(&self) -> Result<String> {
        let url = "https://api.github.com/repos/aditya-xq/llmr/releases/latest";
        let client = reqwest::Client::new();
        let resp = client.get(url).send().await.map_err(|e| Error::Other {
            message: format!("Failed to check for updates: {}", e),
        })?;

        let json: serde_json::Value = resp.json().await.map_err(|e| Error::Other {
            message: format!("Failed to parse response: {}", e),
        })?;

        let tag = json
            .get("tag_name")
            .and_then(|v| v.as_str())
            .unwrap_or("0.0.0")
            .trim_start_matches('v');

        Ok(tag.to_string())
    }

    async fn perform_update(&self) -> Result<()> {
        let update_script = if cfg!(windows) {
            "irm https://raw.githubusercontent.com/aditya-xq/llmr/main/install.ps1 | iex"
        } else {
            "curl -sSL https://raw.githubusercontent.com/aditya-xq/llmr/main/install.sh | sh"
        };

        println!("  {}", self.style.info("→") + " Running installer...");
        println!();

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(update_script)
            .output()
            .await
            .map_err(|e| Error::Other {
                message: format!("Failed to run update: {}", e),
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        print!("{}", stdout);
        eprint!("{}", stderr);

        if output.status.success() {
            println!();
            println!("  {} Update complete!", self.style.success("✓"))
        } else {
            println!();
            println!(
                "  {} Update may have failed. Try manually:",
                self.style.warning("!")
            );
            println!("  {}", self.style.accent(update_script));
        }

        Ok(())
    }
}

pub struct TuneCommand {
    args: TuneArgs,
    style: Style,
}

impl TuneCommand {
    pub fn new(args: TuneArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let style = &self.style;

        let _ = ensure_docker(style).await?;

        let model_path = if let Some(model) = &self.args.model {
            if !Path::new(model).exists() {
                return Err(Error::ModelNotFound {
                    path: model.clone(),
                });
            }
            model.clone()
        } else {
            match self.interactive_model_select().await? {
                Some(p) => p,
                None => {
                    println!("{}", style.error("No model selected"));
                    return Ok(());
                }
            }
        };

        println!();
        println!("  {}", style.title("Tuning Profile"));
        println!("  Model: {}", style.accent(&model_path));
        println!();

        println!("  {} Detecting hardware...", style.info("→"));
        let hardware = crate::hardware::detect().await?;
        println!(
            "    CPU: {} cores ({} threads)",
            hardware.cpu.cores, hardware.cpu.threads
        );
        if let Some(ref gpu) = hardware.gpu {
            println!("    GPU: {} x {}", gpu.names.join(", "), gpu.type_);
            let total_vram: u64 = gpu.vram_mb.iter().sum();
            println!("    VRAM: {} MB", total_vram);
        }
        println!("    RAM: {} GB", hardware.ram.total_gb);
        println!();

        println!("  {} Extracting GGUF metadata...", style.info("→"));
        let extractor = GgufExtractor;
        let facts = extractor
            .extract(Path::new(&model_path))
            .map_err(|e| Error::Other {
                message: format!("Failed to parse GGUF: {e}"),
            })?;
        println!("    Architecture: {}", facts.architecture);
        if let Some(ctx) = facts.context_length {
            println!("    Context Length: {}", ctx);
        }
        if let Some(layers) = facts.block_count {
            println!("    Layers: {}", layers);
        }
        println!();

        println!("  {} Running optimization benchmarks...", style.info("→"));
        std::io::stdout().flush()?;

        let hardware_profile: crate::tuning::HardwareProfile = hardware.clone().into();

        let opts = OptimizeOptions {
            benchmark_ctx_size: Some(facts.context_length.unwrap_or(4096).min(4096)),
            prompt_tokens: self.args.prompt_tokens.unwrap_or(512),
            generation_tokens: self.args.generation_tokens.unwrap_or(128),
            parallel_requests: 1,
            max_rounds: if self.args.quick {
                1
            } else {
                self.args.max_rounds.unwrap_or(4)
            },
            min_relative_gain: 0.01,
            warmup_samples: 2,
            benchmark_samples: 5,
            early_stop_samples: 2,
            racing_keep_fraction: 0.25,
            coarse_budget_ms: 500,
            fine_budget_ms: 3000,
            search_strategy: crate::tuning::SearchStrategy::Racing,
        };

        let model_path_for_bench = model_path.clone();
        let backend = Backend::ACTIVE;
        let docker_image = Profile::select_docker_image(&hardware.gpu, backend);

        let result = crate::tuning::optimize_llama_cpp_profile(
            &model_path,
            hardware_profile,
            &extractor,
            move |llama_profile| {
                let model_path = model_path_for_bench.clone();
                let docker_image = docker_image.clone();
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        run_benchmark(
                            &model_path,
                            &docker_image,
                            llama_profile,
                            opts.warmup_samples,
                            opts.benchmark_samples,
                            opts.prompt_tokens,
                            opts.generation_tokens,
                        )
                        .await
                    })
                })
            },
            opts,
        )
        .map_err(|e| Error::Other {
            message: format!("Optimization failed: {e}"),
        })?;

        let profile = &result.profile;
        let stats = result.stats;

        println!();
        println!("  {} Best configuration found:", style.success("✓"));
        println!(
            "    Tested: {} candidates | Cache: {} entries",
            stats.candidates_tested, stats.cache_size
        );
        println!(
            "    Threads: {} | Batch: {} | UBatch: {}",
            profile.threads, profile.batch_size, profile.ubatch_size
        );
        println!("    GPU Layers: {:?}", profile.n_gpu_layers);
        println!("    Split Mode: {:?}", profile.split_mode);
        println!("    Cache Type K: {:?}", profile.cache_type_k);
        println!("    Cache Type V: {:?}", profile.cache_type_v);
        if let Some(ref res) = profile.estimated_result {
            println!(
                "    Estimated: prompt={:.1}, decode={:.1} tok/s, latency={:.0}ms",
                res.prompt_tps, res.decode_tps, res.latency_ms
            );
        }
        println!();

        if self.args.dry_run {
            println!("  {} Dry run - not saving profile", style.info("→"));
            println!();
            println!("  Command-line args:");
            for arg in profile.to_cli_args() {
                println!("    {}", arg);
            }
            return Ok(());
        }

        let profile = convert_to_profile(profile, &model_path, &hardware);

        let profile_manager = ProfileManager::new();
        profile_manager.save(&profile).await?;

        println!(
            "{}  Profile saved for model '{}'",
            style.success("✓"),
            style.accent(&profile.model_file)
        );
        println!("    Key: {}", style.muted(profile.key()));

        Ok(())
    }

    async fn interactive_model_select(&self) -> Result<Option<String>> {
        let style = &self.style;
        let scanner = ModelScanner::new();

        println!();
        println!("  {}", style.title("Searching for GGUF models..."));
        println!();

        let profile_manager = ProfileManager::new();
        let cached_folders = profile_manager.get_cached_model_folders();
        let quick_scan_results = if !cached_folders.is_empty() {
            println!("  {} Checking cached locations...", style.info("→"));
            scanner.scan_paths(&cached_folders)
        } else {
            Vec::new()
        };

        let all = if !quick_scan_results.is_empty() {
            quick_scan_results
        } else {
            println!("  {} Scanning disks for GGUF files...", style.info("→"));
            scanner.scan_disks()
        };

        if all.is_empty() {
            println!(
                "  {} No GGUF models found on local disks",
                style.warning("!")
            );
            println!();
            print!("  {} Enter model path manually: ", style.info("→"));
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let path = input.trim().to_string();
            if path.is_empty() {
                return Ok(None);
            }
            return Ok(Some(path));
        }

        println!(
            "  {} Found {} model(s)",
            style.success("✓"),
            style.accent(&all.len().to_string())
        );
        println!();

        for (i, model) in all.iter().take(20).enumerate() {
            println!(
                "  {}.  {}  {}",
                i + 1,
                style.accent(&model.name),
                style.muted(&model.size_formatted)
            );
        }

        println!();
        print!("  {} Select model (1-{}): ", style.info("→"), all.len());
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        let choice: usize = input.trim().parse().unwrap_or(0);
        if choice == 0 || choice > all.len() {
            println!("{}", style.error("Invalid selection"));
            return Ok(None);
        }

        let selected = &all[choice - 1];
        Ok(Some(selected.path.to_string_lossy().to_string()))
    }
}

async fn run_benchmark(
    model_path: &str,
    docker_image: &str,
    llama_profile: &LlamaCppProfile,
    warmup_samples: usize,
    benchmark_samples: usize,
    prompt_tokens: u32,
    generation_tokens: u32,
) -> std::result::Result<crate::tuning::BenchmarkResult, OptimizeError> {
    use tokio::process::Command as TokioCommand;

    let port = find_free_port(18090);
    let container_name = format!("llmr_tune_bench_{}", std::process::id());

    let model_dir = Path::new(model_path)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".");
    let model_file = Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf");

    let args = {
        let mut args = llama_profile.to_cli_args();
        if let Some(idx) = args.iter().position(|a| a == "-m") {
            if idx + 1 < args.len() {
                args[idx + 1] = format!("/models/{}", model_file);
            }
        }
        args.push("--no-mmap".to_string());
        args
    };

    let mut docker_args = vec![
        "run".to_string(),
        "-d".to_string(),
        "--name".to_string(),
        container_name.clone(),
    ];
    if Profile::is_gpu_image(docker_image) {
        docker_args.extend(["--gpus".to_string(), "all".to_string()]);
    }
    docker_args.extend([
        "-v".to_string(),
        format!("{}:/models:ro", model_dir),
        "-p".to_string(),
        format!("{}:8080", port),
        docker_image.to_string(),
    ]);

    let output = TokioCommand::new("docker")
        .args(&docker_args)
        .args(&args)
        .output()
        .await
        .map_err(OptimizeError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("  Docker run failed: {}", stderr);
        return Err(OptimizeError::Benchmark(stderr.to_string()));
    }

    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    eprint!(
        "    → Testing (t={},b={},ub={}) ... ",
        llama_profile.threads, llama_profile.batch_size, llama_profile.ubatch_size
    );
    std::io::stdout().flush()?;
    let healthy = loop {
        if start.elapsed().as_secs() > 180 {
            eprintln!("    → Container timeout - fetching logs...");
            let logs = TokioCommand::new("docker")
                .args(["logs", &container_name])
                .output()
                .await;
            if let Ok(logs) = logs {
                let log_err = String::from_utf8_lossy(&logs.stderr);
                if !log_err.is_empty() {
                    let lines: Vec<&str> = log_err.lines().collect();
                    let last_30 = lines
                        .iter()
                        .rev()
                        .take(30)
                        .rev()
                        .copied()
                        .collect::<Vec<_>>()
                        .join("\n");
                    eprintln!("    Container stderr (last 30 lines):\n{}", last_30);
                }
            }
            let _ = TokioCommand::new("docker")
                .args(["rm", "-f", &container_name])
                .output()
                .await;
            return Err(OptimizeError::Benchmark(
                "Timeout waiting for container".to_string(),
            ));
        }

        if let Ok(resp) = client
            .get(format!("http://127.0.0.1:{}/health", port))
            .timeout(Duration::from_secs(3))
            .send()
            .await
        {
            if resp.text().await.unwrap_or_default().contains("ok") {
                break true;
            }
        }
        if start.elapsed().as_secs() > 0 && start.elapsed().as_secs().is_multiple_of(15) {
            eprint!(".");
            let _ = std::io::stdout().flush();
        }
        sleep(Duration::from_secs(1)).await;
    };

    if !healthy {
        let _ = TokioCommand::new("docker")
            .args(["rm", "-f", &container_name])
            .output()
            .await;
        return Err(OptimizeError::Benchmark(
            "Container not healthy".to_string(),
        ));
    }

    let prompt = "Write a detailed explanation of quantum computing, covering superposition, entanglement, and quantum gates. Be thorough and include examples.".chars().take(prompt_tokens as usize).collect::<String>();
    if prompt.len() < prompt_tokens as usize {
        let repeat = (prompt_tokens as usize / prompt.len()) + 1;
        let prompt = prompt.repeat(repeat);
        let _prompt = prompt
            .chars()
            .take(prompt_tokens as usize)
            .collect::<String>();
    }

    let prompt_for_request = prompt.clone();
    let gen_tokens = generation_tokens;

    let run_single_request = async |client: &reqwest::Client,
                                    port: u16,
                                    prompt: &str,
                                    n_predict: u32|
           -> std::result::Result<(f32, f32, f32, u32), String> {
        let request_body = serde_json::json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0,
            "stream": false
        });

        let start = std::time::Instant::now();
        let response = client
            .post(format!("http://127.0.0.1:{}/completion", port))
            .json(&request_body)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return Err("Invalid elapsed time".to_string());
        }

        let json: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;

        let _tokens_predicted = json
            .get("tokens_predicted")
            .and_then(|v| v.as_u64())
            .unwrap_or(n_predict as u64) as f64;

        let tokens_evaluated = json
            .get("tokens_evaluated")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as f64;

        let prompt_ms = json
            .get("timings")
            .and_then(|t| t.get("prompt_ms"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let tokens_predicted = json
            .get("tokens_predicted")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as f64;

        let predicted_ms = json
            .get("timings")
            .and_then(|t| t.get("predicted_ms"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let prompt_tps = if prompt_ms > 0.0 {
            (tokens_evaluated / (prompt_ms / 1_000.0)) as f32
        } else {
            0.0
        };

        let decode_tps = if predicted_ms > 0.0 {
            (tokens_predicted / (predicted_ms / 1_000.0)) as f32
        } else {
            0.0
        };

        let latency_ms = (elapsed * 1000.0) as f32;

        Ok((prompt_tps, decode_tps, latency_ms, 0))
    };

    if warmup_samples > 0 {
        eprint!(" [warmup");
        for _ in 0..warmup_samples {
            let _ = run_single_request(&client, port, &prompt_for_request, 16).await;
            eprint!(".");
        }
        eprint!("]");
    }

    let mut runs = Vec::with_capacity(benchmark_samples);

    for i in 0..benchmark_samples {
        match run_single_request(&client, port, &prompt_for_request, gen_tokens).await {
            Ok((prompt_tps, decode_tps, latency_ms, memory_mib)) => {
                runs.push(crate::tuning::BenchmarkRun {
                    prompt_tps,
                    decode_tps,
                    latency_ms,
                    memory_mib,
                    failed: false,
                });
                eprint!(".");
            }
            Err(e) => {
                runs.push(crate::tuning::BenchmarkRun {
                    prompt_tps: 0.0,
                    decode_tps: 0.0,
                    latency_ms: 0.0,
                    memory_mib: 0,
                    failed: true,
                });
                eprintln!("\n    → Run {} failed: {}", i + 1, e);
            }
        }
        let _ = std::io::stdout().flush();
    }

    let _ = TokioCommand::new("docker")
        .args(["rm", "-f", &container_name])
        .output()
        .await;

    let result = crate::tuning::BenchmarkResult::from_runs(runs)
        .ok_or_else(|| OptimizeError::Benchmark("All benchmark runs failed".to_string()))?;

    println!(
        " ✓ prompt={:.1}, decode={:.1}, lat={:.0}ms, stable={:.1}%",
        result.prompt_tps,
        result.decode_tps,
        result.latency_ms,
        result.stability * 100.0
    );

    Ok(result)
}

fn find_free_port(start: u16) -> u16 {
    for port in start..start + 100 {
        if std::net::TcpListener::bind(format!("127.0.0.1:{}", port)).is_ok() {
            return port;
        }
    }
    start
}

pub struct BenchCommand {
    args: BenchArgs,
    style: Style,
}

impl BenchCommand {
    pub fn new(args: BenchArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let style = &self.style;
        let test_type = self
            .args
            .test_type
            .clone()
            .unwrap_or_else(|| "perf".to_string());

        let config = if let Some(config_path) = &self.args.config {
            let path = std::path::Path::new(config_path);
            if !path.exists() {
                return Err(Error::Other {
                    message: format!("Config file not found: {}", config_path.display()),
                });
            }
            match config::load_config(path) {
                Ok(cfg) => Some(cfg),
                Err(e) => {
                    return Err(Error::Other {
                        message: format!("Config error: {}", e),
                    })
                }
            }
        } else {
            let default_config_path = std::path::Path::new("config.yaml");
            if default_config_path.exists() {
                match config::load_config(default_config_path) {
                    Ok(cfg) => Some(cfg),
                    Err(e) => {
                        eprintln!("Warning: Failed to load config.yaml: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        };

        let base_url = if let Some(ref cfg) = config {
            let url = cfg.server.endpoint.as_str();
            let url = url
                .trim_end_matches("/v1/chat/completions")
                .trim_end_matches("/v1")
                .trim_end_matches("/chat/completions");
            url
        } else {
            &self.args.base_url
        };

        println!();
        println!("  {}", style.title("Benchmark"));
        println!("  Test Type: {}", style.accent(&test_type));
        println!("  Server: {}", style.accent(base_url));
        if let Some(ref cfg) = config {
            if !cfg.performance.prompts.is_empty() {
                println!("  Prompts: {}", cfg.performance.prompts.len());
            }
            if cfg.quality.enabled && !cfg.quality.tasks.is_empty() {
                println!("  Quality Tasks: {}", cfg.quality.tasks.join(", "));
            }
        }
        println!();

        match test_type.as_str() {
            "quality" | "eval" | "lm-eval" => {
                self.run_quality_eval(style, base_url, config.as_ref())
                    .await?;
            }
            "perf" | "performance" | "speed" => {
                self.run_performance_eval(style, base_url, config.as_ref())
                    .await?;
            }
            _ => {
                return Err(Error::Other {
                    message: format!(
                        "Unknown test type: {}. Use quality, perf, or eval",
                        test_type
                    ),
                });
            }
        }

        Ok(())
    }

    async fn run_quality_eval(
        &self,
        style: &Style,
        base_url: &str,
        config: Option<&BenchmarkConfig>,
    ) -> Result<()> {
        use std::io::Write;
        use tokio::process::Command as TokioCommand;

        let tasks_str = if let Some(cfg) = config {
            let tasks = if !cfg.quality.tasks.is_empty() {
                cfg.quality.tasks.join(",")
            } else {
                self.args
                    .tasks
                    .clone()
                    .unwrap_or_else(|| "gsm8k".to_string())
            };
            let fewshot = cfg.quality.num_fewshot.unwrap_or(self.args.fewshot);
            (tasks, fewshot)
        } else {
            (
                self.args
                    .tasks
                    .clone()
                    .unwrap_or_else(|| "gsm8k".to_string()),
                self.args.fewshot,
            )
        };
        let tasks = &tasks_str.0;
        let fewshot = tasks_str.1;

        println!("  Tasks: {}", style.accent(tasks));
        println!("  Few-shot: {}", fewshot);
        println!();

        let mut check = TokioCommand::new("python")
            .args(["-c", "import lm_eval"])
            .output()
            .await;

        if check.is_err() || !check.as_ref().map(|o| o.status.success()).unwrap_or(false) {
            println!("  {} Installing lm-eval[api]...", style.info("→"));
            let install = TokioCommand::new("pip")
                .args(["install", "lm-eval[api]"])
                .output()
                .await
                .map_err(|e| Error::Other {
                    message: format!("Failed to install lm-eval: {}", e),
                })?;

            if !install.status.success() {
                let stderr = String::from_utf8_lossy(&install.stderr);
                return Err(Error::Other {
                    message: format!("pip install failed: {}", stderr),
                });
            }

            check = TokioCommand::new("python")
                .args(["-c", "import lm_eval"])
                .output()
                .await;

            if check.is_err() || !check.as_ref().map(|o| o.status.success()).unwrap_or(false) {
                println!("  {} Repairing broken installation...", style.info("→"));
                let repair = TokioCommand::new("pip")
                    .args(["install", "--force-reinstall", "--no-deps", "lm-eval"])
                    .output()
                    .await
                    .map_err(|e| Error::Other {
                        message: format!("Failed to repair lm-eval: {}", e),
                    })?;

                if !repair.status.success() {
                    let stderr = String::from_utf8_lossy(&repair.stderr);
                    return Err(Error::Other {
                        message: format!("Repair failed: {}", stderr),
                    });
                }
            }

            println!("  {} lm-eval installed", style.success("✓"));
        }

        if self.args.dry_run {
            println!("  {} Dry run - would run lm-eval", style.info("→"));
            println!();
            let cmd = format!(
                "python -m lm_eval run --model local-chat-completions --model_args base_url={}/v1 --tasks {} --num_fewshot {}",
                base_url, tasks, fewshot
            );
            println!("  Command: {}", style.muted(&cmd));
            return Ok(());
        }

        println!("  {} Running quality evaluation...", style.info("→"));
        std::io::stdout().flush()?;

        let output = TokioCommand::new("python")
            .args(["-m", "lm_eval", "run", "--model", "local-chat-completions"])
            .args(["--model_args", &format!("base_url={}/v1", base_url)])
            .args(["--tasks", tasks])
            .args(["--num_fewshot", &fewshot.to_string()])
            .output()
            .await
            .map_err(|e| Error::Other {
                message: format!("Failed to run lm-eval: {}", e),
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            eprintln!("{}", stderr);
            return Err(Error::Other {
                message: "lm-eval failed".to_string(),
            });
        }

        println!("{}", stdout);
        if !stderr.is_empty() && !stderr.contains("INFO") {
            eprintln!("{}", stderr);
        }

        println!();
        println!("  {} Quality evaluation complete", style.success("✓"));
        Ok(())
    }

    async fn run_performance_eval(
        &self,
        style: &Style,
        base_url: &str,
        config: Option<&BenchmarkConfig>,
    ) -> Result<()> {
        use std::time::Duration;

        let (
            prompt_tokens,
            generation_tokens,
            warmup,
            samples,
            prompts,
            _max_tokens,
            temperature,
            top_p,
            stream,
        ) = if let Some(cfg) = config {
            let perf = &cfg.performance;
            let prompt_tokens = self.args.prompt_tokens.unwrap_or(512);
            let generation_tokens = self.args.generation_tokens.unwrap_or(perf.max_tokens);
            let warmup = if self.args.quick {
                1
            } else {
                perf.warmup_runs.max(1)
            };
            let samples = if self.args.quick {
                1
            } else {
                perf.measured_runs
            };
            let prompts = if !perf.prompts.is_empty() {
                perf.prompts.clone()
            } else {
                vec!["Write a detailed explanation of quantum computing, covering superposition, entanglement, and quantum gates. Be thorough and include examples.".to_string()]
            };
            (
                prompt_tokens,
                generation_tokens,
                warmup,
                samples,
                prompts,
                perf.max_tokens,
                perf.temperature,
                perf.top_p,
                perf.stream,
            )
        } else {
            let prompt_tokens = self.args.prompt_tokens.unwrap_or(512);
            let generation_tokens = self.args.generation_tokens.unwrap_or(128);
            let warmup = if self.args.quick { 1 } else { 2 };
            let samples = if self.args.quick { 1 } else { 3 };
            (prompt_tokens, generation_tokens, warmup, samples, vec!["Write a detailed explanation of quantum computing, covering superposition, entanglement, and quantum gates. Be thorough and include examples.".to_string()], 256, 1.0, 1.0, false)
        };

        println!("  Prompts: {}", prompts.len());
        println!("  Prompt tokens: {}", prompt_tokens);
        println!("  Generation tokens: {}", generation_tokens);
        println!("  Samples: {} (warmup: {})", samples, warmup);
        println!();

        if self.args.dry_run {
            println!("  {} Dry run - would run performance test", style.info("→"));
            return Ok(());
        }

        println!("  {} Running performance test...", style.info("→"));
        std::io::stdout().flush()?;

        let client = reqwest::Client::new();

        let prompt_base = prompts.first().unwrap();
        let prompt_base_len = prompt_base.chars().count() as u32;

        let prompt: String = if prompt_base_len < prompt_tokens {
            prompt_base
                .repeat((prompt_tokens as usize / prompt_base_len as usize) + 1)
                .chars()
                .take(prompt_tokens as usize)
                .collect()
        } else {
            prompt_base.chars().take(prompt_tokens as usize).collect()
        };

        let run_request = async |client: &reqwest::Client,
                                 url: &str,
                                 prompt: &str,
                                 n_predict: u32|
               -> std::result::Result<(f64, f64, f64), String> {
            let req = serde_json::json!({
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            });

            let start = std::time::Instant::now();
            let resp = client
                .post(url)
                .json(&req)
                .timeout(Duration::from_secs(120))
                .send()
                .await
                .map_err(|e| format!("Request failed: {}", e))?;

            let elapsed = start.elapsed().as_secs_f64();
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| format!("JSON parse failed: {}", e))?;

            let tokens_predicted = json
                .get("tokens_predicted")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as f64;
            let tokens_evaluated = json
                .get("tokens_evaluated")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as f64;
            let predicted_ms = json
                .get("timings")
                .and_then(|t| t.get("predicted_ms"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let prompt_ms = json
                .get("timings")
                .and_then(|t| t.get("prompt_ms"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            let prompt_tps = if prompt_ms > 0.0 {
                tokens_evaluated / (prompt_ms / 1000.0)
            } else {
                0.0
            };
            let decode_tps = if predicted_ms > 0.0 {
                tokens_predicted / (predicted_ms / 1000.0)
            } else {
                0.0
            };
            let latency_ms = elapsed * 1000.0;

            Ok((prompt_tps, decode_tps, latency_ms))
        };

        let url = format!("{}/completion", base_url);

        let mut results = Vec::new();

        if warmup > 0 {
            eprint!("    Warmup: ");
            for _ in 0..warmup {
                let _ = run_request(&client, &url, &prompt, 16).await;
                eprint!(".");
            }
            eprintln!(" done");
        }

        eprint!("    Testing: ");
        std::io::stdout().flush()?;

        for _ in 0..samples {
            match run_request(&client, &url, &prompt, generation_tokens).await {
                Ok((prompt_tps, decode_tps, latency_ms)) => {
                    results.push((prompt_tps, decode_tps, latency_ms));
                    eprint!(".");
                }
                Err(e) => {
                    eprintln!("\n    Error: {}", e);
                }
            }
            std::io::stdout().flush()?;
        }
        eprintln!();

        if results.is_empty() {
            return Err(Error::Other {
                message: "All tests failed".to_string(),
            });
        }

        let n = results.len() as f64;
        let avg_prompt: f64 = results.iter().map(|(p, _, _)| p).sum::<f64>() / n;
        let avg_decode: f64 = results.iter().map(|(_, d, _)| d).sum::<f64>() / n;
        let avg_latency: f64 = results.iter().map(|(_, _, l)| l).sum::<f64>() / n;

        println!();
        println!("  {} Results:", style.success("✓"));
        println!("    Prompt processing: {:.1} tokens/sec", avg_prompt);
        println!("    Token generation: {:.1} tokens/sec", avg_decode);
        println!("    Latency: {:.0} ms", avg_latency);
        println!(
            "    Total throughput: {:.1} tokens/sec",
            avg_prompt + avg_decode
        );

        Ok(())
    }
}

fn convert_to_profile(
    llama_profile: &LlamaCppProfile,
    model_path: &str,
    hardware: &HardwareInfo,
) -> Profile {
    let model_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let gpu_count = hardware
        .gpu
        .as_ref()
        .map(|g| g.names.len() as u32)
        .unwrap_or(0);
    let gpu_vram_total_mb = hardware
        .gpu
        .as_ref()
        .map(|g| g.vram_mb.iter().sum())
        .unwrap_or(0);

    let gpu_layers = match llama_profile.n_gpu_layers {
        crate::tuning::GpuLayerSpec::Exact(n) => n as i32,
        crate::tuning::GpuLayerSpec::All => 999,
        crate::tuning::GpuLayerSpec::Auto => 999,
    };

    let split_mode = match llama_profile.split_mode {
        crate::tuning::SplitMode::None => "none",
        crate::tuning::SplitMode::Layer => "layer",
        crate::tuning::SplitMode::Row => "row",
    }
    .to_string();

    let cache_type_k = match llama_profile.cache_type_k {
        crate::tuning::KvCacheType::F32 => "f32",
        crate::tuning::KvCacheType::F16 => "f16",
        crate::tuning::KvCacheType::BF16 => "bf16",
        crate::tuning::KvCacheType::Q8_0 => "q8_0",
        crate::tuning::KvCacheType::Q4_0 => "q4_0",
        crate::tuning::KvCacheType::Q4_1 => "q4_1",
        crate::tuning::KvCacheType::IQ4_NL => "iq4_nl",
        crate::tuning::KvCacheType::Q5_0 => "q5_0",
        crate::tuning::KvCacheType::Q5_1 => "q5_1",
    }
    .to_string();

    let cache_type_v = match llama_profile.cache_type_v {
        crate::tuning::KvCacheType::F32 => "f32",
        crate::tuning::KvCacheType::F16 => "f16",
        crate::tuning::KvCacheType::BF16 => "bf16",
        crate::tuning::KvCacheType::Q8_0 => "q8_0",
        crate::tuning::KvCacheType::Q4_0 => "q4_0",
        crate::tuning::KvCacheType::Q4_1 => "q4_1",
        crate::tuning::KvCacheType::IQ4_NL => "iq4_nl",
        crate::tuning::KvCacheType::Q5_0 => "q5_0",
        crate::tuning::KvCacheType::Q5_1 => "q5_1",
    }
    .to_string();

    let backend = Backend::ACTIVE;
    let docker_image = Profile::select_docker_image(&hardware.gpu, backend);

    let gpu_type = hardware
        .gpu
        .as_ref()
        .map(|g| g.type_.clone())
        .unwrap_or_else(|| "none".to_string());

    Profile {
        model_file: model_path.to_string(),
        model_size_bytes: model_size,
        cpu_cores: hardware.cpu.cores,
        gpu_count,
        gpu_vram_total_mb,
        docker_image,
        backend,
        threads: llama_profile.threads as u32,
        batch_size: llama_profile.batch_size,
        ubatch_size: llama_profile.ubatch_size,
        gpu_layers,
        split_mode,
        context_size: llama_profile.ctx_size,
        cache_type_k,
        cache_type_v,
        parallel_slots: llama_profile.parallel as u32,
        gpu_type,
        has_nvlink: hardware.has_nvlink,
        created_at: chrono::Utc::now().to_rfc3339(),
        best_tps: llama_profile
            .estimated_result
            .clone()
            .map(|r| (r.prompt_tps as f64 + r.decode_tps as f64) / 2.0),
    }
}
