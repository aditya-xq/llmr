use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

pub mod candidates;
pub mod gguf;
pub mod search;

pub use search::SearchStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Backend {
    #[default]
    LlamaCpp,
    Vllm,
    Sglang,
}

impl Backend {
    pub const ACTIVE: Self = Self::LlamaCpp;

    pub fn display_name(&self) -> &'static str {
        match self {
            Backend::LlamaCpp => "llama.cpp",
            Backend::Vllm => "vLLM",
            Backend::Sglang => "SGLang",
        }
    }

    pub fn support_status(&self) -> &'static str {
        match self {
            Backend::LlamaCpp => "supported",
            Backend::Vllm | Backend::Sglang => "planned",
        }
    }

    pub fn supports_serving(&self) -> bool {
        matches!(self, Backend::LlamaCpp)
    }

    pub fn unsupported_message(&self) -> String {
        format!(
            "{} support is planned but not wired for serve/tune yet. Current releases only run llama.cpp.",
            self.display_name()
        )
    }

    pub fn docker_registry(&self) -> &'static str {
        match self {
            Backend::LlamaCpp => "ghcr.io/ggml-org",
            Backend::Vllm => "ghcr.io/vllm-project",
            Backend::Sglang => "ghcr.io/lm-sys",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl From<crate::hardware::HardwareInfo> for HardwareProfile {
    fn from(info: crate::hardware::HardwareInfo) -> Self {
        let gpus = info
            .gpu
            .as_ref()
            .map(|g| {
                g.names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| GpuProfile {
                        index: i as u32,
                        name: name.clone(),
                        vram_bytes: g.vram_mb.get(i).copied().unwrap_or(0) * 1024 * 1024,
                        supports_flash_attn: g.type_ == "nvidia" || g.type_ == "amd",
                        supports_bf16: g.type_ == "nvidia",
                    })
                    .collect()
            })
            .unwrap_or_default();

        HardwareProfile {
            system_ram_bytes: info.ram.total,
            cpu_physical_cores: info.cpu.cores as u16,
            cpu_logical_cores: info.cpu.threads as u16,
            gpus,
            numa_nodes: None,
        }
    }
}

impl From<crate::hardware::GpuInfo> for GpuProfile {
    fn from(gpu: crate::hardware::GpuInfo) -> Self {
        GpuProfile {
            index: 0,
            name: gpu
                .names
                .first()
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string()),
            vram_bytes: gpu.vram_mb.first().copied().unwrap_or(0) * 1024 * 1024,
            supports_flash_attn: gpu.type_ == "nvidia" || gpu.type_ == "amd",
            supports_bf16: gpu.type_ == "nvidia",
        }
    }
}

pub struct GgufExtractor;

impl GgufFactsExtractor for GgufExtractor {
    fn extract(&self, path: &Path) -> Result<GgufFacts, OptimizeError> {
        let file = std::fs::File::open(path).map_err(OptimizeError::Io)?;
        let metadata = file.metadata().map_err(OptimizeError::Io)?;
        let mut reader = std::io::BufReader::new(file);
        gguf::extract_from_reader(&mut reader, path, metadata.len())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub system_ram_bytes: u64,
    pub cpu_physical_cores: u16,
    pub cpu_logical_cores: u16,
    pub gpus: Vec<GpuProfile>,
    pub numa_nodes: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub index: u32,
    pub name: String,
    pub vram_bytes: u64,
    pub supports_flash_attn: bool,
    pub supports_bf16: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GgufFacts {
    pub path: PathBuf,
    pub architecture: String,
    pub model_name: Option<String>,
    pub size_label: Option<String>,
    pub quantization_version: Option<u32>,
    pub file_type: Option<u32>,
    pub alignment: Option<u32>,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub block_count: Option<u32>,
    pub feed_forward_length: Option<u32>,
    pub attention_head_count: Option<u32>,
    pub attention_head_count_kv: Option<u32>,
    pub rope_dimension_count: Option<u32>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context_length: Option<u32>,
    pub chat_template: Option<String>,
    pub tensor_count: usize,
    pub weight_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GpuLayerSpec {
    Auto,
    All,
    Exact(u32),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SplitMode {
    None,
    Layer,
    Row,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FlashAttn {
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum KvCacheType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_1,
    IQ4_NL,
    Q5_0,
    Q5_1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NumaMode {
    Distribute,
    Isolate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SpecType {
    Eager,
    NGram,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppProfile {
    pub model_path: PathBuf,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: u16,
    pub threads_batch: u16,
    pub parallel: u16,
    pub n_gpu_layers: GpuLayerSpec,
    pub split_mode: SplitMode,
    pub tensor_split: Option<Vec<f32>>,
    pub main_gpu: Option<u32>,
    pub flash_attn: FlashAttn,
    pub cache_type_k: KvCacheType,
    pub cache_type_v: KvCacheType,
    pub kv_offload: bool,
    pub repack: bool,
    pub mmap: bool,
    pub mlock: bool,
    pub numa: Option<NumaMode>,
    pub cont_batching: bool,
    pub no_host: bool,
    pub poll: Option<u32>,
    pub poll_batch: Option<u32>,
    pub cpu_mask: Option<String>,
    pub cpu_range: Option<String>,
    pub cpu_strict: bool,
    pub draft_model_path: Option<PathBuf>,
    pub draft_gpu_layers: GpuLayerSpec,
    pub draft_threads: u16,
    pub draft_cache_type: Option<KvCacheType>,
    pub spec_type: Option<SpecType>,
    pub spec_extra: Option<String>,
    pub fit: bool,
    pub fit_target_mib: Vec<u32>,
    #[serde(skip)]
    pub estimated_result: Option<BenchmarkResult>,
    #[serde(skip)]
    pub facts: GgufFacts,
    #[serde(skip)]
    pub notes: Vec<String>,
}

impl LlamaCppProfile {
    pub fn to_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "-m".to_string(),
            self.model_path.display().to_string(),
            "-c".to_string(),
            self.ctx_size.to_string(),
            "-b".to_string(),
            self.batch_size.to_string(),
            "-ub".to_string(),
            self.ubatch_size.to_string(),
            "-t".to_string(),
            self.threads.to_string(),
            "-tb".to_string(),
            self.threads_batch.to_string(),
            "-np".to_string(),
            self.parallel.to_string(),
            "-fa".to_string(),
            match self.flash_attn {
                FlashAttn::Auto => "auto",
                FlashAttn::On => "on",
                FlashAttn::Off => "off",
            }
            .to_string(),
            "-ctk".to_string(),
            kv_cache_type_str(self.cache_type_k).to_string(),
            "-ctv".to_string(),
            kv_cache_type_str(self.cache_type_v).to_string(),
            "-fit".to_string(),
            if self.fit { "on" } else { "off" }.to_string(),
            "-fitt".to_string(),
            self.fit_target_mib
                .first()
                .copied()
                .unwrap_or(1024)
                .to_string(),
            "-fitc".to_string(),
            self.ctx_size.to_string(),
        ];

        match self.n_gpu_layers {
            GpuLayerSpec::Auto => {
                args.push("-ngl".to_string());
                args.push("auto".to_string());
            }
            GpuLayerSpec::All => {
                args.push("-ngl".to_string());
                args.push("all".to_string());
            }
            GpuLayerSpec::Exact(n) => {
                args.push("-ngl".to_string());
                args.push(n.to_string());
            }
        }

        args.push("-sm".to_string());
        args.push(
            match self.split_mode {
                SplitMode::None => "none",
                SplitMode::Layer => "layer",
                SplitMode::Row => "row",
            }
            .to_string(),
        );

        if let Some(split) = &self.tensor_split {
            args.push("-ts".to_string());
            args.push(
                split
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }

        if let Some(main_gpu) = self.main_gpu {
            args.push("-mg".to_string());
            args.push(main_gpu.to_string());
        }

        args.push(if self.kv_offload {
            "-kvo".to_string()
        } else {
            "-nkvo".to_string()
        });

        args.push(if self.repack {
            "--repack".to_string()
        } else {
            "--no-repack".to_string()
        });

        args.push(if self.mmap {
            "--mmap".to_string()
        } else {
            "--no-mmap".to_string()
        });

        if self.mlock {
            args.push("--mlock".to_string());
        }

        if let Some(numa) = self.numa {
            args.push("--numa".to_string());
            args.push(
                match numa {
                    NumaMode::Distribute => "distribute",
                    NumaMode::Isolate => "isolate",
                }
                .to_string(),
            );
        }

        if self.cont_batching {
            args.push("--cont-batching".to_string());
        } else {
            args.push("--no-cont-batching".to_string());
        }

        if self.no_host {
            args.push("--no-host".to_string());
        }

        if let Some(poll) = self.poll {
            args.push("--poll".to_string());
            args.push(poll.to_string());
        }

        if let Some(poll_batch) = self.poll_batch {
            args.push("--poll-batch".to_string());
            args.push(poll_batch.to_string());
        }

        if let Some(cpu_mask) = &self.cpu_mask {
            args.push("--cpu-mask".to_string());
            args.push(cpu_mask.clone());
        }

        if let Some(cpu_range) = &self.cpu_range {
            args.push("--cpu-range".to_string());
            args.push(cpu_range.clone());
        }

        if self.cpu_strict {
            args.push("--cpu-strict".to_string());
        }

        if let Some(draft_model) = &self.draft_model_path {
            args.push("--model-draft".to_string());
            args.push(draft_model.display().to_string());

            match self.draft_gpu_layers {
                GpuLayerSpec::Auto => {
                    args.push("--draft-n-gl".to_string());
                    args.push("auto".to_string());
                }
                GpuLayerSpec::All => {
                    args.push("--draft-n-gl".to_string());
                    args.push("all".to_string());
                }
                GpuLayerSpec::Exact(n) => {
                    args.push("--draft-n-gl".to_string());
                    args.push(n.to_string());
                }
            }

            args.push("-dt".to_string());
            args.push(self.draft_threads.to_string());

            if let Some(draft_cache) = &self.draft_cache_type {
                args.push("--draft-cache-type".to_string());
                args.push(kv_cache_type_str(*draft_cache).to_string());
            }
        }

        if let Some(spec) = self.spec_type {
            args.push("--spec-type".to_string());
            args.push(
                match spec {
                    SpecType::Eager => "eager",
                    SpecType::NGram => "ngram",
                }
                .to_string(),
            );

            if let Some(extra) = &self.spec_extra {
                args.push("--spec-extra".to_string());
                args.push(extra.clone());
            }
        }

        args
    }

    pub fn fingerprint(&self) -> String {
        format!(
            "ctx={};b={};ub={};t={};tb={};np={};ngl={:?};sm={:?};ts={:?};mg={:?};fa={:?};ctk={:?};ctv={:?};kvo={};rep={};mmap={};mlock={};numa={:?};cb={};nh={};poll={:?};pollb={:?};cpu_mask={:?};cpu_range={:?};cpu_strict={};draft={:?};draft_ngl={:?};draft_t={:?};draft_cache={:?};spec={:?};spec_extra={:?};fit={};fitt={:?}",
            self.ctx_size,
            self.batch_size,
            self.ubatch_size,
            self.threads,
            self.threads_batch,
            self.parallel,
            self.n_gpu_layers,
            self.split_mode,
            self.tensor_split,
            self.main_gpu,
            self.flash_attn,
            self.cache_type_k,
            self.cache_type_v,
            self.kv_offload,
            self.repack,
            self.mmap,
            self.mlock,
            self.numa,
            self.cont_batching,
            self.no_host,
            self.poll,
            self.poll_batch,
            self.cpu_mask,
            self.cpu_range,
            self.cpu_strict,
            self.draft_model_path.is_some(),
            self.draft_gpu_layers,
            Some(self.draft_threads),
            self.draft_cache_type,
            self.spec_type,
            self.spec_extra,
            self.fit,
            self.fit_target_mib,
        )
    }

    pub fn candidate_key(&self) -> CandidateKey {
        let tensor_split_str = self.tensor_split.as_ref().map(|v| {
            v.iter()
                .map(|f| format!("{:.6}", f))
                .collect::<Vec<_>>()
                .join(",")
        });
        CandidateKey {
            ctx_size: self.ctx_size,
            batch_size: self.batch_size,
            ubatch_size: self.ubatch_size,
            threads: self.threads,
            threads_batch: self.threads_batch,
            parallel: self.parallel,
            n_gpu_layers: self.n_gpu_layers,
            split_mode: self.split_mode,
            tensor_split: tensor_split_str,
            main_gpu: self.main_gpu,
            flash_attn: self.flash_attn,
            cache_type_k: self.cache_type_k,
            cache_type_v: self.cache_type_v,
            kv_offload: self.kv_offload,
            repack: self.repack,
            mmap: self.mmap,
            mlock: self.mlock,
            numa: self.numa,
            cont_batching: self.cont_batching,
            no_host: self.no_host,
            poll: self.poll,
            poll_batch: self.poll_batch,
            cpu_mask: self.cpu_mask.clone(),
            cpu_range: self.cpu_range.clone(),
            cpu_strict: self.cpu_strict,
            draft_model_path: self.draft_model_path.is_some(),
            draft_gpu_layers: self.draft_gpu_layers,
            draft_threads: self.draft_threads,
            draft_cache_type: self.draft_cache_type,
            spec_type: self.spec_type,
            spec_extra: self.spec_extra.clone(),
            fit: self.fit,
            fit_target_mib: self.fit_target_mib.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandidateKey {
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: u16,
    pub threads_batch: u16,
    pub parallel: u16,
    pub n_gpu_layers: GpuLayerSpec,
    pub split_mode: SplitMode,
    pub tensor_split: Option<String>,
    pub main_gpu: Option<u32>,
    pub flash_attn: FlashAttn,
    pub cache_type_k: KvCacheType,
    pub cache_type_v: KvCacheType,
    pub kv_offload: bool,
    pub repack: bool,
    pub mmap: bool,
    pub mlock: bool,
    pub numa: Option<NumaMode>,
    pub cont_batching: bool,
    pub no_host: bool,
    pub poll: Option<u32>,
    pub poll_batch: Option<u32>,
    pub cpu_mask: Option<String>,
    pub cpu_range: Option<String>,
    pub cpu_strict: bool,
    pub draft_model_path: bool,
    pub draft_gpu_layers: GpuLayerSpec,
    pub draft_threads: u16,
    pub draft_cache_type: Option<KvCacheType>,
    pub spec_type: Option<SpecType>,
    pub spec_extra: Option<String>,
    pub fit: bool,
    pub fit_target_mib: Vec<u32>,
}

impl CandidateKey {
    pub fn fingerprint(&self) -> String {
        let ts = self.tensor_split.clone().unwrap_or_default();
        format!(
            "ctx={};b={};ub={};t={};tb={};np={};ngl={:?};sm={:?};ts={};mg={:?};fa={:?};ctk={:?};ctv={:?};kvo={};rep={};mmap={};mlock={};numa={:?};cb={};nh={};poll={:?};pollb={:?};cpu_mask={:?};cpu_range={:?};cpu_strict={};draft={};draft_ngl={:?};draft_t={};draft_cache={:?};spec={:?};spec_extra={:?};fit={};fitt={:?}",
            self.ctx_size, self.batch_size, self.ubatch_size, self.threads, self.threads_batch,
            self.parallel, self.n_gpu_layers, self.split_mode, ts, self.main_gpu,
            self.flash_attn, self.cache_type_k, self.cache_type_v, self.kv_offload,
            self.repack, self.mmap, self.mlock, self.numa, self.cont_batching, self.no_host,
            self.poll, self.poll_batch, self.cpu_mask, self.cpu_range, self.cpu_strict,
            self.draft_model_path, self.draft_gpu_layers, self.draft_threads,
            self.draft_cache_type, self.spec_type, self.spec_extra, self.fit, self.fit_target_mib
        )
    }
}

impl PartialEq for CandidateKey {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint() == other.fingerprint()
    }
}

impl Eq for CandidateKey {}

impl std::hash::Hash for CandidateKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.fingerprint().hash(state);
    }
}

impl From<&LlamaCppProfile> for CandidateKey {
    fn from(profile: &LlamaCppProfile) -> Self {
        profile.candidate_key()
    }
}

#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    pub benchmark_ctx_size: Option<u32>,
    pub prompt_tokens: u32,
    pub generation_tokens: u32,
    pub parallel_requests: u16,
    pub max_rounds: usize,
    pub min_relative_gain: f32,
    pub warmup_samples: usize,
    pub benchmark_samples: usize,
    pub early_stop_samples: usize,
    pub racing_keep_fraction: f32,
    pub coarse_budget_ms: u32,
    pub fine_budget_ms: u32,
    pub search_strategy: SearchStrategy,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            benchmark_ctx_size: Some(4096),
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
            search_strategy: SearchStrategy::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SearchStrategy {
    Greedy,
    #[default]
    Racing,
    Exhaustive,
    Fast,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ReasonTag {
    SeededFromMetadata,
    GpuSplitAdjusted,
    KvCacheDownshifted,
    BenchmarkFailedRetried,
    MemoryFitAdjusted,
    ThreadsOptimized,
    BatchSizeOptimized,
    FlashAttnToggled,
    ContBatchingToggled,
    CpuAffinityApplied,
    SpecDecodingEnabled,
    NumaModeApplied,
    CoordinateRefinement,
    StageSurvivor,
}

impl fmt::Display for ReasonTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SeededFromMetadata => write!(f, "seeded_from_metadata"),
            Self::GpuSplitAdjusted => write!(f, "gpu_split_adjusted"),
            Self::KvCacheDownshifted => write!(f, "kv_cache_downshifted"),
            Self::BenchmarkFailedRetried => write!(f, "benchmark_failed_retried"),
            Self::MemoryFitAdjusted => write!(f, "memory_fit_adjusted"),
            Self::ThreadsOptimized => write!(f, "threads_optimized"),
            Self::BatchSizeOptimized => write!(f, "batch_size_optimized"),
            Self::FlashAttnToggled => write!(f, "flash_attn_toggled"),
            Self::ContBatchingToggled => write!(f, "cont_batching_toggled"),
            Self::CpuAffinityApplied => write!(f, "cpu_affinity_applied"),
            Self::SpecDecodingEnabled => write!(f, "spec_decoding_enabled"),
            Self::NumaModeApplied => write!(f, "numa_mode_applied"),
            Self::CoordinateRefinement => write!(f, "coordinate_refinement"),
            Self::StageSurvivor => write!(f, "stage_survivor"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnReason {
    pub tag: ReasonTag,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearnedProfile {
    pub overrides: std::collections::HashMap<String, String>,
    pub reasons: Vec<LearnReason>,
    pub baseline_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub prompt_tps: f32,
    pub decode_tps: f32,
    pub latency_ms: f32,
    pub memory_mib: u32,
    pub stability: f32,
    pub run_count: u32,
    pub failure_count: u32,
    pub prompt_tps_variance: f32,
    pub decode_tps_variance: f32,
}

impl BenchmarkResult {
    pub fn from_runs(runs: Vec<BenchmarkRun>) -> Option<Self> {
        if runs.is_empty() {
            return None;
        }

        let failures = runs.iter().filter(|r| r.failed).count() as u32;
        let success_runs: Vec<&BenchmarkRun> = runs.iter().filter(|r| !r.failed).collect();

        if success_runs.is_empty() {
            return None;
        }

        let prompt_tps_values: Vec<f32> = success_runs.iter().map(|r| r.prompt_tps).collect();
        let decode_tps_values: Vec<f32> = success_runs.iter().map(|r| r.decode_tps).collect();
        let latency_values: Vec<f32> = success_runs.iter().map(|r| r.latency_ms).collect();

        let prompt_tps = trimmed_mean(&prompt_tps_values, 0.1);
        let decode_tps = trimmed_mean(&decode_tps_values, 0.1);
        let latency_ms = trimmed_mean(&latency_values, 0.1);

        let prompt_tps_variance = variance(&prompt_tps_values);
        let decode_tps_variance = variance(&decode_tps_values);

        let stability = 1.0 - (failures as f32 / runs.len() as f32);

        Some(Self {
            prompt_tps,
            decode_tps,
            latency_ms,
            memory_mib: success_runs.first().map(|r| r.memory_mib).unwrap_or(0),
            stability,
            run_count: runs.len() as u32,
            failure_count: failures,
            prompt_tps_variance,
            decode_tps_variance,
        })
    }

    pub fn combined_score(&self) -> f32 {
        let tps = (self.prompt_tps + self.decode_tps) / 2.0;
        let latency_score = 1000.0 / (self.latency_ms + 1.0);
        let stability_weight = 0.25;
        let variance_penalty = self.variance_penalty();
        tps * (1.0 - stability_weight) * (1.0 - variance_penalty)
            + latency_score * stability_weight * 10.0
    }

    pub fn throughput_score(&self) -> f32 {
        let tps = self.prompt_tps * 0.25 + self.decode_tps * 0.75;
        let stability_weight = 0.15;
        let variance_penalty = self.variance_penalty();
        tps * (1.0 - stability_weight) * (1.0 - variance_penalty)
    }

    pub fn variance_penalty(&self) -> f32 {
        let combined_variance = (self.prompt_tps_variance + self.decode_tps_variance) / 2.0;
        let mean_tps = (self.prompt_tps + self.decode_tps) / 2.0;
        if mean_tps <= 0.0 {
            return 0.0;
        }
        let cv = (combined_variance.sqrt()) / mean_tps;
        (cv.min(1.0) * 0.3).min(0.3)
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkRun {
    pub prompt_tps: f32,
    pub decode_tps: f32,
    pub latency_ms: f32,
    pub memory_mib: u32,
    pub failed: bool,
}

pub fn trimmed_mean(values: &[f32], trim_fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let trim_count = ((sorted.len() as f32) * trim_fraction).floor() as usize;
    let trim_count = trim_count.min(sorted.len() / 2);

    if trim_count >= sorted.len() {
        return values.iter().sum::<f32>() / values.len() as f32;
    }

    let trimmed = &sorted[trim_count..sorted.len() - trim_count];
    if trimmed.is_empty() {
        return values.iter().sum::<f32>() / values.len() as f32;
    }
    trimmed.iter().sum::<f32>() / trimmed.len() as f32
}

pub fn variance(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    pub profile_key: String,
    pub model_fingerprint: String,
    pub hardware_fingerprint: String,
}

pub struct SearchCache {
    results: std::collections::HashMap<CacheKey, BenchmarkResult>,
}

impl SearchCache {
    pub fn new() -> Self {
        Self {
            results: std::collections::HashMap::new(),
        }
    }

    pub fn get(&self, key: &CacheKey) -> Option<&BenchmarkResult> {
        self.results.get(key)
    }

    pub fn insert(&mut self, key: CacheKey, result: BenchmarkResult) {
        self.results.insert(key, result);
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

impl Default for SearchCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FeasibilityResult {
    pub viable: bool,
    pub reason: Option<String>,
    pub weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub batch_buffer_bytes: u64,
    pub host_buffer_bytes: u64,
    pub total_needed: u64,
    pub vram_available: u64,
    pub ram_available: u64,
}

impl FeasibilityResult {
    pub fn estimate(
        profile: &LlamaCppProfile,
        facts: &GgufFacts,
        hardware: &HardwareProfile,
    ) -> Self {
        let weight_bytes = estimate_weight_bytes(profile, facts);
        let kv_cache_bytes = estimate_kv_cache_bytes(profile, facts);
        let batch_buffer_bytes = estimate_batch_buffer_bytes(profile, facts);
        let host_buffer_bytes = estimate_host_buffer_overhead(profile, facts);

        let total_needed = weight_bytes + kv_cache_bytes + batch_buffer_bytes + host_buffer_bytes;

        let (vram_available, ram_available) = if hardware.gpus.is_empty() {
            (0, hardware.system_ram_bytes)
        } else {
            let total_vram: u64 = hardware.gpus.iter().map(|g| g.vram_bytes).sum();
            (total_vram, hardware.system_ram_bytes)
        };

        let has_gpu = !hardware.gpus.is_empty();

        let mut viable = true;
        let mut reason = None;

        if has_gpu {
            let offloaded_layers = match profile.n_gpu_layers {
                GpuLayerSpec::Exact(n) => n,
                GpuLayerSpec::All => facts.block_count.unwrap_or(0),
                GpuLayerSpec::Auto => facts.block_count.unwrap_or(0),
            };

            let offloaded_weight_bytes =
                if offloaded_layers > 0 && facts.block_count.unwrap_or(0) > 0 {
                    let ratio = offloaded_layers as f64 / facts.block_count.unwrap_or(1) as f64;
                    (weight_bytes as f64 * ratio) as u64
                } else {
                    0
                };

            let gpu_memory_needed = offloaded_weight_bytes + kv_cache_bytes + batch_buffer_bytes;

            if gpu_memory_needed > vram_available {
                viable = false;
                reason = Some(format!(
                    "GPU memory {} MB exceeds available {} MB",
                    gpu_memory_needed / 1024 / 1024,
                    vram_available / 1024 / 1024
                ));
            }
        } else {
            if total_needed > ram_available {
                viable = false;
                reason = Some(format!(
                    "RAM {} MB exceeds available {} MB",
                    total_needed / 1024 / 1024,
                    ram_available / 1024 / 1024
                ));
            }
        }

        if let Some(fit_target) = profile.fit_target_mib.first() {
            let margin_bytes = if has_gpu {
                (vram_available as i64 - total_needed as i64).max(0) as u64
            } else {
                (ram_available as i64 - total_needed as i64).max(0) as u64
            };
            let margin_mib = margin_bytes / 1024 / 1024;

            if margin_mib < (*fit_target as u64) {
                viable = false;
                reason = Some(format!(
                    "Memory margin {} MiB below target {} MiB",
                    margin_mib, fit_target
                ));
            }
        }

        Self {
            viable,
            reason,
            weight_bytes,
            kv_cache_bytes,
            batch_buffer_bytes,
            host_buffer_bytes,
            total_needed,
            vram_available,
            ram_available,
        }
    }
}

fn estimate_weight_bytes(_profile: &LlamaCppProfile, facts: &GgufFacts) -> u64 {
    facts.weight_bytes
}

pub fn estimate_kv_cache_bytes(profile: &LlamaCppProfile, facts: &GgufFacts) -> u64 {
    let ctx = profile.ctx_size as u64;
    let batch = profile.batch_size as u64;
    let layers = facts.block_count.unwrap_or(32) as u64;
    let heads = facts.attention_head_count.unwrap_or(32) as u64;
    let kv_heads = facts.attention_head_count_kv.unwrap_or(heads as u32) as u64;
    let hidden = facts.embedding_length.unwrap_or(4096) as u64;

    let head_dim = hidden / heads.max(1);

    let k_bytes = ctx
        .saturating_mul(batch)
        .saturating_mul(layers)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2);
    let v_bytes = k_bytes;

    let type_size: f64 = match profile.cache_type_k {
        KvCacheType::F32 => 4.0,
        KvCacheType::F16 => 2.0,
        KvCacheType::BF16 => 2.0,
        KvCacheType::Q8_0 => 1.0,
        KvCacheType::Q4_0 => 0.5,
        KvCacheType::Q4_1 => 0.5,
        KvCacheType::IQ4_NL => 0.5,
        KvCacheType::Q5_0 => 0.625,
        KvCacheType::Q5_1 => 0.625,
    };

    ((k_bytes.saturating_add(v_bytes) as f64) * type_size) as u64
}

pub fn estimate_batch_buffer_bytes(profile: &LlamaCppProfile, facts: &GgufFacts) -> u64 {
    let batch = profile.ubatch_size.max(profile.batch_size) as u64;
    let ctx = profile.ctx_size as u64;
    let hidden = facts.embedding_length.unwrap_or(4096) as u64;
    let layers = facts.block_count.unwrap_or(32) as u64;

    let bytes_per_token = hidden.saturating_mul(2);
    batch
        .saturating_mul(ctx)
        .saturating_mul(bytes_per_token)
        .saturating_mul(layers)
        .saturating_div(8)
}

fn estimate_host_buffer_overhead(_profile: &LlamaCppProfile, facts: &GgufFacts) -> u64 {
    let overhead_per_layer = 1024 * 1024;
    let layers = facts.block_count.unwrap_or(32);
    (layers as u64) * overhead_per_layer
}

pub fn estimate_max_gpu_layers(
    facts: &GgufFacts,
    hardware: &HardwareProfile,
    profile: &LlamaCppProfile,
) -> u32 {
    if hardware.gpus.is_empty() {
        return 0;
    }

    let vram: u64 = hardware.gpus.iter().map(|g| g.vram_bytes).sum();
    let weight_bytes = facts.weight_bytes;

    let kv_bytes = estimate_kv_cache_bytes(profile, facts);
    let batch_bytes = estimate_batch_buffer_bytes(profile, facts);

    let available_for_weights = vram.saturating_sub(kv_bytes + batch_bytes);
    let available_for_weights = (available_for_weights as f64 * 0.85) as u64;

    let total_layers = facts.block_count.unwrap_or(32);
    if weight_bytes == 0 || total_layers == 0 {
        return 0;
    }

    let per_layer_bytes = weight_bytes / total_layers as u64;
    if per_layer_bytes == 0 {
        return total_layers;
    }

    (available_for_weights / per_layer_bytes).min(total_layers as u64) as u32
}

pub fn is_context_plausible(facts: &GgufFacts, ctx_size: u32) -> bool {
    let max_plausible = estimate_max_plausible_context(facts);
    ctx_size <= max_plausible
}

fn estimate_max_plausible_context(facts: &GgufFacts) -> u32 {
    let base_ctx = facts.context_length.unwrap_or(4096) as u64;

    let rope_scaling = match facts.rope_scaling_type.as_deref() {
        Some("linear") | Some("yarn") => {
            let factor = facts.rope_scaling_factor.unwrap_or(1.0) as u64;
            base_ctx * factor.max(1)
        }
        _ => base_ctx,
    };

    let original_ctx = facts
        .rope_scaling_original_context_length
        .unwrap_or(base_ctx as u32) as u64;

    rope_scaling.max(original_ctx).min(1_000_000) as u32
}

pub trait GgufFactsExtractor {
    fn extract(&self, path: &Path) -> Result<GgufFacts, OptimizeError>;
}

#[derive(Debug)]
pub enum OptimizeError {
    Io(std::io::Error),
    Parse(String),
    Benchmark(String),
    NoValidConfiguration,
}

#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub profile: LlamaCppProfile,
    pub stats: SearchStats,
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizeError::Io(e) => write!(f, "I/O error: {e}"),
            OptimizeError::Parse(e) => write!(f, "parse error: {e}"),
            OptimizeError::Benchmark(e) => write!(f, "benchmark error: {e}"),
            OptimizeError::NoValidConfiguration => write!(f, "no valid configuration found"),
        }
    }
}

impl std::error::Error for OptimizeError {}

impl From<std::io::Error> for OptimizeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub fn optimize_llama_cpp_profile<E, F>(
    gguf_path: impl AsRef<Path>,
    hardware: HardwareProfile,
    extractor: &E,
    mut benchmark: F,
    opts: OptimizeOptions,
) -> Result<OptimizeResult, OptimizeError>
where
    E: GgufFactsExtractor,
    F: FnMut(&LlamaCppProfile) -> Result<BenchmarkResult, OptimizeError>,
{
    let path = gguf_path.as_ref();
    let facts = match extractor.extract(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("⚠ GGUF parse failed, falling back: {}", e);
            GgufFacts {
                path: path.to_path_buf(),
                architecture: "unknown".to_string(),
                model_name: None,
                size_label: None,
                quantization_version: None,
                file_type: None,
                alignment: None,
                context_length: Some(4096),
                embedding_length: None,
                block_count: None,
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

    let model_fingerprint = format!(
        "{}:{}:{}:{}",
        facts.architecture,
        facts.context_length.unwrap_or(0),
        facts.block_count.unwrap_or(0),
        facts.weight_bytes
    );
    let hardware_fingerprint = format!(
        "gpu{}:ram{}:cores{}",
        hardware.gpus.len(),
        hardware.system_ram_bytes,
        hardware.cpu_physical_cores
    );

    let seed = seed_profile(path.to_path_buf(), facts.clone(), &hardware, &opts);
    let _seed_fingerprint = seed.fingerprint();

    let mut engine = search::SearchEngine::new();

    let mut best = engine.search(
        opts.search_strategy,
        &seed,
        &facts,
        &hardware,
        &opts,
        &mut benchmark,
        &model_fingerprint,
        &hardware_fingerprint,
    )?;

    let stats = engine.stats();

    best.notes.push(ReasonTag::SeededFromMetadata.to_string());
    best.notes.push(format!(
        "prompt_tps={:.1}, decode_tps={:.1}, latency={:.0}ms",
        best.estimated_result
            .as_ref()
            .map(|r| r.prompt_tps)
            .unwrap_or(0.0),
        best.estimated_result
            .as_ref()
            .map(|r| r.decode_tps)
            .unwrap_or(0.0),
        best.estimated_result
            .as_ref()
            .map(|r| r.latency_ms)
            .unwrap_or(0.0)
    ));

    best.facts = facts;
    Ok(OptimizeResult {
        profile: best,
        stats,
    })
}

fn seed_profile(
    model_path: PathBuf,
    facts: GgufFacts,
    hardware: &HardwareProfile,
    opts: &OptimizeOptions,
) -> LlamaCppProfile {
    let model_ctx = facts.context_length.unwrap_or(4096);
    let ctx_size = opts
        .benchmark_ctx_size
        .unwrap_or(model_ctx)
        .min(model_ctx.max(1));

    let cpu_physical = hardware.cpu_physical_cores.max(1);
    let cpu_logical = hardware.cpu_logical_cores.max(cpu_physical);

    let threads = if hardware.gpus.is_empty() {
        cpu_logical
    } else {
        cpu_physical
    }
    .max(1);

    let threads_batch = (threads.saturating_mul(2)).min(cpu_logical).max(threads);

    let has_gpu = !hardware.gpus.is_empty();
    let multi_gpu = hardware.gpus.len() > 1;

    let tensor_split = if multi_gpu {
        Some(normalized_vram_split(&hardware.gpus))
    } else {
        None
    };

    let fit_target_mib = if hardware.gpus.is_empty() {
        vec![1024]
    } else {
        hardware
            .gpus
            .iter()
            .map(|g| {
                let mib = (g.vram_bytes / 1024 / 1024) as u32;
                mib.saturating_div(16).clamp(256, 2048)
            })
            .collect()
    };

    LlamaCppProfile {
        model_path,
        ctx_size,
        batch_size: 2048,
        ubatch_size: 512,
        threads,
        threads_batch,
        parallel: opts.parallel_requests.max(1),
        n_gpu_layers: if has_gpu {
            GpuLayerSpec::Auto
        } else {
            GpuLayerSpec::Exact(0)
        },
        split_mode: if multi_gpu {
            SplitMode::Layer
        } else {
            SplitMode::None
        },
        tensor_split,
        main_gpu: hardware.gpus.first().map(|g| g.index),
        flash_attn: if has_gpu {
            FlashAttn::On
        } else {
            FlashAttn::Off
        },
        cache_type_k: if has_gpu {
            KvCacheType::Q8_0
        } else {
            KvCacheType::F32
        },
        cache_type_v: if has_gpu {
            KvCacheType::Q8_0
        } else {
            KvCacheType::F32
        },
        kv_offload: has_gpu,
        repack: true,
        mmap: true,
        mlock: false,
        numa: hardware
            .numa_nodes
            .filter(|n| *n > 1)
            .map(|_| NumaMode::Distribute),
        cont_batching: true,
        no_host: false,
        poll: None,
        poll_batch: None,
        cpu_mask: None,
        cpu_range: None,
        cpu_strict: false,
        draft_model_path: None,
        draft_gpu_layers: GpuLayerSpec::Exact(0),
        draft_threads: 1,
        draft_cache_type: None,
        spec_type: None,
        spec_extra: None,
        fit: true,
        fit_target_mib,
        estimated_result: None,
        facts,
        notes: vec!["seeded from GGUF metadata".to_string()],
    }
}

fn initial_variants(
    base: &LlamaCppProfile,
    facts: &GgufFacts,
    hardware: &HardwareProfile,
    opts: &OptimizeOptions,
) -> Vec<LlamaCppProfile> {
    let mut out = vec![base.clone()];

    let mut bigger_batch = base.clone();
    bigger_batch.batch_size = next_pow2(base.batch_size.saturating_mul(2), 4096);
    bigger_batch.ubatch_size = next_pow2(base.ubatch_size.saturating_mul(2), 2048);
    out.push(bigger_batch);

    let mut smaller_batch = base.clone();
    smaller_batch.batch_size = next_pow2((base.batch_size / 2).max(32), 4096);
    smaller_batch.ubatch_size = next_pow2((base.ubatch_size / 2).max(16), 2048);
    out.push(smaller_batch);

    if !hardware.gpus.is_empty() {
        let mut all_gpu = base.clone();
        all_gpu.n_gpu_layers = GpuLayerSpec::All;
        all_gpu.flash_attn = FlashAttn::On;
        out.push(all_gpu);

        let mut q8_kv = base.clone();
        q8_kv.cache_type_k = KvCacheType::Q8_0;
        q8_kv.cache_type_v = KvCacheType::Q8_0;
        out.push(q8_kv);

        let mut bf16 = base.clone();
        if hardware.gpus.iter().any(|g| g.supports_bf16) {
            bf16.cache_type_k = KvCacheType::BF16;
            bf16.cache_type_v = KvCacheType::BF16;
            out.push(bf16);
        }
    }

    if let Some(layers) = facts.block_count {
        let mut exact_full = base.clone();
        exact_full.n_gpu_layers = GpuLayerSpec::Exact(layers);
        out.push(exact_full);

        let mut half = base.clone();
        half.n_gpu_layers = GpuLayerSpec::Exact((layers / 2).max(1));
        out.push(half);

        let mut quarter = base.clone();
        quarter.n_gpu_layers = GpuLayerSpec::Exact((layers / 4).max(1));
        out.push(quarter);
    }

    if hardware.gpus.len() > 1 {
        let mut row = base.clone();
        row.split_mode = SplitMode::Row;
        out.push(row);
    }

    if opts.prompt_tokens > 1024 {
        let mut more_threads = base.clone();
        more_threads.threads_batch = next_pow2(
            base.threads_batch as u32 * 2,
            (hardware.cpu_logical_cores as u32) * 2,
        ) as u16;
        out.push(more_threads);
    }

    dedupe(out)
}

fn coordinate_variants(
    best: &LlamaCppProfile,
    facts: &GgufFacts,
    hardware: &HardwareProfile,
    opts: &OptimizeOptions,
) -> Vec<LlamaCppProfile> {
    let mut out = Vec::new();

    for threads in candidate_threads(hardware, best.threads) {
        let mut p = best.clone();
        p.threads = threads;
        p.threads_batch = p.threads_batch.max(threads);
        out.push(p);
    }

    for tb in candidate_threads_batch(hardware, best.threads_batch, best.threads) {
        let mut p = best.clone();
        p.threads_batch = tb;
        out.push(p);
    }

    for batch in candidate_batches(best.batch_size) {
        let mut p = best.clone();
        p.batch_size = batch;
        out.push(p);
    }

    for ubatch in candidate_ubatches(best.ubatch_size, best.batch_size) {
        let mut p = best.clone();
        p.ubatch_size = ubatch.min(best.batch_size).max(1);
        out.push(p);
    }

    for ngl in candidate_gpu_layers(facts.block_count, hardware.gpus.len(), best.n_gpu_layers) {
        let mut p = best.clone();
        p.n_gpu_layers = ngl;
        out.push(p);
    }

    for flash in candidate_flash_attn(hardware, best.flash_attn) {
        let mut p = best.clone();
        p.flash_attn = flash;
        out.push(p);
    }

    for k_type in candidate_kv_cache_types(hardware, best.cache_type_k) {
        for v_type in candidate_kv_cache_types(hardware, best.cache_type_v) {
            let mut p = best.clone();
            p.cache_type_k = k_type;
            p.cache_type_v = v_type;
            out.push(p);
        }
    }

    if hardware.gpus.len() > 1 {
        for split_mode in [SplitMode::Layer, SplitMode::Row] {
            let mut p = best.clone();
            p.split_mode = split_mode;
            p.tensor_split = Some(normalized_vram_split(&hardware.gpus));
            out.push(p);
        }
    }

    if hardware.gpus.is_empty() {
        let mut p = best.clone();
        p.kv_offload = false;
        p.flash_attn = FlashAttn::Off;
        p.n_gpu_layers = GpuLayerSpec::Exact(0);
        out.push(p);
    }

    for cb in candidate_cont_batching(best.cont_batching, opts) {
        let mut p = best.clone();
        p.cont_batching = cb;
        out.push(p);
    }

    if hardware.gpus.is_empty() {
        for poll in candidate_poll(best.poll) {
            let mut p = best.clone();
            p.poll = Some(poll);
            out.push(p);
        }

        for poll_batch in candidate_poll_batch(best.poll_batch) {
            let mut p = best.clone();
            p.poll_batch = Some(poll_batch);
            out.push(p);
        }

        if hardware.cpu_physical_cores > 4 {
            for mask in candidate_cpu_mask(hardware) {
                let mut p = best.clone();
                p.cpu_mask = Some(mask);
                out.push(p);
            }

            for range in candidate_cpu_range(hardware) {
                let mut p = best.clone();
                p.cpu_range = Some(range);
                out.push(p);
            }
        }
    }

    if hardware.gpus.is_empty() && opts.generation_tokens > 32 {
        for spec in candidate_spec_type() {
            let mut p = best.clone();
            p.spec_type = Some(spec);
            if spec == SpecType::NGram {
                p.spec_extra = Some(format!("n={}", (opts.generation_tokens / 4).clamp(2, 8)));
            }
            out.push(p);
        }
    }

    dedupe(out)
}

fn candidate_threads(hardware: &HardwareProfile, current: u16) -> Vec<u16> {
    let phys = hardware.cpu_physical_cores.max(1);
    let logi = hardware.cpu_logical_cores.max(phys);

    dedupe_u16(vec![
        current,
        phys,
        logi,
        phys.saturating_sub(1).max(1),
        (phys / 2).max(1),
    ])
}

fn candidate_threads_batch(hardware: &HardwareProfile, current: u16, threads: u16) -> Vec<u16> {
    let phys = hardware.cpu_physical_cores.max(1);
    let logi = hardware.cpu_logical_cores.max(phys);

    dedupe_u16(vec![
        current,
        threads,
        phys,
        logi,
        threads.saturating_mul(2).min(logi).max(1),
    ])
}

pub fn candidate_batches(current: u32) -> Vec<u32> {
    dedupe_u32(vec![
        current,
        next_pow2(current / 2, 4096).max(32),
        next_pow2(current.saturating_mul(2), 4096),
        256,
        512,
        1024,
        2048,
        4096,
    ])
}

pub fn candidate_ubatches(current: u32, batch: u32) -> Vec<u32> {
    dedupe_u32(vec![
        current,
        next_pow2(current / 2, batch.max(1)),
        next_pow2(current.saturating_mul(2), batch.max(1)),
        32,
        64,
        128,
        256,
        512,
        1024,
    ])
    .into_iter()
    .map(|v| v.min(batch.max(1)))
    .filter(|v| *v > 0)
    .collect()
}

pub fn candidate_gpu_layers(
    block_count: Option<u32>,
    gpu_count: usize,
    current: GpuLayerSpec,
) -> Vec<GpuLayerSpec> {
    let mut out = vec![current, GpuLayerSpec::Auto, GpuLayerSpec::All];

    if let Some(layers) = block_count {
        out.push(GpuLayerSpec::Exact(layers));
        out.push(GpuLayerSpec::Exact((layers / 2).max(1)));
        out.push(GpuLayerSpec::Exact((layers / 4).max(1)));
        out.push(GpuLayerSpec::Exact((layers * 3 / 4).max(1)));
        out.push(GpuLayerSpec::Exact(0));
    } else if gpu_count == 0 {
        out.push(GpuLayerSpec::Exact(0));
    }

    dedupe_gpu_layers(out)
}

fn candidate_flash_attn(hardware: &HardwareProfile, current: FlashAttn) -> Vec<FlashAttn> {
    let has_gpu = !hardware.gpus.is_empty();
    if !has_gpu {
        return vec![FlashAttn::Off];
    }
    dedupe_flash(vec![
        current,
        FlashAttn::Auto,
        FlashAttn::On,
        FlashAttn::Off,
    ])
}

pub fn candidate_kv_cache_types(
    hardware: &HardwareProfile,
    current: KvCacheType,
) -> Vec<KvCacheType> {
    let has_gpu = !hardware.gpus.is_empty();
    if !has_gpu {
        return vec![KvCacheType::F32];
    }

    let mut out = vec![
        current,
        KvCacheType::F16,
        KvCacheType::BF16,
        KvCacheType::Q8_0,
    ];

    if hardware.system_ram_bytes < 32 * 1024 * 1024 * 1024 {
        out.push(KvCacheType::Q4_0);
        out.push(KvCacheType::Q4_1);
    }

    dedupe_kv(out)
}

pub fn candidate_cont_batching(current: bool, opts: &OptimizeOptions) -> Vec<bool> {
    let mut out = vec![current];
    if opts.generation_tokens <= 16 || opts.parallel_requests == 1 {
        out.push(false);
    }
    out
}

pub fn candidate_poll(current: Option<u32>) -> Vec<u32> {
    dedupe_u32(vec![current.unwrap_or(0), 0, 32, 64, 128])
}

fn candidate_poll_batch(current: Option<u32>) -> Vec<u32> {
    dedupe_u32(vec![current.unwrap_or(0), 0, 32, 64])
}

fn candidate_cpu_mask(hardware: &HardwareProfile) -> Vec<String> {
    let cores = hardware.cpu_physical_cores as usize;
    if cores <= 1 {
        return vec![];
    }

    let half = cores / 2;
    vec![
        format!("0-{}", cores - 1),
        format!("0-{}", half - 1),
        format!("{}-{}", half, cores - 1),
    ]
}

fn candidate_cpu_range(hardware: &HardwareProfile) -> Vec<String> {
    let cores = hardware.cpu_physical_cores as usize;
    if cores < 4 {
        return vec![];
    }

    let quarter = cores / 4;
    vec![
        format!("0@0-{}", quarter - 1),
        format!("0@{}-{}", quarter, quarter * 2 - 1),
    ]
}

fn candidate_spec_type() -> Vec<SpecType> {
    vec![SpecType::Eager, SpecType::NGram]
}

fn normalized_vram_split(gpus: &[GpuProfile]) -> Vec<f32> {
    let total: f64 = gpus.iter().map(|g| g.vram_bytes as f64).sum();
    if total <= 0.0 {
        return vec![1.0 / gpus.len().max(1) as f32; gpus.len()];
    }

    gpus.iter()
        .map(|g| (g.vram_bytes as f64 / total) as f32)
        .collect()
}

fn next_pow2(value: u32, ceiling: u32) -> u32 {
    if value <= 1 {
        return 1.min(ceiling);
    }
    let mut n = value - 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    (n + 1).min(ceiling.max(1))
}

fn kv_cache_type_str(t: KvCacheType) -> &'static str {
    match t {
        KvCacheType::F32 => "f32",
        KvCacheType::F16 => "f16",
        KvCacheType::BF16 => "bf16",
        KvCacheType::Q8_0 => "q8_0",
        KvCacheType::Q4_0 => "q4_0",
        KvCacheType::Q4_1 => "q4_1",
        KvCacheType::IQ4_NL => "iq4_nl",
        KvCacheType::Q5_0 => "q5_0",
        KvCacheType::Q5_1 => "q5_1",
    }
}

fn dedupe(mut items: Vec<LlamaCppProfile>) -> Vec<LlamaCppProfile> {
    let mut seen = HashSet::new();
    items.retain(|item| seen.insert(item.fingerprint()));
    items
}

fn dedupe_u16(items: Vec<u16>) -> Vec<u16> {
    let mut seen = HashSet::new();
    items.into_iter().filter(|v| seen.insert(*v)).collect()
}

fn dedupe_u32(items: Vec<u32>) -> Vec<u32> {
    let mut seen = HashSet::new();
    items.into_iter().filter(|v| seen.insert(*v)).collect()
}

fn dedupe_gpu_layers(items: Vec<GpuLayerSpec>) -> Vec<GpuLayerSpec> {
    let mut seen = HashSet::new();
    items.into_iter().filter(|v| seen.insert(*v)).collect()
}

fn dedupe_flash(items: Vec<FlashAttn>) -> Vec<FlashAttn> {
    let mut seen = HashSet::new();
    items.into_iter().filter(|v| seen.insert(*v)).collect()
}

fn dedupe_kv(items: Vec<KvCacheType>) -> Vec<KvCacheType> {
    let mut seen = HashSet::new();
    items.into_iter().filter(|v| seen.insert(*v)).collect()
}
