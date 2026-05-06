use crate::tuning::{
    FlashAttn, GgufFacts, GpuLayerSpec, HardwareProfile, KvCacheType, LlamaCppProfile, NumaMode,
    OptimizeOptions, SpecType, SplitMode,
};

pub fn generate_all_candidates(
    base: &LlamaCppProfile,
    facts: &GgufFacts,
    hardware: &HardwareProfile,
    opts: &OptimizeOptions,
) -> Vec<LlamaCppProfile> {
    let mut candidates = vec![];

    candidates.extend(generate_gpu_offload_candidates(base, hardware));
    candidates.extend(generate_split_mode_candidates(base, hardware));
    candidates.extend(generate_context_batch_candidates(
        base, facts, hardware, opts,
    ));
    candidates.extend(generate_thread_candidates(base, hardware));
    candidates.extend(generate_kv_cache_candidates(base, hardware));
    candidates.extend(generate_flash_attn_candidates(base, hardware));
    candidates.extend(generate_memory_fit_candidates(base, hardware));
    candidates.extend(generate_cpu_affinity_candidates(base, hardware));
    candidates.extend(generate_spec_candidates(base, hardware, opts));

    dedupe_profiles(candidates)
}

pub fn generate_stage_candidates(
    base: &LlamaCppProfile,
    facts: &GgufFacts,
    hardware: &HardwareProfile,
    opts: &OptimizeOptions,
    stage: Stage,
) -> Vec<LlamaCppProfile> {
    let candidates = match stage {
        Stage::GpuOffload => generate_gpu_offload_candidates(base, hardware),
        Stage::SplitMode => generate_split_mode_candidates(base, hardware),
        Stage::ContextBatch => generate_context_batch_candidates(base, facts, hardware, opts),
        Stage::Threads => generate_thread_candidates(base, hardware),
        Stage::KvCache => generate_kv_cache_candidates(base, hardware),
        Stage::FlashAttn => generate_flash_attn_candidates(base, hardware),
        Stage::MemoryFit => generate_memory_fit_candidates(base, hardware),
        Stage::CpuAffinity => generate_cpu_affinity_candidates(base, hardware),
        Stage::Speculative => generate_spec_candidates(base, hardware, opts),
    };
    dedupe_profiles(candidates)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    GpuOffload,
    SplitMode,
    ContextBatch,
    Threads,
    KvCache,
    FlashAttn,
    MemoryFit,
    CpuAffinity,
    Speculative,
}

impl Stage {
    pub fn all() -> Vec<Self> {
        vec![
            Self::GpuOffload,
            Self::SplitMode,
            Self::ContextBatch,
            Self::Threads,
            Self::KvCache,
            Self::FlashAttn,
            Self::MemoryFit,
            Self::CpuAffinity,
            Self::Speculative,
        ]
    }

    pub fn tier(&self) -> &'static str {
        match self {
            Self::GpuOffload => "high",
            Self::SplitMode => "high",
            Self::ContextBatch => "high",
            Self::Threads => "medium",
            Self::KvCache => "medium",
            Self::FlashAttn => "medium",
            Self::MemoryFit => "low",
            Self::CpuAffinity => "low",
            Self::Speculative => "low",
        }
    }
}

pub fn generate_gpu_offload_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
) -> Vec<LlamaCppProfile> {
    if hardware.gpus.is_empty() {
        return vec![];
    }

    let block_count = base.facts.block_count.unwrap_or(32);
    let max_layers = crate::tuning::estimate_max_gpu_layers(&base.facts, hardware, base);

    let mut candidates = vec![];

    candidates.push({
        let mut p = base.clone();
        p.n_gpu_layers = GpuLayerSpec::Exact(0);
        p.kv_offload = false;
        p
    });

    if max_layers > 0 {
        for l in [1, 2, 4, 8, 16] {
            if l <= max_layers && l <= block_count {
                let mut p = base.clone();
                p.n_gpu_layers = GpuLayerSpec::Exact(l);
                p.kv_offload = true;
                candidates.push(p);
            }
        }
    }

    if max_layers > block_count / 2 {
        for frac in [0.25, 0.5, 0.75, 1.0] {
            let layers = ((block_count as f32) * frac).ceil() as u32;
            if layers > 0 && layers <= max_layers {
                let mut p = base.clone();
                p.n_gpu_layers = GpuLayerSpec::Exact(layers);
                p.kv_offload = true;
                candidates.push(p);
            }
        }
    }

    candidates.push({
        let mut p = base.clone();
        p.n_gpu_layers = GpuLayerSpec::Auto;
        p.kv_offload = true;
        p
    });

    candidates.push({
        let mut p = base.clone();
        p.n_gpu_layers = GpuLayerSpec::All;
        p.kv_offload = true;
        p
    });

    candidates
}

pub fn generate_split_mode_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
) -> Vec<LlamaCppProfile> {
    if hardware.gpus.len() < 2 {
        return vec![];
    }

    let gpu_count = hardware.gpus.len();
    let base_split = normalized_vram_split(&hardware.gpus);

    let mut candidates = vec![];

    let ratio_variations: Vec<Vec<f32>> = if gpu_count == 2 {
        vec![
            base_split.clone(),
            vec![0.5, 0.5],
            vec![0.6, 0.4],
            vec![0.4, 0.6],
            vec![0.7, 0.3],
            vec![0.3, 0.7],
        ]
    } else {
        let mut variations = vec![base_split.clone()];
        for i in 0..gpu_count {
            let mut ratio = vec![0.0; gpu_count];
            ratio[i] = 1.0;
            variations.push(ratio);
        }
        variations
    };

    for split in ratio_variations {
        for mode in [SplitMode::Layer, SplitMode::Row] {
            let mut p = base.clone();
            p.split_mode = mode;
            p.tensor_split = Some(split.clone());
            candidates.push(p);
        }
    }

    candidates.push({
        let mut p = base.clone();
        p.split_mode = SplitMode::None;
        p.tensor_split = None;
        p
    });

    candidates
}

fn generate_context_batch_candidates(
    base: &LlamaCppProfile,
    facts: &GgufFacts,
    _hardware: &HardwareProfile,
    opts: &OptimizeOptions,
) -> Vec<LlamaCppProfile> {
    let model_ctx = facts.context_length.unwrap_or(4096);
    let ctx_size = opts
        .benchmark_ctx_size
        .unwrap_or(model_ctx)
        .min(model_ctx.max(1));

    let batch_sizes = multiplicative_ladder(base.batch_size, 32, 2048);
    let ubatch_sizes = multiplicative_ladder(base.ubatch_size.min(base.batch_size), 16, 512);

    let mut candidates = vec![];

    for ctx in multiplicative_ladder(ctx_size, 512, model_ctx.max(4096)) {
        for batch in batch_sizes.iter().take(6) {
            for ubatch in ubatch_sizes.iter().take(4) {
                if *ubatch > *batch {
                    continue;
                }
                let mut p = base.clone();
                p.ctx_size = ctx;
                p.batch_size = *batch;
                p.ubatch_size = *ubatch;
                candidates.push(p);
            }
        }
    }

    candidates
}

fn generate_thread_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
) -> Vec<LlamaCppProfile> {
    let phys = hardware.cpu_physical_cores.max(1);
    let logi = hardware.cpu_logical_cores.max(phys);

    let mut thread_vals = vec![
        base.threads,
        phys,
        logi,
        (phys / 2).max(1),
        (phys.saturating_sub(1)).max(1),
    ];
    thread_vals.sort();
    thread_vals.dedup();

    let mut thread_batch_vals = vec![
        base.threads_batch,
        base.threads,
        phys,
        logi,
        phys.saturating_mul(2).min(logi),
    ];
    thread_batch_vals.sort();
    thread_batch_vals.dedup();

    let mut candidates = vec![];

    for threads in thread_vals.iter().copied() {
        for threads_batch in thread_batch_vals.iter().copied() {
            if threads_batch > 0 {
                let mut p = base.clone();
                p.threads = threads;
                p.threads_batch = threads_batch;
                candidates.push(p);
            }
        }
    }

    candidates
}

pub fn generate_kv_cache_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
) -> Vec<LlamaCppProfile> {
    if hardware.gpus.is_empty() {
        return vec![];
    }

    let types = vec![
        (KvCacheType::F16, KvCacheType::F16),
        (KvCacheType::F16, KvCacheType::Q8_0),
        (KvCacheType::Q8_0, KvCacheType::Q8_0),
        (KvCacheType::BF16, KvCacheType::BF16),
    ];

    types
        .into_iter()
        .map(|(k, v)| {
            let mut p = base.clone();
            p.cache_type_k = k;
            p.cache_type_v = v;
            p
        })
        .collect()
}

fn generate_flash_attn_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
) -> Vec<LlamaCppProfile> {
    if hardware.gpus.is_empty() {
        return vec![];
    }

    vec![
        {
            let mut p = base.clone();
            p.flash_attn = FlashAttn::On;
            p
        },
        {
            let mut p = base.clone();
            p.flash_attn = FlashAttn::Auto;
            p
        },
        {
            let mut p = base.clone();
            p.flash_attn = FlashAttn::Off;
            p
        },
    ]
}

fn generate_memory_fit_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
) -> Vec<LlamaCppProfile> {
    if hardware.gpus.is_empty() {
        return vec![];
    }

    let targets = hardware
        .gpus
        .iter()
        .map(|g| ((g.vram_bytes / 1024 / 1024) as u32 / 16).clamp(256, 2048))
        .collect::<Vec<_>>();

    let mut candidates = vec![];

    for fit in [true, false] {
        for fit_target in [targets.clone(), vec![512], vec![1024], vec![2048]] {
            let mut p = base.clone();
            p.fit = fit;
            p.fit_target_mib = fit_target;
            candidates.push(p);
        }
    }

    for (mmap, mlock) in [(true, true), (true, false), (false, true), (false, false)] {
        let mut p = base.clone();
        p.mmap = mmap;
        p.mlock = mlock;
        candidates.push(p);
    }

    candidates.push({
        let mut p = base.clone();
        p.no_host = true;
        p
    });

    candidates
}

pub fn generate_cpu_affinity_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
) -> Vec<LlamaCppProfile> {
    if !hardware.gpus.is_empty() {
        return vec![];
    }

    let cores = hardware.cpu_physical_cores as usize;
    if cores < 4 {
        return vec![];
    }

    let half = cores / 2;
    let quarter = cores / 4;
    let three_quarter = cores * 3 / 4;

    let mut candidates = vec![];

    let masks = vec![
        None,
        Some(format!("0-{}", cores - 1)),
        Some(format!("0-{}", half - 1)),
        Some(format!("{}-{}", half, cores - 1)),
        Some(format!("0-{}", quarter - 1)),
        Some(format!("{}-{}", quarter, three_quarter - 1)),
    ];

    for mask in masks {
        for strict in [false, true] {
            let mut p = base.clone();
            p.cpu_mask = mask.clone();
            p.cpu_strict = strict;
            candidates.push(p);
        }
    }

    if let Some(numa_nodes) = hardware.numa_nodes {
        if numa_nodes > 1 {
            for numa in [Some(NumaMode::Distribute), Some(NumaMode::Isolate)] {
                let mut p = base.clone();
                p.numa = numa;
                p.cpu_mask = None;
                candidates.push(p);
            }
        }
    }

    for poll in [None, Some(0), Some(32), Some(64)] {
        let mut p = base.clone();
        p.poll = poll;
        candidates.push(p);
    }

    for poll_batch in [None, Some(0), Some(32)] {
        let mut p = base.clone();
        p.poll_batch = poll_batch;
        candidates.push(p);
    }

    candidates
}

fn generate_spec_candidates(
    base: &LlamaCppProfile,
    hardware: &HardwareProfile,
    opts: &OptimizeOptions,
) -> Vec<LlamaCppProfile> {
    if hardware.gpus.is_empty() || opts.generation_tokens <= 32 {
        return vec![];
    }

    let mut candidates = vec![];

    for n in [2, 3, 4, 5, 6] {
        let mut p = base.clone();
        p.spec_type = Some(SpecType::NGram);
        p.spec_extra = Some(format!("n={}", n));
        candidates.push(p);
    }

    if base.draft_model_path.is_some() {
        let mut p_with_draft = base.clone();
        p_with_draft.draft_gpu_layers = GpuLayerSpec::Exact(0);
        p_with_draft.draft_threads = 1;
        candidates.push(p_with_draft);

        if let Some(layers) = base.facts.block_count {
            let mut p_draft_some = base.clone();
            p_draft_some.draft_gpu_layers = GpuLayerSpec::Exact((layers / 4).max(1));
            p_draft_some.draft_threads = 2;
            candidates.push(p_draft_some);
        }
    }

    candidates
}

fn multiplicative_ladder(current: u32, min_val: u32, max_val: u32) -> Vec<u32> {
    let seed_neighbors = vec![
        current.saturating_div(2).max(min_val),
        current.saturating_sub(1).max(min_val),
        current,
        current.saturating_add(1).min(max_val),
        current.saturating_mul(2).min(max_val),
    ];

    let default_region = vec![32, 64, 128, 256, 512, 1024, 2048]
        .into_iter()
        .filter(|v| *v >= min_val && *v <= max_val)
        .collect::<Vec<_>>();

    let mut all: Vec<u32> = seed_neighbors.into_iter().chain(default_region).collect();
    all.sort();
    all.dedup();
    all.into_iter()
        .filter(|v| *v >= min_val && *v <= max_val)
        .collect()
}

fn normalized_vram_split(gpus: &[crate::tuning::GpuProfile]) -> Vec<f32> {
    let total: f64 = gpus.iter().map(|g| g.vram_bytes as f64).sum();
    if total <= 0.0 {
        return vec![1.0 / gpus.len().max(1) as f32; gpus.len()];
    }
    gpus.iter()
        .map(|g| (g.vram_bytes as f64 / total) as f32)
        .collect()
}

fn dedupe_profiles(mut profiles: Vec<LlamaCppProfile>) -> Vec<LlamaCppProfile> {
    profiles.sort_by_key(|a| a.fingerprint());
    profiles.dedup_by(|a, b| a.fingerprint() == b.fingerprint());
    profiles
}
