use llmr::tuning::{
    BenchmarkResult, BenchmarkRun, FlashAttn, GgufFacts, GgufFactsExtractor, GpuLayerSpec,
    HardwareProfile, KvCacheType, LlamaCppProfile, OptimizeError, OptimizeOptions, SearchStrategy,
    SplitMode,
};
use std::io::Cursor;
use std::path::PathBuf;

fn create_test_hardware() -> HardwareProfile {
    HardwareProfile {
        system_ram_bytes: 32 * 1024 * 1024 * 1024,
        cpu_physical_cores: 8,
        cpu_logical_cores: 16,
        gpus: vec![llmr::tuning::GpuProfile {
            index: 0,
            name: "NVIDIA RTX 3080".to_string(),
            vram_bytes: 10 * 1024 * 1024 * 1024,
            supports_flash_attn: true,
            supports_bf16: true,
        }],
        numa_nodes: None,
    }
}

fn create_test_gguf_facts() -> GgufFacts {
    GgufFacts {
        path: PathBuf::from("test.gguf"),
        architecture: "llama".to_string(),
        model_name: Some("Test Model".to_string()),
        size_label: Some("4.0GB".to_string()),
        quantization_version: Some(2),
        file_type: Some(2),
        alignment: Some(4096),
        context_length: Some(4096),
        embedding_length: Some(4096),
        block_count: Some(32),
        feed_forward_length: Some(11008),
        attention_head_count: Some(32),
        attention_head_count_kv: Some(32),
        rope_dimension_count: Some(128),
        rope_scaling_type: Some("linear".to_string()),
        rope_scaling_factor: Some(2.0),
        rope_scaling_original_context_length: Some(4096),
        chat_template: Some("{% for message in messages %}".to_string()),
        tensor_count: 100,
        weight_bytes: 4_000_000_000,
    }
}

fn create_test_profile() -> LlamaCppProfile {
    LlamaCppProfile {
        model_path: PathBuf::from("test.gguf"),
        ctx_size: 4096,
        batch_size: 512,
        ubatch_size: 128,
        threads: 8,
        threads_batch: 16,
        parallel: 1,
        n_gpu_layers: GpuLayerSpec::Auto,
        split_mode: SplitMode::None,
        tensor_split: None,
        main_gpu: None,
        flash_attn: FlashAttn::Auto,
        cache_type_k: KvCacheType::F16,
        cache_type_v: KvCacheType::F16,
        kv_offload: true,
        repack: true,
        mmap: true,
        mlock: false,
        numa: None,
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
        fit_target_mib: vec![1024],
        estimated_result: None,
        facts: create_test_gguf_facts(),
        notes: vec![],
    }
}

mod cli_mapping {
    use super::*;

    #[test]
    fn test_fit_target_mib_mapping() {
        let mut profile = create_test_profile();
        profile.fit = true;
        profile.fit_target_mib = vec![512];

        let args = profile.to_cli_args();
        let fit_idx = args.iter().position(|a| a == "-fitt").unwrap();
        assert_eq!(args[fit_idx + 1], "512");
    }

    #[test]
    fn test_ctx_size_mapping() {
        let mut profile = create_test_profile();
        profile.ctx_size = 8192;

        let args = profile.to_cli_args();
        let ctx_idx = args.iter().position(|a| a == "-c").unwrap();
        assert_eq!(args[ctx_idx + 1], "8192");
    }

    #[test]
    fn test_batch_size_mapping() {
        let mut profile = create_test_profile();
        profile.batch_size = 1024;

        let args = profile.to_cli_args();
        let batch_idx = args.iter().position(|a| a == "-b").unwrap();
        assert_eq!(args[batch_idx + 1], "1024");
    }

    #[test]
    fn test_cache_type_k_mapping() {
        let mut profile = create_test_profile();
        profile.cache_type_k = KvCacheType::Q8_0;

        let args = profile.to_cli_args();
        let ctk_idx = args.iter().position(|a| a == "-ctk").unwrap();
        assert_eq!(args[ctk_idx + 1], "q8_0");
    }

    #[test]
    fn test_cache_type_v_mapping() {
        let mut profile = create_test_profile();
        profile.cache_type_v = KvCacheType::Q4_0;

        let args = profile.to_cli_args();
        let ctv_idx = args.iter().position(|a| a == "-ctv").unwrap();
        assert_eq!(args[ctv_idx + 1], "q4_0");
    }

    #[test]
    fn test_gpu_layers_auto() {
        let mut profile = create_test_profile();
        profile.n_gpu_layers = GpuLayerSpec::Auto;

        let args = profile.to_cli_args();
        let ngl_idx = args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(args[ngl_idx + 1], "auto");
    }

    #[test]
    fn test_gpu_layers_all() {
        let mut profile = create_test_profile();
        profile.n_gpu_layers = GpuLayerSpec::All;

        let args = profile.to_cli_args();
        let ngl_idx = args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(args[ngl_idx + 1], "all");
    }

    #[test]
    fn test_gpu_layers_exact() {
        let mut profile = create_test_profile();
        profile.n_gpu_layers = GpuLayerSpec::Exact(16);

        let args = profile.to_cli_args();
        let ngl_idx = args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(args[ngl_idx + 1], "16");
    }

    #[test]
    fn test_split_mode_layer() {
        let mut profile = create_test_profile();
        profile.split_mode = SplitMode::Layer;

        let args = profile.to_cli_args();
        let sm_idx = args.iter().position(|a| a == "-sm").unwrap();
        assert_eq!(args[sm_idx + 1], "layer");
    }

    #[test]
    fn test_split_mode_row() {
        let mut profile = create_test_profile();
        profile.split_mode = SplitMode::Row;

        let args = profile.to_cli_args();
        let sm_idx = args.iter().position(|a| a == "-sm").unwrap();
        assert_eq!(args[sm_idx + 1], "row");
    }

    #[test]
    fn test_flash_attn_on() {
        let mut profile = create_test_profile();
        profile.flash_attn = FlashAttn::On;

        let args = profile.to_cli_args();
        let fa_idx = args.iter().position(|a| a == "-fa").unwrap();
        assert_eq!(args[fa_idx + 1], "on");
    }

    #[test]
    fn test_flash_attn_off() {
        let mut profile = create_test_profile();
        profile.flash_attn = FlashAttn::Off;

        let args = profile.to_cli_args();
        let fa_idx = args.iter().position(|a| a == "-fa").unwrap();
        assert_eq!(args[fa_idx + 1], "off");
    }
}

mod candidate_generation {
    use super::*;
    use llmr::tuning::candidates::{self, Stage};

    #[test]
    fn test_generate_all_candidates() {
        let profile = create_test_profile();
        let facts = create_test_gguf_facts();
        let hardware = create_test_hardware();
        let opts = OptimizeOptions::default();

        let all = candidates::generate_all_candidates(&profile, &facts, &hardware, &opts);
        assert!(!all.is_empty());
    }

    #[test]
    fn test_gpu_layer_candidates_auto_all_exact() {
        let profile = create_test_profile();
        let _facts = create_test_gguf_facts();
        let hardware = create_test_hardware();

        let candidates = candidates::generate_gpu_offload_candidates(&profile, &hardware);

        let has_auto = candidates
            .iter()
            .any(|p| matches!(p.n_gpu_layers, GpuLayerSpec::Auto));
        let has_all = candidates
            .iter()
            .any(|p| matches!(p.n_gpu_layers, GpuLayerSpec::All));
        let has_exact = candidates
            .iter()
            .any(|p| matches!(p.n_gpu_layers, GpuLayerSpec::Exact(_)));

        assert!(has_auto, "Should have Auto GPU layers");
        assert!(has_all, "Should have All GPU layers");
        assert!(has_exact, "Should have Exact GPU layers");
    }

    #[test]
    fn test_kv_cache_separate_kv() {
        let profile = create_test_profile();
        let hardware = create_test_hardware();

        let candidates = candidates::generate_kv_cache_candidates(&profile, &hardware);

        let has_different_kv = candidates.iter().any(|p| p.cache_type_k != p.cache_type_v);

        assert!(
            has_different_kv,
            "Should have candidates with different K and V cache types"
        );
    }

    #[test]
    fn test_split_mode_multi_gpu() {
        let mut hardware = create_test_hardware();
        hardware.gpus.push(llmr::tuning::GpuProfile {
            index: 1,
            name: "NVIDIA RTX 3090".to_string(),
            vram_bytes: 24 * 1024 * 1024 * 1024,
            supports_flash_attn: true,
            supports_bf16: true,
        });

        let profile = create_test_profile();
        let candidates = candidates::generate_split_mode_candidates(&profile, &hardware);

        let has_layer = candidates.iter().any(|p| p.split_mode == SplitMode::Layer);
        let has_row = candidates.iter().any(|p| p.split_mode == SplitMode::Row);

        assert!(has_layer, "Should have Layer split mode");
        assert!(has_row, "Should have Row split mode");
    }

    #[test]
    fn test_cpu_affinity_variants() {
        let mut hardware = create_test_hardware();
        hardware.gpus.clear();
        hardware.cpu_physical_cores = 8;

        let profile = create_test_profile();
        let candidates = candidates::generate_cpu_affinity_candidates(&profile, &hardware);

        let has_cpu_mask = candidates.iter().any(|p| p.cpu_mask.is_some());
        let has_numa = candidates.iter().any(|p| p.numa.is_some());

        assert!(
            has_cpu_mask || has_numa,
            "Should have CPU affinity candidates"
        );
    }

    #[test]
    fn test_stage_all_stages() {
        let stages = Stage::all();
        assert_eq!(stages.len(), 9);
    }

    #[test]
    fn test_stage_tier() {
        assert_eq!(Stage::GpuOffload.tier(), "high");
        assert_eq!(Stage::Threads.tier(), "medium");
        assert_eq!(Stage::MemoryFit.tier(), "low");
    }
}

mod benchmark_determinism {
    use crate::tuning::{create_test_profile, BenchmarkResult, BenchmarkRun};

    #[test]
    fn test_benchmark_result_determinism() {
        let runs = vec![
            BenchmarkRun {
                prompt_tps: 100.0,
                decode_tps: 50.0,
                latency_ms: 100.0,
                memory_mib: 2048,
                failed: false,
            },
            BenchmarkRun {
                prompt_tps: 101.0,
                decode_tps: 49.0,
                latency_ms: 99.0,
                memory_mib: 2047,
                failed: false,
            },
            BenchmarkRun {
                prompt_tps: 99.0,
                decode_tps: 51.0,
                latency_ms: 101.0,
                memory_mib: 2049,
                failed: false,
            },
        ];

        let result1 = BenchmarkResult::from_runs(runs.clone()).unwrap();
        let result2 = BenchmarkResult::from_runs(runs).unwrap();

        assert_eq!(result1.prompt_tps, result2.prompt_tps);
        assert_eq!(result1.decode_tps, result2.decode_tps);
    }

    #[test]
    fn test_fingerprint_determinism() {
        let profile = create_test_profile();
        let fp1 = profile.fingerprint();
        let fp2 = profile.fingerprint();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_candidate_key_determinism() {
        let profile = create_test_profile();
        let key1 = profile.candidate_key();
        let key2 = profile.candidate_key();
        assert_eq!(key1, key2);
    }
}

mod regression_tests {
    use super::*;

    #[test]
    fn test_fit_target_mib_not_zero() {
        let mut profile = create_test_profile();
        profile.fit = true;
        profile.fit_target_mib = vec![];

        let args = profile.to_cli_args();
        let fitt_idx = args.iter().position(|a| a == "-fitt").unwrap();
        assert!(args[fitt_idx + 1].parse::<u32>().unwrap() > 0);
    }

    #[test]
    fn test_kv_cache_can_differ() {
        let mut profile = create_test_profile();
        profile.cache_type_k = KvCacheType::F16;
        profile.cache_type_v = KvCacheType::Q8_0;

        let args = profile.to_cli_args();
        let ctk_idx = args.iter().position(|a| a == "-ctk").unwrap();
        let ctv_idx = args.iter().position(|a| a == "-ctv").unwrap();

        assert_eq!(args[ctk_idx + 1], "f16");
        assert_eq!(args[ctv_idx + 1], "q8_0");
    }

    #[test]
    fn test_rope_scaling_preserved() {
        let mut facts = create_test_gguf_facts();
        facts.rope_scaling_type = Some("linear".to_string());
        facts.rope_scaling_factor = Some(2.0);
        facts.rope_scaling_original_context_length = Some(4096);

        let mut profile = create_test_profile();
        profile.n_gpu_layers = GpuLayerSpec::Exact(0);
        profile.kv_offload = false;
        facts.weight_bytes = 1_000_000_000;

        let hardware = create_test_hardware();
        let _feasibility = llmr::tuning::FeasibilityResult::estimate(&profile, &facts, &hardware);
    }

    #[allow(clippy::approx_constant)]
    #[test]
    fn test_read_f32() {
        let data: Vec<u8> = f32::to_le_bytes(3.14).to_vec();
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_f32(&mut cursor).unwrap();
        assert!((result - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_all_kv_cache_types_serializable() {
        use serde_json;

        let types = [
            KvCacheType::F32,
            KvCacheType::F16,
            KvCacheType::BF16,
            KvCacheType::Q8_0,
            KvCacheType::Q4_0,
            KvCacheType::Q4_1,
            KvCacheType::IQ4_NL,
            KvCacheType::Q5_0,
            KvCacheType::Q5_1,
        ];

        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let restored: KvCacheType = serde_json::from_str(&json).unwrap();
            assert_eq!(t, restored);
        }
    }
}

mod serialization {
    use super::*;

    #[test]
    fn test_profile_serialization_excludes_runtime_fields() {
        use serde_json;

        let profile = create_test_profile();
        let json = serde_json::to_string(&profile).unwrap();

        assert!(!json.contains("estimated_result"));
        assert!(!json.contains("facts"));
        assert!(!json.contains("notes"));
    }

    #[test]
    fn test_profile_round_trip() {
        use serde_json;

        let profile = create_test_profile();
        let json = serde_json::to_string(&profile).unwrap();
        let restored: LlamaCppProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(profile.model_path, restored.model_path);
        assert_eq!(profile.ctx_size, restored.ctx_size);
        assert_eq!(profile.batch_size, restored.batch_size);
        assert_eq!(profile.n_gpu_layers, restored.n_gpu_layers);
    }

    #[test]
    fn test_gpu_layer_spec_round_trip() {
        use serde_json;

        let specs = [
            GpuLayerSpec::Auto,
            GpuLayerSpec::All,
            GpuLayerSpec::Exact(42),
        ];

        for spec in specs {
            let json = serde_json::to_string(&spec).unwrap();
            let restored: GpuLayerSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(spec, restored);
        }
    }
}

mod candidate_key_tests {
    use super::*;

    #[test]
    fn test_candidate_key_equality() {
        let profile1 = create_test_profile();
        let profile2 = create_test_profile();

        let key1 = profile1.candidate_key();
        let key2 = profile2.candidate_key();

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_candidate_key_hash() {
        use std::collections::HashSet;

        let profile = create_test_profile();
        let key = profile.candidate_key();

        let mut set = HashSet::new();
        set.insert(key.clone());
        set.insert(key.clone());

        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_candidate_key_fingerprint() {
        let profile = create_test_profile();
        let key = profile.candidate_key();

        let fp1 = key.fingerprint();
        let fp2 = key.fingerprint();

        assert_eq!(fp1, fp2);
        assert!(fp1.contains("ctx="));
    }
}

mod optimization_options {
    use super::*;

    #[test]
    fn backend_support_status_marks_only_llama_cpp_supported() {
        assert!(llmr::tuning::Backend::LlamaCpp.supports_serving());
        assert_eq!(
            llmr::tuning::Backend::LlamaCpp.support_status(),
            "supported"
        );
        assert!(!llmr::tuning::Backend::Vllm.supports_serving());
        assert_eq!(llmr::tuning::Backend::Vllm.support_status(), "planned");
        assert!(!llmr::tuning::Backend::Sglang.supports_serving());
        assert_eq!(llmr::tuning::Backend::Sglang.support_status(), "planned");
    }

    #[test]
    fn test_optimize_options_default() {
        let opts = OptimizeOptions::default();

        assert!(opts.benchmark_ctx_size.is_some());
        assert_eq!(opts.prompt_tokens, 512);
        assert_eq!(opts.generation_tokens, 128);
        assert_eq!(opts.search_strategy, SearchStrategy::Racing);
    }

    #[test]
    fn test_search_strategy_variants() {
        let strategies = [
            SearchStrategy::Greedy,
            SearchStrategy::Racing,
            SearchStrategy::Exhaustive,
            SearchStrategy::Fast,
        ];

        for s in strategies {
            let _ = format!("{:?}", s);
        }
    }
}

mod feasibility_tests {
    use super::*;

    #[test]
    fn test_feasibility_gpu_memory() {
        let mut profile = create_test_profile();
        profile.n_gpu_layers = GpuLayerSpec::Exact(0);

        let mut facts = create_test_gguf_facts();
        facts.weight_bytes = 1_000_000_000;

        let hardware = create_test_hardware();

        let result = llmr::tuning::FeasibilityResult::estimate(&profile, &facts, &hardware);

        assert!(result.vram_available > 0);
    }

    #[test]
    fn test_feasibility_insufficient_vram() {
        let mut profile = create_test_profile();
        profile.n_gpu_layers = GpuLayerSpec::Exact(999);

        let facts = create_test_gguf_facts();
        let mut hardware = create_test_hardware();
        hardware.gpus[0].vram_bytes = 100_000; // Very small

        let result = llmr::tuning::FeasibilityResult::estimate(&profile, &facts, &hardware);

        assert!(!result.viable);
        assert!(result.reason.is_some());
    }

    #[test]
    fn test_feasibility_cpu_only() {
        let mut hardware = create_test_hardware();
        hardware.gpus.clear();

        let mut profile = create_test_profile();
        profile.n_gpu_layers = GpuLayerSpec::Exact(0);

        let facts = create_test_gguf_facts();

        let result = llmr::tuning::FeasibilityResult::estimate(&profile, &facts, &hardware);

        assert!(result.viable || result.reason.is_some());
    }
}

mod memory_estimate_tests {
    use super::*;

    #[test]
    fn test_kv_cache_estimate_no_overflow() {
        let profile = create_test_profile();
        let facts = create_test_gguf_facts();

        let kv_bytes = llmr::tuning::estimate_kv_cache_bytes(&profile, &facts);

        assert!(kv_bytes > 0);
        assert!(kv_bytes < u64::MAX);
    }

    #[test]
    fn test_batch_buffer_estimate_no_overflow() {
        let profile = create_test_profile();
        let facts = create_test_gguf_facts();

        let batch_bytes = llmr::tuning::estimate_batch_buffer_bytes(&profile, &facts);

        assert!(batch_bytes > 0);
    }

    #[test]
    fn test_max_gpu_layers_calculation() {
        let mut facts = create_test_gguf_facts();
        facts.weight_bytes = 4_000_000_000;

        let hardware = create_test_hardware();

        let profile = create_test_profile();

        let max_layers = llmr::tuning::estimate_max_gpu_layers(&facts, &hardware, &profile);

        assert!(max_layers <= 100);
    }
}

mod math_utils {

    #[test]
    fn test_trimmed_mean_empty() {
        let values: Vec<f32> = vec![];
        let result = llmr::tuning::trimmed_mean(&values, 0.1);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_trimmed_mean_single() {
        let values = vec![42.0];
        let result = llmr::tuning::trimmed_mean(&values, 0.1);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_trimmed_mean_two() {
        let values = vec![1.0, 3.0];
        let result = llmr::tuning::trimmed_mean(&values, 0.1);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_trimmed_mean_trims_outliers() {
        let values = vec![1.0, 2.0, 3.0, 100.0];
        let result = llmr::tuning::trimmed_mean(&values, 0.25);
        assert_eq!(result, 2.5);
    }

    #[test]
    fn test_trimmed_mean_handles_nan() {
        let values = vec![1.0, f32::NAN, 3.0];
        let result = llmr::tuning::trimmed_mean(&values, 0.1);
        assert!(result.is_nan());
    }

    #[test]
    fn test_variance_empty() {
        let values: Vec<f32> = vec![];
        let result = llmr::tuning::variance(&values);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_variance_single() {
        let values = vec![5.0];
        let result = llmr::tuning::variance(&values);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_variance_correct() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = llmr::tuning::variance(&values);
        assert!((result - 4.571_429).abs() < 0.01);
    }
}

mod benchmark_result_edge_cases {
    use super::*;

    #[test]
    fn test_from_runs_all_failed() {
        let runs = vec![
            BenchmarkRun {
                prompt_tps: 0.0,
                decode_tps: 0.0,
                latency_ms: 0.0,
                memory_mib: 0,
                failed: true,
            },
            BenchmarkRun {
                prompt_tps: 0.0,
                decode_tps: 0.0,
                latency_ms: 0.0,
                memory_mib: 0,
                failed: true,
            },
        ];
        let result = BenchmarkResult::from_runs(runs);
        assert!(result.is_none());
    }

    #[test]
    fn test_from_runs_empty() {
        let runs: Vec<BenchmarkRun> = vec![];
        let result = BenchmarkResult::from_runs(runs);
        assert!(result.is_none());
    }

    #[test]
    fn test_from_runs_mixed_success_failure() {
        let runs = vec![
            BenchmarkRun {
                prompt_tps: 100.0,
                decode_tps: 50.0,
                latency_ms: 100.0,
                memory_mib: 2048,
                failed: false,
            },
            BenchmarkRun {
                prompt_tps: 0.0,
                decode_tps: 0.0,
                latency_ms: 0.0,
                memory_mib: 0,
                failed: true,
            },
            BenchmarkRun {
                prompt_tps: 110.0,
                decode_tps: 55.0,
                latency_ms: 95.0,
                memory_mib: 2050,
                failed: false,
            },
        ];
        let result = BenchmarkResult::from_runs(runs).unwrap();
        assert_eq!(result.failure_count, 1);
        assert_eq!(result.run_count, 3);
        assert!((result.stability - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_variance_penalty_high() {
        let result = BenchmarkResult {
            prompt_tps: 100.0,
            decode_tps: 50.0,
            latency_ms: 100.0,
            memory_mib: 2048,
            stability: 1.0,
            run_count: 5,
            failure_count: 0,
            prompt_tps_variance: 900.0,
            decode_tps_variance: 400.0,
        };
        let penalty = result.variance_penalty();
        assert!(penalty > 0.0);
    }

    #[test]
    fn test_variance_penalty_zero_on_low_variance() {
        let result = BenchmarkResult {
            prompt_tps: 100.0,
            decode_tps: 50.0,
            latency_ms: 100.0,
            memory_mib: 2048,
            stability: 1.0,
            run_count: 5,
            failure_count: 0,
            prompt_tps_variance: 0.0,
            decode_tps_variance: 0.0,
        };
        let penalty = result.variance_penalty();
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn test_combined_score_zero_tps() {
        let result = BenchmarkResult {
            prompt_tps: 0.0,
            decode_tps: 0.0,
            latency_ms: 100.0,
            memory_mib: 2048,
            stability: 1.0,
            run_count: 5,
            failure_count: 0,
            prompt_tps_variance: 0.0,
            decode_tps_variance: 0.0,
        };
        let score = result.combined_score();
        assert!(score > 0.0);
    }
}

mod search_error_handling {
    use super::*;
    use std::path::Path;

    struct StaticExtractor;

    impl GgufFactsExtractor for StaticExtractor {
        fn extract(&self, _path: &Path) -> Result<GgufFacts, OptimizeError> {
            Ok(create_test_gguf_facts())
        }
    }

    #[test]
    fn optimize_propagates_benchmark_errors() {
        let result = llmr::tuning::optimize_llama_cpp_profile(
            "test.gguf",
            create_test_hardware(),
            &StaticExtractor,
            |_profile| Err(OptimizeError::Benchmark("docker unavailable".to_string())),
            OptimizeOptions {
                max_rounds: 1,
                search_strategy: SearchStrategy::Fast,
                ..OptimizeOptions::default()
            },
        );

        assert!(
            matches!(result, Err(OptimizeError::Benchmark(message)) if message == "docker unavailable")
        );
    }
}

mod cache_key_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_cache_key_equality() {
        let key1 = llmr::tuning::CacheKey {
            profile_key: "test".to_string(),
            model_fingerprint: "model".to_string(),
            hardware_fingerprint: "hw".to_string(),
        };
        let key2 = llmr::tuning::CacheKey {
            profile_key: "test".to_string(),
            model_fingerprint: "model".to_string(),
            hardware_fingerprint: "hw".to_string(),
        };
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_hash() {
        let key1 = llmr::tuning::CacheKey {
            profile_key: "test".to_string(),
            model_fingerprint: "model".to_string(),
            hardware_fingerprint: "hw".to_string(),
        };
        let key2 = llmr::tuning::CacheKey {
            profile_key: "test".to_string(),
            model_fingerprint: "model".to_string(),
            hardware_fingerprint: "hw".to_string(),
        };
        let mut set = HashSet::new();
        set.insert(key1);
        set.insert(key2);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_search_cache_insert_get() {
        let mut cache = llmr::tuning::SearchCache::new();
        let key = llmr::tuning::CacheKey {
            profile_key: "test".to_string(),
            model_fingerprint: "model".to_string(),
            hardware_fingerprint: "hw".to_string(),
        };
        let result = BenchmarkResult {
            prompt_tps: 100.0,
            decode_tps: 50.0,
            latency_ms: 100.0,
            memory_mib: 2048,
            stability: 1.0,
            run_count: 1,
            failure_count: 0,
            prompt_tps_variance: 0.0,
            decode_tps_variance: 0.0,
        };
        cache.insert(key.clone(), result);
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn test_search_cache_len() {
        let mut cache = llmr::tuning::SearchCache::new();
        for i in 0..3 {
            let key = llmr::tuning::CacheKey {
                profile_key: format!("test{}", i),
                model_fingerprint: "model".to_string(),
                hardware_fingerprint: "hw".to_string(),
            };
            let result = BenchmarkResult {
                prompt_tps: 100.0,
                decode_tps: 50.0,
                latency_ms: 100.0,
                memory_mib: 2048,
                stability: 1.0,
                run_count: 1,
                failure_count: 0,
                prompt_tps_variance: 0.0,
                decode_tps_variance: 0.0,
            };
            cache.insert(key, result);
        }
        assert_eq!(cache.len(), 3);
    }
}

mod context_plausibility {
    use super::*;

    #[test]
    fn test_context_plausible_valid() {
        let mut facts = create_test_gguf_facts();
        facts.context_length = Some(4096);
        facts.rope_scaling_factor = Some(2.0);
        facts.rope_scaling_original_context_length = Some(4096);

        let result = llmr::tuning::is_context_plausible(&facts, 8192);
        assert!(result);
    }

    #[test]
    fn test_context_plausible_too_large() {
        let mut facts = create_test_gguf_facts();
        facts.context_length = Some(4096);
        facts.rope_scaling_factor = Some(2.0);
        facts.rope_scaling_original_context_length = Some(4096);

        let result = llmr::tuning::is_context_plausible(&facts, 131072);
        assert!(!result);
    }

    #[test]
    fn test_context_plausible_no_rope() {
        let mut facts = create_test_gguf_facts();
        facts.context_length = Some(8192);
        facts.rope_scaling_factor = None;

        let result = llmr::tuning::is_context_plausible(&facts, 4096);
        assert!(result);
    }

    #[test]
    fn test_context_plausible_no_info() {
        let facts = create_test_gguf_facts();

        let result = llmr::tuning::is_context_plausible(&facts, 4096);
        assert!(result);
    }
}

mod gguf_parsing_tests {
    use std::io::Cursor;

    #[test]
    fn test_read_u64() {
        let data: Vec<u8> = u64::to_le_bytes(12345).to_vec();
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_u64(&mut cursor).unwrap();
        assert_eq!(result, 12345);
    }

    #[test]
    fn test_read_u32() {
        let data: Vec<u8> = u32::to_le_bytes(42).to_vec();
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_u32(&mut cursor).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_read_u16() {
        let data: Vec<u8> = u16::to_le_bytes(100).to_vec();
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_u16(&mut cursor).unwrap();
        assert_eq!(result, 100);
    }

    #[test]
    fn test_read_u8() {
        let data = vec![42u8];
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_u8(&mut cursor).unwrap();
        assert_eq!(result, 42);
    }

    #[allow(clippy::approx_constant)]
    #[test]
    fn test_read_f32() {
        let data: Vec<u8> = f32::to_le_bytes(3.14).to_vec();
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_f32(&mut cursor).unwrap();
        assert!((result - 3.14).abs() < 0.001);
    }

    #[allow(clippy::approx_constant)]
    #[test]
    fn test_read_f64() {
        let data: Vec<u8> = f64::to_le_bytes(2.71828).to_vec();
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_f64(&mut cursor).unwrap();
        assert!((result - 2.71828).abs() < 0.0001);
    }

    #[test]
    fn test_read_string() {
        let test_str = b"hello";
        let mut data = vec![];
        data.extend_from_slice(&u64::to_le_bytes(test_str.len() as u64));
        data.extend_from_slice(test_str);
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_string(&mut cursor).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_read_string_too_long() {
        let mut data = vec![];
        data.extend_from_slice(&u64::to_le_bytes(100_000_000));
        let mut cursor = Cursor::new(data);
        let result = llmr::tuning::gguf::read_string(&mut cursor);
        assert!(result.is_err());
    }
}

mod candidate_generation_edge_cases {
    use super::*;

    #[test]
    fn test_candidate_batches_no_change() {
        let batches = llmr::tuning::candidate_batches(512);
        assert!(batches.contains(&512));
    }

    #[test]
    fn test_candidate_ubatches_bounded_by_batch() {
        let ubatches = llmr::tuning::candidate_ubatches(256, 128);
        for ub in &ubatches {
            assert!(*ub <= 256);
        }
    }

    #[test]
    fn test_candidate_gpu_layers_range() {
        let layers =
            llmr::tuning::candidate_gpu_layers(Some(32), 1, llmr::tuning::GpuLayerSpec::Exact(0));
        assert!(!layers.is_empty());
    }

    #[test]
    fn test_candidate_kv_cache_types_includes_current() {
        let types =
            llmr::tuning::candidate_kv_cache_types(&create_test_hardware(), KvCacheType::Q4_0);
        assert!(types.contains(&KvCacheType::Q4_0));
    }

    #[test]
    fn test_candidate_poll_includes_zero() {
        let poll = llmr::tuning::candidate_poll(None);
        let has_zero = poll.contains(&0);
        assert!(has_zero, "candidate_poll should include 0 as valid option");
    }

    #[test]
    fn test_candidate_cont_batching_toggle() {
        let modes = llmr::tuning::candidate_cont_batching(true, &OptimizeOptions::default());
        assert!(modes.contains(&true));
    }
}
