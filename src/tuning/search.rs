use crate::tuning::{
    estimate_max_gpu_layers, is_context_plausible, BenchmarkResult, CacheKey, FeasibilityResult,
    GgufFacts, GpuLayerSpec, HardwareProfile, LlamaCppProfile, OptimizeOptions, SearchCache,
    SearchStrategy,
};

#[derive(Debug, Clone)]
pub struct SearchStats {
    pub candidates_tested: usize,
    pub cache_size: usize,
    pub best_score_history: Vec<f32>,
}

pub struct SearchEngine {
    cache: SearchCache,
    candidates_tested: usize,
    best_score_history: Vec<f32>,
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            cache: SearchCache::new(),
            candidates_tested: 0,
            best_score_history: Vec::new(),
        }
    }

    pub fn stats(&self) -> SearchStats {
        SearchStats {
            candidates_tested: self.candidates_tested,
            cache_size: self.cache.len(),
            best_score_history: self.best_score_history.clone(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search<F>(
        &mut self,
        strategy: SearchStrategy,
        base: &LlamaCppProfile,
        facts: &GgufFacts,
        hardware: &HardwareProfile,
        opts: &OptimizeOptions,
        benchmark: &mut F,
        model_fp: &str,
        hw_fp: &str,
    ) -> Result<LlamaCppProfile, crate::tuning::OptimizeError>
    where
        F: FnMut(&LlamaCppProfile) -> Result<BenchmarkResult, crate::tuning::OptimizeError>,
    {
        match strategy {
            SearchStrategy::Greedy => {
                self.greedy_search(base, facts, hardware, opts, benchmark, model_fp, hw_fp)
            }
            SearchStrategy::Racing => {
                self.racing_search(base, facts, hardware, opts, benchmark, model_fp, hw_fp)
            }
            SearchStrategy::Exhaustive => {
                self.exhaustive_search(base, facts, hardware, opts, benchmark, model_fp, hw_fp)
            }
            SearchStrategy::Fast => {
                self.fast_search(base, facts, hardware, opts, benchmark, model_fp, hw_fp)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn greedy_search<F>(
        &mut self,
        base: &LlamaCppProfile,
        facts: &GgufFacts,
        hardware: &HardwareProfile,
        opts: &OptimizeOptions,
        benchmark: &mut F,
        model_fp: &str,
        hw_fp: &str,
    ) -> Result<LlamaCppProfile, crate::tuning::OptimizeError>
    where
        F: FnMut(&LlamaCppProfile) -> Result<BenchmarkResult, crate::tuning::OptimizeError>,
    {
        let seed_result = self.benchmark_cached(base, benchmark, model_fp, hw_fp)?;
        let mut best = base.clone();
        best.estimated_result = Some(seed_result.clone());
        self.best_score_history.push(seed_result.combined_score());

        for _ in 0..opts.max_rounds {
            let mut improved = false;
            let variants = crate::tuning::coordinate_variants(&best, facts, hardware, opts);

            for variant in variants {
                if !self.is_feasible(&variant, facts, hardware) {
                    continue;
                }

                let result = self.benchmark_cached(&variant, benchmark, model_fp, hw_fp)?;
                if result.combined_score()
                    > best
                        .estimated_result
                        .as_ref()
                        .map(|r| r.combined_score())
                        .unwrap_or(0.0)
                {
                    best = variant;
                    best.estimated_result = Some(result);
                    improved = true;
                }
            }

            if let Some(score) = best.estimated_result.as_ref().map(|r| r.combined_score()) {
                self.best_score_history.push(score);
            }

            if !improved {
                break;
            }
        }

        Ok(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn racing_search<F>(
        &mut self,
        base: &LlamaCppProfile,
        facts: &GgufFacts,
        hardware: &HardwareProfile,
        opts: &OptimizeOptions,
        benchmark: &mut F,
        model_fp: &str,
        hw_fp: &str,
    ) -> Result<LlamaCppProfile, crate::tuning::OptimizeError>
    where
        F: FnMut(&LlamaCppProfile) -> Result<BenchmarkResult, crate::tuning::OptimizeError>,
    {
        let seed_result = self.benchmark_cached(base, benchmark, model_fp, hw_fp)?;
        let mut best = base.clone();
        best.estimated_result = Some(seed_result.clone());
        self.best_score_history.push(seed_result.combined_score());

        let initial = crate::tuning::initial_variants(&best, facts, hardware, opts);

        let feasible_initial: Vec<_> = initial
            .into_iter()
            .filter(|p| self.is_feasible(p, facts, hardware))
            .collect();

        let mut candidates: Vec<(LlamaCppProfile, BenchmarkResult)> = feasible_initial
            .into_iter()
            .map(|mut p| {
                let result = self.benchmark_cached(&p, benchmark, model_fp, hw_fp)?;
                p.estimated_result = Some(result.clone());
                Ok((p, result))
            })
            .collect::<Result<_, crate::tuning::OptimizeError>>()?;

        candidates.sort_by(|a, b| {
            b.1.combined_score()
                .partial_cmp(&a.1.combined_score())
                .unwrap()
        });

        let keep_count = ((candidates.len() as f32) * opts.racing_keep_fraction).max(1.0) as usize;
        let survivors: Vec<LlamaCppProfile> = candidates
            .clone()
            .into_iter()
            .take(keep_count)
            .map(|(p, _)| p)
            .collect();

        for stage in crate::tuning::candidates::Stage::all() {
            let stage_candidates = crate::tuning::candidates::generate_stage_candidates(
                &best, facts, hardware, opts, stage,
            );

            let feasible_stage: Vec<_> = stage_candidates
                .into_iter()
                .filter(|p| self.is_feasible(p, facts, hardware))
                .collect();

            let evaluated: Vec<(LlamaCppProfile, BenchmarkResult)> = feasible_stage
                .into_iter()
                .map(|mut p| {
                    let result = self.benchmark_cached(&p, benchmark, model_fp, hw_fp)?;
                    p.estimated_result = Some(result.clone());
                    Ok((p, result))
                })
                .collect::<Result<_, crate::tuning::OptimizeError>>()?;

            let mut all: Vec<LlamaCppProfile> = survivors
                .iter()
                .cloned()
                .chain(evaluated.iter().map(|(p, _)| p.clone()))
                .collect();
            all.sort_by(|a, b| {
                let a_score = a
                    .estimated_result
                    .as_ref()
                    .map(|r| r.combined_score())
                    .unwrap_or(0.0);
                let b_score = b
                    .estimated_result
                    .as_ref()
                    .map(|r| r.combined_score())
                    .unwrap_or(0.0);
                b_score.partial_cmp(&a_score).unwrap()
            });

            let stage_keep = ((all.len() as f32) * opts.racing_keep_fraction).max(1.0) as usize;
            candidates = all
                .into_iter()
                .take(stage_keep)
                .filter_map(|p| p.estimated_result.clone().map(|r| (p, r)))
                .collect();

            if let Some((ref best_candidate, _)) = candidates.first() {
                if let Some(score) = best_candidate
                    .estimated_result
                    .as_ref()
                    .map(|r| r.combined_score())
                {
                    self.best_score_history.push(score);
                }
            }
        }

        if let Some((best_candidate, _)) = candidates.into_iter().max_by(|a, b| {
            let a_score = a.1.combined_score();
            let b_score = b.1.combined_score();
            a_score.partial_cmp(&b_score).unwrap()
        }) {
            if best_candidate
                .estimated_result
                .as_ref()
                .map(|r| r.combined_score())
                .unwrap_or(0.0)
                > best
                    .estimated_result
                    .as_ref()
                    .map(|r| r.combined_score())
                    .unwrap_or(0.0)
            {
                best = best_candidate;
            }
        }

        Ok(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn exhaustive_search<F>(
        &mut self,
        base: &LlamaCppProfile,
        facts: &GgufFacts,
        hardware: &HardwareProfile,
        opts: &OptimizeOptions,
        benchmark: &mut F,
        model_fp: &str,
        hw_fp: &str,
    ) -> Result<LlamaCppProfile, crate::tuning::OptimizeError>
    where
        F: FnMut(&LlamaCppProfile) -> Result<BenchmarkResult, crate::tuning::OptimizeError>,
    {
        let seed_result = self.benchmark_cached(base, benchmark, model_fp, hw_fp)?;
        let mut best = base.clone();
        best.estimated_result = Some(seed_result.clone());
        self.best_score_history.push(seed_result.combined_score());

        let candidates =
            crate::tuning::candidates::generate_all_candidates(base, facts, hardware, opts);

        for mut candidate in candidates {
            if !self.is_feasible(&candidate, facts, hardware) {
                continue;
            }

            let result = self.benchmark_cached(&candidate, benchmark, model_fp, hw_fp)?;
            if result.combined_score()
                > best
                    .estimated_result
                    .as_ref()
                    .map(|r| r.combined_score())
                    .unwrap_or(0.0)
            {
                candidate.estimated_result = Some(result.clone());
                best = candidate;
            }

            if let Some(score) = best.estimated_result.as_ref().map(|r| r.combined_score()) {
                self.best_score_history.push(score);
            }
        }

        Ok(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn fast_search<F>(
        &mut self,
        base: &LlamaCppProfile,
        facts: &GgufFacts,
        hardware: &HardwareProfile,
        opts: &OptimizeOptions,
        benchmark: &mut F,
        model_fp: &str,
        hw_fp: &str,
    ) -> Result<LlamaCppProfile, crate::tuning::OptimizeError>
    where
        F: FnMut(&LlamaCppProfile) -> Result<BenchmarkResult, crate::tuning::OptimizeError>,
    {
        let fast_opts = OptimizeOptions {
            max_rounds: 1,
            benchmark_samples: 2,
            warmup_samples: 1,
            ..opts.clone()
        };
        self.racing_search(
            base, facts, hardware, &fast_opts, benchmark, model_fp, hw_fp,
        )
    }

    fn benchmark_cached<F>(
        &mut self,
        profile: &LlamaCppProfile,
        benchmark: &mut F,
        model_fp: &str,
        hw_fp: &str,
    ) -> Result<BenchmarkResult, crate::tuning::OptimizeError>
    where
        F: FnMut(&LlamaCppProfile) -> Result<BenchmarkResult, crate::tuning::OptimizeError>,
    {
        let key = CacheKey {
            profile_key: profile.fingerprint(),
            model_fingerprint: model_fp.to_string(),
            hardware_fingerprint: hw_fp.to_string(),
        };

        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }

        self.candidates_tested += 1;

        let result = benchmark(profile)?;

        self.cache.insert(key, result.clone());
        Ok(result)
    }

    fn is_feasible(
        &self,
        profile: &LlamaCppProfile,
        facts: &GgufFacts,
        hardware: &HardwareProfile,
    ) -> bool {
        let feasibility = FeasibilityResult::estimate(profile, facts, hardware);
        if !feasibility.viable {
            return false;
        }
        if !is_context_plausible(facts, profile.ctx_size) {
            return false;
        }
        let max_layers = estimate_max_gpu_layers(facts, hardware, profile);
        if let GpuLayerSpec::Exact(layers) = profile.n_gpu_layers {
            if layers > max_layers {
                return false;
            }
        }
        true
    }
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self::new()
    }
}
