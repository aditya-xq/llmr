use crate::bench::error::{ReportWriteError, Result};
use crate::bench::metrics::{BenchmarkMetrics, PerformanceSummary};
use crate::bench::types::{BenchmarkConfig, ServerMetadata};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    pub run_id: String,
    pub timestamp: String,
    pub config: ConfigSnapshot,
    pub server_metadata: Option<ServerMetadata>,
    pub performance: Option<PerformanceSummary>,
    pub quality: Option<QualityResults>,
}

#[derive(Debug, Serialize)]
pub struct ConfigSnapshot {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct QualityResults {
    pub overall_score: f64,
    pub task_scores: std::collections::HashMap<String, f64>,
}

impl BenchmarkReport {
    pub fn new(
        model_name: &str,
        config: &BenchmarkConfig,
        metadata: Option<ServerMetadata>,
        summary: Option<PerformanceSummary>,
        quality_metrics: Option<BenchmarkMetrics>,
    ) -> Self {
        let config_snapshot = ConfigSnapshot {
            model: model_name.to_string(),
            max_tokens: config.performance.max_tokens,
            temperature: config.performance.temperature,
            top_p: config.performance.top_p,
            seed: config.performance.seed,
            warmup_runs: config.performance.warmup_runs,
            measured_runs: config.performance.measured_runs,
            stream: config.performance.stream,
        };

        let quality = quality_metrics.map(|q| QualityResults {
            overall_score: q.tokens_per_second,
            task_scores: std::collections::HashMap::new(),
        });

        Self {
            run_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: config_snapshot,
            server_metadata: metadata,
            performance: summary,
            quality,
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ReportWriteError::Write(e.to_string()).into())
    }

    pub fn to_console(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Benchmark Report ===\n");
        output.push_str(&format!("Run ID: {}\n", self.run_id));
        output.push_str(&format!("Timestamp: {}\n", self.timestamp));
        output.push_str("\n=== Config ===\n");
        output.push_str(&format!("Model: {}\n", self.config.model));
        output.push_str(&format!("Max tokens: {}\n", self.config.max_tokens));
        output.push_str(&format!("Temperature: {:.2}\n", self.config.temperature));
        output.push_str(&format!("Top-p: {:.2}\n", self.config.top_p));
        output.push_str(&format!("Seed: {}\n", self.config.seed));
        output.push_str(&format!("Warmup runs: {}\n", self.config.warmup_runs));
        output.push_str(&format!("Measured runs: {}\n", self.config.measured_runs));
        output.push_str(&format!("Stream: {}\n", self.config.stream));

        if let Some(ref meta) = self.server_metadata {
            output.push_str("\n=== Server Metadata ===\n");
            if let Some(ref name) = meta.model_name {
                output.push_str(&format!("Model name: {}\n", name));
            }
            if let Some(size) = meta.ctx_size {
                output.push_str(&format!("Context size: {}\n", size));
            }
            if let Some(layers) = meta.gpu_layers {
                output.push_str(&format!("GPU layers: {}\n", layers));
            }
            if let Some(ref quant) = meta.quantization {
                output.push_str(&format!("Quantization: {}\n", quant));
            }
        }

        if let Some(ref perf) = self.performance {
            output.push_str("\n=== Performance ===\n");
            output.push_str(&format!(
                "  Runs: {}/{} successful\n",
                perf.successful_runs, perf.total_runs
            ));
            output.push_str(&format!("  Error rate: {:.2}%\n", perf.error_rate * 100.0));
            output.push_str("\n  Latency:\n");
            output.push_str(&format!("    Mean: {:.2} ms\n", perf.latency_avg));
            output.push_str(&format!("    Min:  {:.2} ms\n", perf.latency_min));
            output.push_str(&format!("    Max:  {:.2} ms\n", perf.latency_max));
            output.push_str(&format!("    p50:  {:.2} ms\n", perf.latency_p50));
            output.push_str(&format!("    p90:  {:.2} ms\n", perf.latency_p90));
            output.push_str(&format!("    p95:  {:.2} ms\n", perf.latency_p95));
            output.push_str(&format!("    p99:  {:.2} ms\n", perf.latency_p99));
            output.push_str(&format!("    StdDev: {:.2} ms\n", perf.latency_stddev));
            output.push_str(&format!("    CV: {:.2}%\n", perf.latency_cv));
            output.push_str("\n  First Token (TTFT):\n");
            output.push_str(&format!("    Mean: {:.2} ms\n", perf.ttft_avg));
            output.push_str(&format!("    Min:  {:.2} ms\n", perf.ttft_min));
            output.push_str(&format!("    Max:  {:.2} ms\n", perf.ttft_max));
            output.push_str("\n  Throughput:\n");
            output.push_str(&format!(
                "    Mean: {:.2} tokens/sec\n",
                perf.tokens_per_sec_avg
            ));
            output.push_str(&format!(
                "    Min:  {:.2} tokens/sec\n",
                perf.tokens_per_sec_min
            ));
            output.push_str(&format!(
                "    Max:  {:.2} tokens/sec\n",
                perf.tokens_per_sec_max
            ));
        }

        if let Some(ref quality) = self.quality {
            output.push_str("\n=== Quality ===\n");
            output.push_str(&format!("Overall score: {:.4}\n", quality.overall_score));
            for (task, score) in &quality.task_scores {
                output.push_str(&format!("  {}: {:.4}\n", task, score));
            }
        }

        output
    }
}

pub struct ReportWriter;

impl ReportWriter {
    pub fn write_json(&self, report: &BenchmarkReport, path: &Path) -> Result<()> {
        let json = report.to_json()?;

        let mut file = File::create(path).map_err(ReportWriteError::from)?;

        file.write_all(json.as_bytes())
            .map_err(ReportWriteError::from)?;

        Ok(())
    }

    pub fn write_console(&self, report: &BenchmarkReport) {
        println!("{}", report.to_console());
    }
}
