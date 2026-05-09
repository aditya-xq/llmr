use crate::bench::types::PerformanceRecord;
use chrono::{DateTime, Utc};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkMetrics {
    pub timestamp: DateTime<Utc>,
    pub iterations: u32,
    pub total_tokens: u64,
    pub total_time_ms: u64,
    pub first_token_ms: Option<u64>,
    pub tokens_per_second: f64,
    pub ttft_ms: Option<f64>,
}

impl BenchmarkMetrics {
    pub fn calculate(total_tokens: u64, total_time_ms: u64, first_token_ms: Option<u64>) -> Self {
        let tokens_per_second = if total_time_ms > 0 {
            (total_tokens as f64) / (total_time_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let ttft_ms = first_token_ms.map(|ft| ft as f64);

        Self {
            timestamp: Utc::now(),
            iterations: 0,
            total_tokens,
            total_time_ms,
            first_token_ms,
            tokens_per_second,
            ttft_ms,
        }
    }
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            iterations: 0,
            total_tokens: 0,
            total_time_ms: 0,
            first_token_ms: None,
            tokens_per_second: 0.0,
            ttft_ms: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSummary {
    pub latency_avg: f64,
    pub latency_min: f64,
    pub latency_max: f64,
    pub latency_p50: f64,
    pub latency_p90: f64,
    pub latency_p95: f64,
    pub latency_p99: f64,
    pub latency_stddev: f64,
    pub latency_cv: f64,
    pub ttft_avg: f64,
    pub ttft_min: f64,
    pub ttft_max: f64,
    pub tokens_per_sec_avg: f64,
    pub tokens_per_sec_min: f64,
    pub tokens_per_sec_max: f64,
    pub error_rate: f64,
    pub successful_runs: usize,
    pub total_runs: usize,
}

pub struct MetricsAggregator {
    records: Vec<PerformanceRecord>,
    failed_count: usize,
    warmup_runs: usize,
}

impl MetricsAggregator {
    pub fn new(warmup_runs: usize) -> Self {
        Self {
            records: Vec::new(),
            failed_count: 0,
            warmup_runs,
        }
    }

    pub fn add_result(&mut self, record: Option<PerformanceRecord>) {
        if let Some(r) = record {
            self.records.push(r);
        } else {
            self.failed_count += 1;
        }
    }

    pub fn summarize(&self) -> PerformanceSummary {
        let measured: Vec<_> = self.records.iter().skip(self.warmup_runs).collect();

        let total_runs = self.records.len() + self.failed_count;
        let successful = measured.len();
        let error_rate = if total_runs > 0 {
            self.failed_count as f64 / total_runs as f64
        } else {
            0.0
        };

        let latency_avg = if successful > 0 {
            measured.iter().map(|r| r.latency_ms as f64).sum::<f64>() / successful as f64
        } else {
            0.0
        };

        let latency_p50 = calculate_percentile(&measured, |r| r.latency_ms as f64, 50);
        let latency_p90 = calculate_percentile(&measured, |r| r.latency_ms as f64, 90);
        let latency_p95 = calculate_percentile(&measured, |r| r.latency_ms as f64, 95);
        let latency_p99 = calculate_percentile(&measured, |r| r.latency_ms as f64, 99);

        let (latency_min, latency_max) = if !measured.is_empty() {
            let latencies: Vec<f64> = measured.iter().map(|r| r.latency_ms as f64).collect();
            (
                latencies.iter().cloned().fold(f64::INFINITY, f64::min),
                latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        } else {
            (0.0, 0.0)
        };

        let latency_stddev = if successful > 1 {
            let variance = measured
                .iter()
                .map(|r| {
                    let diff = r.latency_ms as f64 - latency_avg;
                    diff * diff
                })
                .sum::<f64>()
                / (successful - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let latency_cv = if latency_avg > 0.0 {
            (latency_stddev / latency_avg) * 100.0
        } else {
            0.0
        };

        let ttft_avg = if successful > 0 {
            measured.iter().map(|r| r.ttft_ms as f64).sum::<f64>() / successful as f64
        } else {
            0.0
        };

        let (ttft_min, ttft_max) = if !measured.is_empty() {
            let ttfts: Vec<f64> = measured.iter().map(|r| r.ttft_ms as f64).collect();
            (
                ttfts.iter().cloned().fold(f64::INFINITY, f64::min),
                ttfts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        } else {
            (0.0, 0.0)
        };

        let tokens_per_sec_avg = if successful > 0 {
            measured.iter().map(|r| r.tokens_per_sec).sum::<f64>() / successful as f64
        } else {
            0.0
        };

        let (tokens_per_sec_min, tokens_per_sec_max) = if !measured.is_empty() {
            let tps: Vec<f64> = measured.iter().map(|r| r.tokens_per_sec).collect();
            (
                tps.iter().cloned().fold(f64::INFINITY, f64::min),
                tps.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        } else {
            (0.0, 0.0)
        };

        PerformanceSummary {
            latency_avg,
            latency_min,
            latency_max,
            latency_p50,
            latency_p90,
            latency_p95,
            latency_p99,
            latency_stddev,
            latency_cv,
            ttft_avg,
            ttft_min,
            ttft_max,
            tokens_per_sec_avg,
            tokens_per_sec_min,
            tokens_per_sec_max,
            error_rate,
            successful_runs: successful,
            total_runs,
        }
    }
}

fn calculate_percentile<F>(records: &[&PerformanceRecord], accessor: F, percentile: u32) -> f64
where
    F: Fn(&PerformanceRecord) -> f64,
{
    if records.is_empty() {
        return 0.0;
    }

    let mut values: Vec<f64> = records.iter().map(|r| accessor(r)).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = (percentile as f64 / 100.0 * (values.len() - 1) as f64).round() as usize;
    values[index.min(values.len() - 1)]
}
