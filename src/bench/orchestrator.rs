use crate::bench::error::Result;
use crate::bench::metrics::{BenchmarkMetrics, MetricsAggregator, PerformanceSummary};
use crate::bench::quality_runner::QualityRunner;
use crate::bench::report::BenchmarkReport;
use crate::bench::server_client::ServerClient;
use crate::bench::streaming_executor::StreamingExecutor;
use crate::bench::types::{BenchmarkConfig, BenchmarkRequest, PerformanceRecord, ServerMetadata};
use std::time::Instant;

pub struct BenchmarkOrchestrator {
    config: BenchmarkConfig,
    server_client: ServerClient,
    quality_runner: QualityRunner,
}

impl BenchmarkOrchestrator {
    pub fn new(config: BenchmarkConfig) -> Self {
        let base_url = Self::extract_base_url(&config.server.endpoint);
        let server_client = ServerClient::with_endpoints(
            &base_url,
            &config.server.health_endpoint,
            &config.server.props_endpoint,
            &config.server.endpoint,
        );
        let quality_runner = QualityRunner::new(config.clone());

        Self {
            config,
            server_client,
            quality_runner,
        }
    }

    pub async fn run(&self) -> Result<BenchmarkReport> {
        let prompt = self
            .config
            .performance
            .prompts
            .first()
            .cloned()
            .unwrap_or_else(|| "Hello, how are you?".to_string());

        let metadata = self.run_health_and_metadata().await?;

        self.run_warmup(&prompt).await?;

        let summary = self.run_measured(&prompt).await?;

        let quality_metrics = if self.config.quality.enabled {
            Some(self.run_quality(&prompt).await?)
        } else {
            None
        };

        let report =
            BenchmarkReport::from_summary(&self.config, summary, metadata, quality_metrics);

        Ok(report)
    }

    async fn run_health_and_metadata(&self) -> Result<ServerMetadata> {
        self.server_client.health_check().await?;

        let metadata = self.server_client.fetch_props().await?;

        Ok(metadata)
    }

    async fn run_warmup(&self, prompt: &str) -> Result<()> {
        let warmup_runs = self.config.performance.warmup_runs as usize;

        for _ in 0..warmup_runs {
            let request = self.build_request(prompt, false);

            if self.config.performance.stream {
                let stream = self.server_client.stream_chat_request(&request).await?;
                let executor = StreamingExecutor::new();
                let _ = executor.execute(stream).await;
            } else {
                let _ = self.server_client.send_chat_request(&request).await?;
            }
        }

        Ok(())
    }

    async fn run_measured(&self, prompt: &str) -> Result<PerformanceSummary> {
        let measured_runs = self.config.performance.measured_runs as usize;
        let warmup_runs = self.config.performance.warmup_runs as usize;

        let mut aggregator = MetricsAggregator::new(warmup_runs);

        for _ in 0..measured_runs {
            let request = self.build_request(prompt, false);
            let start_time = Instant::now();

            let record = if self.config.performance.stream {
                self.execute_streaming(request).await.ok()
            } else {
                match self.execute_non_streaming(request).await {
                    Ok(_) => {
                        let elapsed = Instant::now().duration_since(start_time);
                        let latency_ms = elapsed.as_millis() as u64;
                        Some(PerformanceRecord {
                            ttft_ms: latency_ms / 2,
                            latency_ms,
                            tokens_generated: self.config.performance.max_tokens,
                            tokens_per_sec: (self.config.performance.max_tokens as f64
                                / latency_ms as f64)
                                * 1000.0,
                        })
                    }
                    Err(_) => None,
                }
            };

            aggregator.add_result(record);
        }

        Ok(aggregator.summarize())
    }

    async fn execute_streaming(&self, request: BenchmarkRequest) -> Result<PerformanceRecord> {
        let stream = self.server_client.stream_chat_request(&request).await?;
        let executor = StreamingExecutor::new();
        let (_content, record) = executor.execute(stream).await?;
        Ok(record)
    }

    async fn execute_non_streaming(&self, request: BenchmarkRequest) -> Result<()> {
        let _response = self.server_client.send_chat_request(&request).await?;
        Ok(())
    }

    async fn run_quality(&self, prompt: &str) -> Result<BenchmarkMetrics> {
        self.quality_runner.run(prompt).await
    }

    fn build_request(&self, prompt: &str, stream: bool) -> BenchmarkRequest {
        BenchmarkRequest {
            prompt: prompt.to_string(),
            max_tokens: self.config.performance.max_tokens,
            temperature: self.config.performance.temperature,
            top_p: self.config.performance.top_p,
            seed: self.config.performance.seed,
            stream,
        }
    }

    fn extract_base_url(endpoint: &str) -> String {
        if endpoint.starts_with("http") {
            endpoint.split('/').take(3).collect::<Vec<_>>().join("/")
        } else {
            "http://127.0.0.1:8080".to_string()
        }
    }
}

impl BenchmarkReport {
    pub fn from_summary(
        config: &BenchmarkConfig,
        summary: PerformanceSummary,
        metadata: ServerMetadata,
        quality_metrics: Option<BenchmarkMetrics>,
    ) -> Self {
        let model_name = &config.model.name;
        Self::new(
            model_name,
            config,
            Some(metadata),
            Some(summary),
            quality_metrics,
        )
    }
}
