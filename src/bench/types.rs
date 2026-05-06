use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub endpoint: String,
    pub health_endpoint: String,
    pub props_endpoint: String,
    pub metrics_endpoint: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            endpoint: "/v1/chat/completions".to_string(),
            health_endpoint: "/health".to_string(),
            props_endpoint: "/props".to_string(),
            metrics_endpoint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelConfig {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub prompts: Vec<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub stream: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            prompts: vec!["Hello, how are you?".to_string()],
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            seed: 42,
            warmup_runs: 1,
            measured_runs: 3,
            stream: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityConfig {
    pub enabled: bool,
    pub tasks: Vec<String>,
    pub num_fewshot: Option<u32>,
    pub batch_size: Option<u32>,
    pub limit: Option<f64>,
    pub model_type: Option<String>,
}

impl QualityConfig {
    pub fn get_num_fewshot(&self) -> u32 {
        self.num_fewshot.unwrap_or(0)
    }

    pub fn get_batch_size(&self) -> &str {
        match self.batch_size {
            Some(n) if n > 0 => "auto",
            _ => "auto",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub performance: PerformanceConfig,
    pub quality: QualityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub truncated: bool,
    pub tokens: u32,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerStats {
    pub tokens_predicted: u32,
    pub tokens_evaluated: u32,
    pub prompt_tokens: u32,
    pub cache_matches: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamChunk {
    pub content: String,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerMetadata {
    pub model_name: Option<String>,
    pub ctx_size: Option<u32>,
    pub gpu_layers: Option<u32>,
    pub quantization: Option<String>,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceRecord {
    pub ttft_ms: u64,
    pub latency_ms: u64,
    pub tokens_generated: u32,
    pub tokens_per_sec: f64,
}
