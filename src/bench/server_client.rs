use crate::bench::error::{HealthCheckError, MetadataError, RequestError, Result};
use crate::bench::types::*;
use futures::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::Serialize;
use std::pin::Pin;
use std::time::Duration;

const DEFAULT_TIMEOUT_SECS: u64 = 300;
const REQUEST_TIMEOUT_SECS: u64 = 120;

pub struct ServerClient {
    client: Client,
    base_url: String,
    health_endpoint: String,
    props_endpoint: String,
    chat_endpoint: String,
}

impl ServerClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
            health_endpoint: "/health".to_string(),
            props_endpoint: "/props".to_string(),
            chat_endpoint: "/v1/chat/completions".to_string(),
        }
    }

    pub fn with_endpoints(
        base_url: &str,
        health_endpoint: &str,
        props_endpoint: &str,
        chat_endpoint: &str,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
            health_endpoint: health_endpoint.to_string(),
            props_endpoint: props_endpoint.to_string(),
            chat_endpoint: chat_endpoint.to_string(),
        }
    }

    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}{}", self.base_url, self.health_endpoint);
        let response = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(HealthCheckError::from)?;

        if !response.status().is_success() {
            return Err(HealthCheckError::Unhealthy(response.status().to_string()).into());
        }

        let health: HealthResponse = response.json().await.map_err(HealthCheckError::from)?;

        if health.status != "ok" || !health.model_loaded {
            return Err(HealthCheckError::Unhealthy(health.status).into());
        }

        Ok(())
    }

    pub async fn fetch_props(&self) -> Result<ServerMetadata> {
        let url = format!("{}{}", self.base_url, self.props_endpoint);
        let response = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(MetadataError::from)?;

        if !response.status().is_success() {
            return Err(MetadataError::Http(response.error_for_status().unwrap_err()).into());
        }

        let raw: serde_json::Value = response.json().await.map_err(MetadataError::from)?;

        let model_name = raw
            .get("model_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let ctx_size = raw
            .get("ctx_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let gpu_layers = raw
            .get("gpu_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let quantization = raw
            .get("quantization")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ServerMetadata {
            model_name,
            ctx_size,
            gpu_layers,
            quantization,
            raw,
        })
    }

    pub async fn send_chat_request(
        &self,
        request: &BenchmarkRequest,
    ) -> Result<CompletionResponse> {
        #[derive(Serialize)]
        struct ChatRequestPayload<'a> {
            messages: Vec<Message<'a>>,
            max_tokens: u32,
            temperature: f32,
            top_p: f32,
            seed: u64,
            stream: bool,
        }

        #[derive(Serialize)]
        struct Message<'a> {
            role: &'a str,
            content: &'a str,
        }

        let payload = ChatRequestPayload {
            messages: vec![Message {
                role: "user",
                content: &request.prompt,
            }],
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            stream: false,
        };

        let url = format!("{}{}", self.base_url, self.chat_endpoint);
        let response = self
            .client
            .post(&url)
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .json(&payload)
            .send()
            .await
            .map_err(RequestError::from)?;

        if !response.status().is_success() {
            return Err(RequestError::Server(format!(
                "Server returned error: {}",
                response.status()
            ))
            .into());
        }

        let body: serde_json::Value = response.json().await.map_err(RequestError::from)?;

        let content = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(String::from)
            .unwrap_or_default();

        let truncated = body
            .get("truncated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let tokens = body
            .get("usage")
            .and_then(|v| v.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(0);

        let stop_reason = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .map(String::from);

        Ok(CompletionResponse {
            content,
            truncated,
            tokens,
            stop_reason,
        })
    }

    pub async fn stream_chat_request(
        &self,
        request: &BenchmarkRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>>>>> {
        #[derive(Serialize)]
        struct ChatRequestPayload<'a> {
            messages: Vec<Message<'a>>,
            max_tokens: u32,
            temperature: f32,
            top_p: f32,
            seed: u64,
            stream: bool,
        }

        #[derive(Serialize)]
        struct Message<'a> {
            role: &'a str,
            content: &'a str,
        }

        let payload = ChatRequestPayload {
            messages: vec![Message {
                role: "user",
                content: &request.prompt,
            }],
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            stream: true,
        };

        let url = format!("{}{}", self.base_url, self.chat_endpoint);
        let response = self
            .client
            .post(&url)
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .json(&payload)
            .send()
            .await
            .map_err(RequestError::from)?;

        if !response.status().is_success() {
            return Err(RequestError::Server(format!(
                "Server returned error: {}",
                response.status()
            ))
            .into());
        }

        let stream = response.bytes_stream().map(move |chunk_result| {
            let chunk = chunk_result.map_err(RequestError::from)?;
            let text = String::from_utf8_lossy(&chunk);

            let content = text.chars().filter(|c| *c != '\n').collect::<String>();

            Ok(StreamChunk {
                content,
                done: false,
            })
        });

        Ok(Box::pin(stream))
    }
}
