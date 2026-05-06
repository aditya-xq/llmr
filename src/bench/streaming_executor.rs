use crate::bench::error::Result;
use crate::bench::types::{PerformanceRecord, StreamChunk};
use futures::Stream;
use futures::StreamExt;
use std::time::Instant;

pub struct StreamingExecutor;

impl Default for StreamingExecutor {
    fn default() -> Self {
        Self
    }
}

impl StreamingExecutor {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute<S>(&self, mut stream: S) -> Result<(String, PerformanceRecord)>
    where
        S: Stream<Item = Result<StreamChunk>> + Unpin,
    {
        let start_time = Instant::now();
        let mut ttft_ms: Option<u64> = None;
        let mut content = String::new();
        let mut chunk_count = 0u32;

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => return Err(e),
            };

            if chunk_count == 0 {
                let elapsed = start_time.elapsed().as_millis() as u64;
                ttft_ms = Some(elapsed);
            }

            if chunk.done {
                break;
            }

            if !chunk.content.is_empty() {
                content.push_str(&chunk.content);
                chunk_count += 1;
            }
        }

        let end_time = start_time.elapsed();
        let latency_ms = end_time.as_millis() as u64;
        let ttft = ttft_ms.unwrap_or(latency_ms);

        let tokens_generated = estimate_tokens(&content);
        let tokens_per_sec = if latency_ms > 0 {
            (tokens_generated as f64 / latency_ms as f64) * 1000.0
        } else {
            0.0
        };

        let record = PerformanceRecord {
            ttft_ms: ttft,
            latency_ms,
            tokens_generated,
            tokens_per_sec,
        };

        Ok((content, record))
    }

    pub fn parse_sse_chunk(line: &str) -> Option<StreamChunk> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        if !line.starts_with("data: ") {
            return None;
        }

        let data = line.strip_prefix("data: ")?;
        if data == "[DONE]" {
            return Some(StreamChunk {
                content: String::new(),
                done: true,
            });
        }

        let json: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return None,
        };

        let content = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str())
            .map(String::from)
            .unwrap_or_default();

        Some(StreamChunk {
            content,
            done: false,
        })
    }
}

fn estimate_tokens(text: &str) -> u32 {
    text.chars().count() as u32 / 4
}
