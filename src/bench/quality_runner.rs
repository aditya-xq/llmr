use crate::bench::error::{QualityError, Result};
use crate::bench::metrics::BenchmarkMetrics;
use crate::bench::types::BenchmarkConfig;
use std::collections::HashMap;
use std::process::Command;
use std::time::Duration;

pub struct QualityRunner {
    config: BenchmarkConfig,
    base_url: String,
}

pub struct QualityResult {
    pub task_scores: HashMap<String, f64>,
    pub overall_score: f64,
}

impl QualityRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        let base_url = Self::extract_base_url(&config.server.endpoint);
        Self { config, base_url }
    }

    pub async fn run(&self, _prompt: &str) -> Result<BenchmarkMetrics> {
        if self.config.quality.tasks.is_empty() {
            return Ok(BenchmarkMetrics::default());
        }

        if !self.config.quality.enabled {
            return Ok(BenchmarkMetrics::default());
        }

        let tasks = self.config.quality.tasks.join(",");
        self.ensure_lm_eval_installed().await?;
        let output = self.run_evaluation(&tasks).await?;

        Ok(BenchmarkMetrics {
            timestamp: chrono::Utc::now(),
            iterations: 0,
            total_tokens: 0,
            total_time_ms: 0,
            first_token_ms: None,
            tokens_per_second: output.overall_score,
            ttft_ms: None,
        })
    }

    async fn ensure_lm_eval_installed(&self) -> Result<()> {
        use tokio::process::Command as TokioCommand;

        let check = TokioCommand::new("python")
            .args(["-c", "import lm_eval"])
            .output()
            .await;

        if check.is_err() || !check.as_ref().map(|o| o.status.success()).unwrap_or(false) {
            return Err(QualityError::Request(
                "lm_eval not installed. Install with: pip install lm_eval[api]".to_string(),
            )
            .into());
        }

        Ok(())
    }

    async fn run_evaluation(&self, tasks: &str) -> Result<QualityResult> {
        let base_url = format!("{}/v1", self.base_url);
        let num_fewshot = self.config.quality.get_num_fewshot();

        let mut cmd = Command::new("python");
        cmd.arg("-m").arg("lm_eval");
        cmd.arg("run");
        cmd.arg("--model").arg("local-chat-completions");
        cmd.arg("--model_args")
            .arg(format!("base_url={}", base_url));
        cmd.arg("--tasks").arg(tasks);
        cmd.arg("--num_fewshot").arg(num_fewshot.to_string());
        cmd.arg("--output_path").arg("/tmp/lm_eval_results");

        if let Some(limit) = self.config.quality.limit {
            cmd.arg("--limit").arg(limit.to_string());
        }

        let output = tokio::time::timeout(
            Duration::from_secs(600),
            tokio::task::spawn_blocking(move || cmd.output()),
        )
        .await
        .map_err(|_| QualityError::Request("Evaluation timed out".to_string()))?
        .map_err(|e| QualityError::Io(std::io::Error::other(e.to_string())))?
        .map_err(|e| QualityError::Io(std::io::Error::other(e.to_string())))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(QualityError::Request(format!("Evaluation failed: {}", stderr)).into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result = Self::parse_output(&stdout)?;

        Ok(result)
    }

    fn parse_output(output: &str) -> Result<QualityResult> {
        let mut task_scores = HashMap::new();
        let mut overall_score = 0.0;

        for line in output.lines() {
            if line.contains("|") && line.contains("acc") {
                let parts: Vec<&str> = line.split('|').filter(|s| !s.is_empty()).collect();
                if parts.len() >= 3 {
                    let task_name = parts[0].trim();
                    let score_str = parts[1].trim().replace("acc", "").replace(":", "");
                    if let Ok(score) = score_str.parse::<f64>() {
                        task_scores.insert(task_name.to_string(), score);
                        overall_score += score;
                    }
                }
            }
        }

        let count = task_scores.len() as f64;
        if count > 0.0 {
            overall_score /= count;
        }

        Ok(QualityResult {
            task_scores,
            overall_score,
        })
    }

    fn extract_base_url(endpoint: &str) -> String {
        if endpoint.starts_with("http") {
            endpoint.split('/').take(3).collect::<Vec<_>>().join("/")
        } else {
            "http://127.0.0.1:8080".to_string()
        }
    }
}
