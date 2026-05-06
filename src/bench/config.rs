use crate::bench::error::{BenchError, ConfigError, Result};
use crate::bench::types::*;
use std::path::Path;
use std::str::FromStr;

pub fn load_config(path: &Path) -> Result<BenchmarkConfig> {
    let contents = std::fs::read_to_string(path).map_err(ConfigError::from)?;

    let config: BenchmarkConfig = serde_yaml::from_str(&contents).map_err(ConfigError::from)?;

    validate_config(&config)?;

    Ok(config)
}

fn validate_config(config: &BenchmarkConfig) -> Result<()> {
    if config.performance.measured_runs == 0 {
        return Err(
            ConfigError::Validation("measured_runs must be greater than 0".to_string()).into(),
        );
    }

    if config.performance.max_tokens == 0 {
        return Err(
            ConfigError::Validation("max_tokens must be greater than 0".to_string()).into(),
        );
    }

    let temp = config.performance.temperature;
    if !(0.0..=2.0).contains(&temp) {
        return Err(ConfigError::Validation(format!(
            "temperature must be in range [0, 2], got {}",
            temp
        ))
        .into());
    }

    let top_p = config.performance.top_p;
    if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        return Err(ConfigError::Validation(format!(
            "top_p must be in range (0, 1], got {}",
            top_p
        ))
        .into());
    }

    if config.performance.warmup_runs > config.performance.measured_runs {
        return Err(ConfigError::Validation(
            "warmup_runs must not exceed measured_runs".to_string(),
        )
        .into());
    }

    if config.performance.prompts.is_empty() {
        return Err(ConfigError::Validation("at least one prompt is required".to_string()).into());
    }

    for prompt in &config.performance.prompts {
        if prompt.is_empty() {
            return Err(ConfigError::Validation("prompts cannot be empty".to_string()).into());
        }
    }

    if config.quality.enabled && config.quality.tasks.is_empty() {
        return Err(ConfigError::Validation(
            "quality.tasks must be specified when quality is enabled".to_string(),
        )
        .into());
    }

    if let Some(fewshot) = config.quality.num_fewshot {
        if fewshot > 10 {
            return Err(ConfigError::Validation(
                "num_fewshot should typically be <= 10".to_string(),
            )
            .into());
        }
    }

    if let Some(limit) = config.quality.limit {
        if !(0.0..=1.0).contains(&limit) {
            return Err(ConfigError::Validation(
                "quality.limit must be in range (0, 1]".to_string(),
            )
            .into());
        }
    }

    Ok(())
}

impl FromStr for BenchmarkConfig {
    type Err = BenchError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let config: BenchmarkConfig = serde_yaml::from_str(s).map_err(ConfigError::from)?;
        validate_config(&config)?;
        Ok(config)
    }
}
