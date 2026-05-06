use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("Config error: {0}")]
    Config(#[from] ConfigError),

    #[error("Health check error: {0}")]
    HealthCheck(#[from] HealthCheckError),

    #[error("Metadata error: {0}")]
    Metadata(#[from] MetadataError),

    #[error("Request error: {0}")]
    Request(#[from] RequestError),

    #[error("Stream parse error: {0}")]
    StreamParse(#[from] StreamParseError),

    #[error("Quality error: {0}")]
    Quality(#[from] QualityError),

    #[error("Report write error: {0}")]
    ReportWrite(#[from] ReportWriteError),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Validation error: {0}")]
    Validation(String),
}

#[derive(Debug, Error)]
pub enum HealthCheckError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Server returned unhealthy status: {0}")]
    Unhealthy(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum RequestError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Server error: {0}")]
    Server(String),
}

#[derive(Debug, Error)]
pub enum StreamParseError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid stream chunk: {0}")]
    InvalidChunk(String),
}

#[derive(Debug, Error)]
pub enum QualityError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Request error: {0}")]
    Request(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum ReportWriteError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to write report: {0}")]
    Write(String),
}

pub type Result<T> = std::result::Result<T, BenchError>;
pub type BenchResult<T> = std::result::Result<T, BenchError>;
