pub mod bench;
pub mod cli;
pub mod diagnostics;
pub mod docker;
pub mod errors;
pub mod hardware;
pub mod models;
pub mod tuning;
pub mod utils;

pub use bench::{BenchError, Result};

pub use cli::{Args, Commands};
pub use diagnostics::{DockerCheck, EnvCheck, HardwareCheck};
pub use docker::{DockerClient, DockerInstallStatus};
pub use errors::Result as ErrorResult;
pub use hardware::{CpuInfo, GpuInfo, HardwareInfo, RamInfo};
pub use models::{ModelInfo, ModelScanner, Profile, ProfileManager};
pub use tuning::{Backend, LlamaCppProfile};
pub use utils::{gpu_style, Logger, Style};
