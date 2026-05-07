use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "llmr")]
#[command(
    about = "A tiny CLI for running optimised llama.cpp inference in Docker",
    long_about = None
)]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[arg(short, long, global = true)]
    pub quiet: bool,

    #[arg(long, global = true)]
    pub version: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Serve(ServeArgs),
    Status(StatusArgs),
    Stop(StopArgs),
    Profiles(ProfilesArgs),
    Tune(TuneArgs),
    Bench(BenchArgs),
    Doctor,
    Version,
    Update(UpdateArgs),
}

#[derive(Parser, Debug)]
pub struct ServeArgs {
    #[arg(short = 'm', long)]
    pub model: Option<String>,

    #[arg(short = 'p', long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long)]
    pub metrics: bool,

    #[arg(long)]
    pub benchmark: bool,

    #[arg(long)]
    pub no_benchmark: bool,

    #[arg(long)]
    pub skip_hardware: bool,

    #[arg(long)]
    pub dry_run: bool,

    #[arg(long)]
    pub public: bool,

    #[arg(long)]
    pub no_gpu: bool,

    #[arg(long)]
    pub quick: bool,

    #[arg(short, long)]
    pub auto: bool,

    #[arg(short = 't', long)]
    pub threads: Option<u32>,

    #[arg(short = 'c', long)]
    pub ctx_size: Option<u32>,

    #[arg(short = 'g', long)]
    pub gpu_layers: Option<u32>,

    #[arg(long)]
    pub split_mode: Option<SplitMode>,

    #[arg(short = 'b', long)]
    pub batch_size: Option<u32>,

    #[arg(short = 'u', long)]
    pub ubatch_size: Option<u32>,

    #[arg(long)]
    pub cache_type_k: Option<String>,

    #[arg(long)]
    pub cache_type_v: Option<String>,

    #[arg(long)]
    pub parallel: Option<u32>,

    #[arg(short, long)]
    pub debug: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum SplitMode {
    Layer,
    Row,
    None,
    Auto,
}

impl SplitMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Layer => "layer",
            Self::Row => "row",
            Self::None => "none",
            Self::Auto => "auto",
        }
    }
}

#[derive(Parser, Debug)]
pub struct StatusArgs {
    #[arg(short, long)]
    pub name: Option<String>,
}

#[derive(Parser, Debug)]
pub struct StopArgs {
    #[arg(short, long)]
    pub name: Option<String>,

    #[arg(long)]
    pub force: bool,
}

#[derive(Parser, Debug)]
pub struct ProfilesArgs {
    #[command(subcommand)]
    pub subcommand: Option<ProfilesSubcommand>,

    #[arg(long)]
    pub file: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum ProfilesSubcommand {
    List,
    Delete { key: String },
    Clear,
    Show { key: String },
}

#[derive(Parser, Debug)]
pub struct TuneArgs {
    #[arg(short = 'm', long)]
    pub model: Option<String>,

    #[arg(long)]
    pub dry_run: bool,

    #[arg(long)]
    pub quick: bool,

    #[arg(long)]
    pub max_rounds: Option<usize>,

    #[arg(long)]
    pub prompt_tokens: Option<u32>,

    #[arg(long)]
    pub generation_tokens: Option<u32>,
}

#[derive(Parser, Debug)]
pub struct BenchArgs {
    #[arg(short = 'm', long)]
    pub model: Option<String>,

    #[arg(short = 'u', long, default_value = "http://127.0.0.1:8080")]
    pub base_url: String,

    #[arg(short = 't', long)]
    pub test_type: Option<String>,

    #[arg(short = 'c', long)]
    pub config: Option<std::path::PathBuf>,

    #[arg(long)]
    pub tasks: Option<String>,

    #[arg(long, default_value = "5")]
    pub fewshot: u32,

    #[arg(long)]
    pub dry_run: bool,

    #[arg(long)]
    pub quick: bool,

    #[arg(long)]
    pub prompt_tokens: Option<u32>,

    #[arg(long)]
    pub generation_tokens: Option<u32>,

    #[arg(long)]
    pub parallel: Option<usize>,

    #[arg(long)]
    pub retries: Option<u32>,
}

#[derive(Parser, Debug)]
pub struct UpdateArgs {
    #[arg(long)]
    pub check: bool,
}
