use clap::Parser;
use llmr::bench::config::load_config;
use llmr::bench::orchestrator::BenchmarkOrchestrator;
use llmr::bench::report::ReportWriter;
use llmr::bench::Result;
use std::path::PathBuf;
use std::process::ExitCode;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[arg(short, long, default_value = "config.yaml")]
    pub config: PathBuf,

    #[arg(short, long)]
    pub output: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub verbose: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            if cli.verbose {
                EnvFilter::new("llmr::bench=debug")
            } else {
                EnvFilter::new("llmr::bench=info")
            }
        }))
        .init();

    tracing::info!("Starting llmr-bench");

    if let Err(e) = run(&cli) {
        eprintln!("Error: {}", e);
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}

fn run(cli: &Cli) -> Result<()> {
    let config = load_config(&cli.config)?;

    tracing::info!("Loaded config from {:?}", cli.config);

    let orchestrator = BenchmarkOrchestrator::new(config);

    tracing::info!("Running benchmark...");

    let report = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(orchestrator.run())?;

    println!("\n{}", report.to_console());

    if let Some(path) = &cli.output {
        let writer = ReportWriter;
        writer.write_json(&report, path)?;
        tracing::info!("Results written to {:?}", path);
    }

    Ok(())
}
