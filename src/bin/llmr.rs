use clap::Parser;

async fn check_for_updates_subtle() {
    use llmr::utils::Style;

    let current = env!("CARGO_PKG_VERSION");
    let url = "https://api.github.com/repos/aditya-xq/llmr/releases/latest";

    if let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        if let Ok(resp) = client.get(url).send().await {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                let latest = json
                    .get("tag_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0.0.0")
                    .trim_start_matches('v');

                if latest != current {
                    let style = Style::default();
                    eprintln!(
                        "{} v{} available (you have {}) - run {} to update",
                        style.info("→"),
                        latest,
                        current,
                        style.accent("llmr update")
                    );
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> llmr::ErrorResult<()> {
    let global_args = llmr::cli::Args::parse();

    if let Err(e) = llmr::utils::Logger::setup(global_args.verbose, global_args.quiet) {
        eprintln!("Warning: {}", e);
    }

    let style = llmr::utils::Style::default();

    let skip_update_check = matches!(
        global_args.command,
        Some(llmr::cli::Commands::Version)
            | Some(llmr::cli::Commands::Update(_))
            | Some(llmr::cli::Commands::Doctor)
    ) || global_args.version;

    if global_args.version {
        llmr::cli::VersionCommand::execute().await?;
        return Ok(());
    }

    if !skip_update_check {
        tokio::spawn(async {
            check_for_updates_subtle().await;
        });
    }

    match global_args.command {
        None => {
            eprintln!("Error: no subcommand provided. Run with --help for usage information.");
            std::process::exit(1);
        }
        Some(llmr::cli::Commands::Serve(args)) => {
            let command = llmr::cli::ServeCommand::new(args, style);
            command.execute().await?;
        }
        Some(llmr::cli::Commands::Status(args)) => {
            let command = llmr::cli::StatusCommand::new(args, style);
            command.execute().await?;
        }
        Some(llmr::cli::Commands::Stop(args)) => {
            let command = llmr::cli::StopCommand::new(args, style);
            command.execute().await?;
        }
        Some(llmr::cli::Commands::Profiles(args)) => {
            let command = llmr::cli::ProfilesCommand::new(args, style);
            command.execute().await?;
        }
        Some(llmr::cli::Commands::Tune(args)) => {
            let command = llmr::cli::TuneCommand::new(args, style);
            command.execute().await?;
        }
        Some(llmr::cli::Commands::Bench(args)) => {
            let command = llmr::cli::BenchCommand::new(args, style);
            command.execute().await?;
        }
        Some(llmr::cli::Commands::Doctor) => {
            let command = llmr::cli::DoctorCommand::new(style);
            command.execute().await?;
        }
        Some(llmr::cli::Commands::Version) => {
            llmr::cli::VersionCommand::execute().await?;
        }
        Some(llmr::cli::Commands::Update(args)) => {
            let command = llmr::cli::UpdateCommand::new(args, style);
            command.execute().await?;
        }
    }

    Ok(())
}
