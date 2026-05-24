use anyhow::Result;
use clap::Parser;

mod cli;

fn main() -> Result<()> {
    let args = cli::Cli::parse();
    cli::dispatch(args)
}
