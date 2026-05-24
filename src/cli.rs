use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rpgrep")]
#[command(about = "Búsqueda semántica probabilística para código", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Construye o actualiza el índice sobre un directorio
    Index {
        /// Directorio raíz a indexar
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Directorio donde persistir el índice
        #[arg(long, default_value = ".rpgrep")]
        out: PathBuf,

        /// Líneas por chunk
        #[arg(long, default_value_t = 40)]
        lines: usize,

        /// Líneas de solapamiento entre chunks consecutivos
        #[arg(long, default_value_t = 8)]
        overlap: usize,
    },

    /// Búsqueda semántica con presupuesto de tokens
    Search {
        /// Query en lenguaje natural
        #[arg(value_name = "QUERY")]
        query: String,

        /// Presupuesto máximo de tokens en el contexto final
        #[arg(long, default_value_t = 4000)]
        budget: usize,

        /// Directorio del índice persistido
        #[arg(long, default_value = ".rpgrep")]
        index: PathBuf,

        /// Top-K aproximado antes del QUBO
        #[arg(long, default_value_t = 50)]
        topk: usize,
    },

    /// Estadísticas del índice persistido
    Stats {
        #[arg(long, default_value = ".rpgrep")]
        index: PathBuf,
    },
}

pub fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Index {
            path,
            out,
            lines,
            overlap,
        } => {
            eprintln!("[rpgrep] Indexando {} → {}", path.display(), out.display());
            // TODO (siguiente paso de implementación):
            //   1. walkdir sobre `path`
            //   2. para cada archivo: chunk::chunk_file(path, lines, overlap)
            //   3. embedder.embed_batch(textos)
            //   4. construir HnswIndex y FileBloomIndex
            //   5. IndexStore::save(&out)
            anyhow::bail!("Comando `index` pendiente de cablear — ver TODO en src/cli.rs");
        }
        Commands::Search {
            query,
            budget,
            index,
            topk,
        } => {
            eprintln!(
                "[rpgrep] Búsqueda \"{}\" (budget={}, topk={})",
                query, budget, topk
            );
            let pipeline =
                rpgrep::SearchPipeline::load(&index).context("cargando índice persistido")?;
            let results = pipeline.search(&query, budget, topk)?;
            for r in results {
                println!(
                    "{}:{}-{}  score={:.3}",
                    r.chunk.file.display(),
                    r.chunk.start_line,
                    r.chunk.end_line,
                    r.score,
                );
            }
            Ok(())
        }
        Commands::Stats { index } => {
            eprintln!("[rpgrep] Stats de {}", index.display());
            // TODO: cargar IndexStore y emitir contadores (chunks, archivos, dim).
            Ok(())
        }
    }
}
