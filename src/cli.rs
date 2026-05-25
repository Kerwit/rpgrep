use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use rpgrep::index::store::IndexStore;

#[derive(Parser)]
#[command(name = "rpgrep")]
#[command(about = "Búsqueda probabilística para código (Xor + BM25 + MinHash + QUBO)", long_about = None)]
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

        /// Extensiones a indexar (sin punto). Por defecto: rs.
        /// Pasa `--ext ""` para indexar TODOS los archivos.
        #[arg(long, value_delimiter = ',', default_values_t = vec!["rs".to_string()])]
        ext: Vec<String>,
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
            ext,
        } => {
            eprintln!("[rpgrep] Indexando {} → {}", path.display(), out.display());
            let exts: Vec<&str> = ext.iter().map(|s| s.as_str()).collect();
            let t0 = Instant::now();
            let store = IndexStore::from_dir(&path, &exts, lines, overlap)
                .with_context(|| format!("construir índice de {}", path.display()))?;
            eprintln!(
                "[rpgrep] {} chunks, {} archivos indexados en {:.2}s",
                store.chunks.len(),
                store.n_files(),
                t0.elapsed().as_secs_f64()
            );
            store
                .save(&out)
                .with_context(|| format!("persistir índice en {}", out.display()))?;
            eprintln!("[rpgrep] índice guardado en {}", out.display());
            Ok(())
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
            let store = IndexStore::load(&index).context("cargando índice persistido")?;
            println!("ruta:           {}", index.display());
            println!("archivos:       {}", store.n_files());
            println!("chunks:         {}", store.chunks.len());
            println!("bm25 n_docs:    {}", store.bm25.n_docs);
            println!("bm25 avg_dl:    {:.1} tokens", store.bm25.avg_doc_len);
            println!("vocab (tokens): {}", store.bm25.doc_freq.len());
            println!("minhash sigs:   {} firmas", store.minhash.len());
            Ok(())
        }
    }
}
