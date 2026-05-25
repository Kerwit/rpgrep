use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode};
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

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

    /// Sirve queries contra un índice en memoria vía Unix socket.
    /// Protocolo: JSON-line (una request = una response = cierre).
    Serve {
        /// Directorio del índice persistido a cargar al arrancar.
        #[arg(long, default_value = ".rpgrep")]
        index: PathBuf,

        /// Path del socket Unix. Default: `<index>/rpgrep.sock`.
        #[arg(long)]
        socket: Option<PathBuf>,
    },

    /// Mantiene un índice sincronizado con un directorio: indexa
    /// inicialmente y re-indexa al detectar cambios.
    Watch {
        #[arg(value_name = "PATH")]
        path: PathBuf,

        #[arg(long, default_value = ".rpgrep")]
        out: PathBuf,

        #[arg(long, default_value_t = 40)]
        lines: usize,

        #[arg(long, default_value_t = 8)]
        overlap: usize,

        #[arg(long, value_delimiter = ',', default_values_t = vec!["rs".to_string()])]
        ext: Vec<String>,

        /// Ventana de quiet period antes de re-indexar (ms).
        #[arg(long, default_value_t = 500)]
        debounce_ms: u64,
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
                    r.chunk.file,
                    r.chunk.start_line,
                    r.chunk.end_line,
                    r.score,
                );
            }
            Ok(())
        }
        Commands::Serve { index, socket } => {
            let sock = socket.unwrap_or_else(|| rpgrep::serve::default_socket_path(&index));
            rpgrep::serve::run(&index, &sock).map_err(anyhow::Error::from)
        }
        Commands::Watch {
            path,
            out,
            lines,
            overlap,
            ext,
            debounce_ms,
        } => run_watch(path, out, lines, overlap, ext, debounce_ms),
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

/// Indexa inicialmente y luego mantiene el índice sincronizado vía
/// `notify-debouncer-mini`. Estrategia: re-index completo en cada
/// quiet period (BM25 mantiene estadísticas globales — `avg_doc_len`,
/// `doc_freq` — que un update incremental tendría que recomputar de
/// todas formas). Filtra eventos del `out` y por extensión para
/// evitar loops (escribir `rpgrep.idx` no debe disparar otro rebuild)
/// y churn de build artifacts (target/, node_modules/, …).
fn run_watch(
    path: PathBuf,
    out: PathBuf,
    lines: usize,
    overlap: usize,
    ext: Vec<String>,
    debounce_ms: u64,
) -> Result<()> {
    let exts: Vec<&str> = ext.iter().map(|s| s.as_str()).collect();

    let rebuild = |reason: &str| -> Result<()> {
        let t0 = Instant::now();
        let store = IndexStore::from_dir(&path, &exts, lines, overlap)
            .with_context(|| format!("indexar {}", path.display()))?;
        let chunks = store.chunks.len();
        let files = store.n_files();
        store
            .save(&out)
            .with_context(|| format!("persistir índice en {}", out.display()))?;
        eprintln!(
            "[rpgrep watch] {reason}: {chunks} chunks, {files} archivos, {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    };

    eprintln!(
        "[rpgrep watch] raíz={} out={} debounce={debounce_ms}ms ext={:?}",
        path.display(),
        out.display(),
        ext
    );
    rebuild("inicial")?;

    // Canonicalizamos `out` tras el primer save: ya existe en disco y
    // podemos comparar prefijos para filtrar eventos auto-disparados.
    let out_canonical = std::fs::canonicalize(&out).ok();

    let (tx, rx) = channel();
    let mut debouncer =
        new_debouncer(Duration::from_millis(debounce_ms), tx).context("crear watcher de notify")?;
    debouncer
        .watcher()
        .watch(&path, RecursiveMode::Recursive)
        .with_context(|| format!("vigilar {}", path.display()))?;
    eprintln!("[rpgrep watch] vigilando cambios. Ctrl-C para salir.");

    for res in rx {
        match res {
            Ok(events) if !events.is_empty() => {
                let relevant_count = events
                    .iter()
                    .filter(|e| event_is_relevant(&e.path, &out_canonical, &exts))
                    .count();
                if relevant_count == 0 {
                    continue;
                }
                rebuild(&format!("re-index tras {relevant_count} cambio(s)"))?;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[rpgrep watch] error: {e:?}");
            }
        }
    }

    Ok(())
}

/// Un evento es relevante si:
/// (1) su path NO está bajo el directorio de output (evita el loop
///     "guardo el .idx → notify lo ve → rebuild → guardo el .idx"); y
/// (2) coincide con una extensión indexable o no tiene extensión
///     (directorios / archivos sin extensión: dejamos pasar, el
///     re-index ya filtra por sus propios criterios).
fn event_is_relevant(event_path: &std::path::Path, out_canonical: &Option<PathBuf>, exts: &[&str]) -> bool {
    if let (Some(o), Some(p)) = (out_canonical, std::fs::canonicalize(event_path).ok()) {
        if p.starts_with(o) {
            return false;
        }
    }
    match event_path.extension().and_then(|x| x.to_str()) {
        Some(ext) => exts.iter().any(|allowed| allowed.eq_ignore_ascii_case(ext)),
        None => true,
    }
}
