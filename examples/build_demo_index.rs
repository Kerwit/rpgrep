//! Constructor de índice programático.
//!
//! Ya no es estrictamente necesario (el binario `rpgrep` cablea el comando
//! `index` directamente), pero se mantiene como ejemplo mínimo de uso
//! programático de `IndexStore::from_dir` y como entry point alternativo
//! para `scripts/bench_compare.sh`.
//!
//! Uso:
//!   cargo run --release --example build_demo_index -- <corpus_dir> <out_dir>

use std::env;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

use rpgrep::index::store::IndexStore;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(anyhow!(
            "uso: cargo run --release --example build_demo_index -- <corpus_dir> <out_dir>"
        ));
    }
    let corpus = Path::new(&args[1]);
    let out_dir = Path::new(&args[2]);

    let t0 = Instant::now();
    let store = IndexStore::from_dir(corpus, &["rs"], 40, 8)
        .with_context(|| format!("construir índice de {}", corpus.display()))?;
    eprintln!(
        "[demo_index] {} chunks, {} archivos en {:.2}s",
        store.chunks.len(),
        store.n_files(),
        t0.elapsed().as_secs_f64()
    );

    store
        .save(out_dir)
        .map_err(|e| anyhow!("IndexStore::save({}): {e}", out_dir.display()))?;

    eprintln!("[demo_index] índice persistido en {}", out_dir.display());
    Ok(())
}
