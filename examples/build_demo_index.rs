//! Constructor de índice de demostración para el harness comparativo (Capa D).
//!
//! Workaround R6: mientras `cli.rs::Commands::Index` no esté cableado
//! (`anyhow::bail!` en cli.rs:70), este example permite que
//! `scripts/bench_compare.sh` disponga de un índice persistido sobre el
//! que rpgrep pueda buscar.
//!
//! Uso:
//!   cargo run --release --example build_demo_index -- <corpus_dir> <out_dir>
//!
//! Requiere red la primera vez (descarga MiniLM L6 v2 ~80 MB).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

use rpgrep::chunk::{chunk_file, Chunk};
use rpgrep::embed::Embedder;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::index::hnsw::HnswIndex;
use rpgrep::index::store::IndexStore;

fn discover_rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                    out.push(p);
                }
            }
        }
    }
    walk(root, &mut out);
    out.sort();
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(anyhow!(
            "uso: cargo run --release --example build_demo_index -- <corpus_dir> <out_dir>"
        ));
    }
    let corpus = Path::new(&args[1]);
    let out_dir = Path::new(&args[2]);

    let files = discover_rust_files(corpus);
    if files.is_empty() {
        return Err(anyhow!("no se encontraron .rs en {}", corpus.display()));
    }

    eprintln!(
        "[demo_index] encontrados {} archivos .rs en {}",
        files.len(),
        corpus.display()
    );

    let mut all_chunks: Vec<Chunk> = Vec::new();
    let mut bloom = FileBloomIndex::new();

    for f in &files {
        let chunks =
            chunk_file(f, 40, 8).with_context(|| format!("chunk_file({})", f.display()))?;
        if !chunks.is_empty() {
            let content = fs::read_to_string(f)?;
            bloom.add_file(f.clone(), &content);
            all_chunks.extend(chunks);
        }
    }
    eprintln!(
        "[demo_index] {} chunks generados; iniciando Embedder…",
        all_chunks.len()
    );

    let t_embed = Instant::now();
    let embedder = Embedder::new().context("Embedder::new() falló (¿red disponible?)")?;
    let texts: Vec<String> = all_chunks.iter().map(|c| c.text.clone()).collect();
    let vectors = embedder.embed_batch(texts).context("embed_batch falló")?;
    eprintln!(
        "[demo_index] embeddings listos en {:.2}s",
        t_embed.elapsed().as_secs_f64()
    );

    let ids: Vec<u64> = all_chunks.iter().map(|c| c.id).collect();
    let hnsw = HnswIndex::build(vectors, ids);

    let store = IndexStore {
        chunks: all_chunks,
        bloom,
        hnsw,
    };
    store
        .save(out_dir)
        .map_err(|e| anyhow!("IndexStore::save({}): {e}", out_dir.display()))?;

    eprintln!("[demo_index] índice persistido en {}", out_dir.display());
    Ok(())
}
