//! Tests de calidad semántica (Capa B).
//!
//! Todos los tests aquí están marcados `#[ignore]` porque requieren:
//!   1. Descargar el modelo ONNX MiniLM L6 v2 (~80 MB, primera vez).
//!   2. Indexar `src/` del propio repositorio con embeddings reales.
//!
//! Ejecución:
//!   cargo test --test semantic_quality -- --ignored --nocapture
//!
//! Las métricas (Recall@5, MRR, Diversity@5) se IMPRIMEN siempre por stderr;
//! los asserts solo verifican un **piso laxo** que detecta regresión brutal,
//! no rendimiento absoluto. Los valores objetivo se documentan tras correr,
//! en `docs/VALIDATION.md` §5.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use rpgrep::chunk::{chunk_file, Chunk};
use rpgrep::embed::Embedder;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::index::hnsw::HnswIndex;
use rpgrep::index::store::IndexStore;
use rpgrep::{SearchPipeline, SearchResult};

const GOLDEN_PATH: &str = "tests/fixtures/golden_corpus.tsv";
const CORPUS_ROOT: &str = "src";
const TOP_K: usize = 5;
const BUDGET: usize = 4000;

// Pisos laxos: el test solo falla si el sistema está completamente roto.
// Los valores reales se reportan por stderr y se documentan en VALIDATION.md.
const RECALL_FLOOR: f32 = 0.30;
const MRR_FLOOR: f32 = 0.20;
const DIVERSITY_FLOOR: f32 = 0.40;

#[derive(Debug)]
struct GoldenPair {
    query: String,
    expected_substrings: Vec<String>,
}

fn load_golden(path: &Path) -> Vec<GoldenPair> {
    let content = fs::read_to_string(path).expect("leer corpus dorado");
    content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                return None;
            }
            let query = parts[0].trim().to_string();
            let subs: Vec<String> = parts[1]
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            if query.is_empty() || subs.is_empty() {
                return None;
            }
            Some(GoldenPair {
                query,
                expected_substrings: subs,
            })
        })
        .collect()
}

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

fn build_index_real(root: &Path) -> IndexStore {
    let files = discover_rust_files(root);
    assert!(
        !files.is_empty(),
        "no se encontraron .rs en {}",
        root.display()
    );

    let mut all_chunks: Vec<Chunk> = Vec::new();
    let mut bloom = FileBloomIndex::new();

    for f in &files {
        let chunks = chunk_file(f, 40, 8).expect("chunk_file falló");
        if !chunks.is_empty() {
            let content = fs::read_to_string(f).expect("read_to_string falló");
            bloom.add_file(f.clone(), &content);
            all_chunks.extend(chunks);
        }
    }

    eprintln!(
        "[semantic_quality] indexando {} archivos, {} chunks; descargando/iniciando MiniLM…",
        files.len(),
        all_chunks.len()
    );

    let embedder = Embedder::new().expect("init Embedder (requiere red la primera vez)");
    let texts: Vec<String> = all_chunks.iter().map(|c| c.text.clone()).collect();
    let vectors = embedder.embed_batch(texts).expect("embed_batch falló");
    let ids: Vec<u64> = all_chunks.iter().map(|c| c.id).collect();
    let hnsw = HnswIndex::build(vectors, ids);

    IndexStore {
        chunks: all_chunks,
        bloom,
        hnsw,
    }
}

fn is_relevant(chunk: &Chunk, expected: &[String]) -> bool {
    let text = chunk.text.to_lowercase();
    expected.iter().any(|needle| text.contains(needle.as_str()))
}

fn diversity_at_k(results: &[SearchResult]) -> f32 {
    let n = results.len();
    if n < 2 {
        return 1.0;
    }
    let trigrams: Vec<HashSet<[u8; 3]>> = results
        .iter()
        .map(|r| {
            let bytes = r.chunk.text.as_bytes();
            let mut s = HashSet::new();
            if bytes.len() >= 3 {
                for w in bytes.windows(3) {
                    s.insert([w[0], w[1], w[2]]);
                }
            }
            s
        })
        .collect();

    let mut sum_sim = 0.0_f32;
    let mut pairs = 0_usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let inter = trigrams[i].intersection(&trigrams[j]).count() as f32;
            let union = trigrams[i].union(&trigrams[j]).count().max(1) as f32;
            sum_sim += inter / union;
            pairs += 1;
        }
    }
    let mean_sim = sum_sim / pairs.max(1) as f32;
    1.0 - mean_sim
}

#[test]
#[ignore = "requiere descarga del modelo ONNX (~80MB); ejecutar con --ignored"]
fn recall_mrr_diversity_above_floor_on_golden_corpus() {
    let golden = load_golden(Path::new(GOLDEN_PATH));
    assert!(
        golden.len() >= 20,
        "corpus dorado debe tener ≥20 pares; encontrados {}",
        golden.len()
    );

    let store = build_index_real(Path::new(CORPUS_ROOT));
    let chunks_snapshot: Vec<Chunk> = store.chunks.clone();

    let tmp = tempfile::tempdir().expect("tempdir");
    store.save(tmp.path()).expect("guardar IndexStore");
    let pipeline = SearchPipeline::load(tmp.path()).expect("cargar SearchPipeline");

    let relevant_per_query: Vec<HashSet<u64>> = golden
        .iter()
        .map(|g| {
            chunks_snapshot
                .iter()
                .filter(|c| is_relevant(c, &g.expected_substrings))
                .map(|c| c.id)
                .collect()
        })
        .collect();

    let mut recall_sum = 0.0_f32;
    let mut mrr_sum = 0.0_f32;
    let mut diversity_sum = 0.0_f32;
    let mut evaluated = 0_usize;
    let mut skipped_no_relevant = 0_usize;

    eprintln!(
        "\n[semantic_quality] {:<55} {:>8} {:>8} {:>8} {:>6}",
        "query", "Recall@5", "MRR", "Div@5", "|R|"
    );
    eprintln!("{}", "-".repeat(95));

    for (g, relevant) in golden.iter().zip(relevant_per_query.iter()) {
        if relevant.is_empty() {
            skipped_no_relevant += 1;
            eprintln!(
                "[skip] {:<55} (sin chunks relevantes en corpus)",
                truncate(&g.query, 55)
            );
            continue;
        }

        let results = pipeline
            .search(&g.query, BUDGET, 50)
            .expect("pipeline.search falló");

        let top_k: Vec<&SearchResult> = results.iter().take(TOP_K).collect();
        let top_ids: HashSet<u64> = top_k.iter().map(|r| r.chunk.id).collect();

        let hit = top_ids.intersection(relevant).count() as f32;
        let recall = hit / relevant.len() as f32;

        let mrr = top_k
            .iter()
            .enumerate()
            .find_map(|(i, r)| {
                if relevant.contains(&r.chunk.id) {
                    Some(1.0 / (i as f32 + 1.0))
                } else {
                    None
                }
            })
            .unwrap_or(0.0);

        let div = diversity_at_k(
            &top_k
                .iter()
                .map(|r| (*r).clone())
                .collect::<Vec<SearchResult>>(),
        );

        eprintln!(
            "       {:<55} {:>8.3} {:>8.3} {:>8.3} {:>6}",
            truncate(&g.query, 55),
            recall,
            mrr,
            div,
            relevant.len()
        );

        recall_sum += recall;
        mrr_sum += mrr;
        diversity_sum += div;
        evaluated += 1;
    }

    assert!(
        evaluated >= 15,
        "muy pocas queries con relevantes ({evaluated}); revisa el corpus dorado"
    );

    let recall_mean = recall_sum / evaluated as f32;
    let mrr_mean = mrr_sum / evaluated as f32;
    let div_mean = diversity_sum / evaluated as f32;

    eprintln!("{}", "-".repeat(95));
    eprintln!(
        "[semantic_quality] MEDIAS  Recall@5={recall_mean:.3}  MRR={mrr_mean:.3}  Diversity@5={div_mean:.3}  evaluadas={evaluated}  skipped={skipped_no_relevant}"
    );

    assert!(
        recall_mean >= RECALL_FLOOR,
        "Recall@5 medio {recall_mean:.3} < piso {RECALL_FLOOR}"
    );
    assert!(
        mrr_mean >= MRR_FLOOR,
        "MRR medio {mrr_mean:.3} < piso {MRR_FLOOR}"
    );
    assert!(
        div_mean >= DIVERSITY_FLOOR,
        "Diversity@5 medio {div_mean:.3} < piso {DIVERSITY_FLOOR}"
    );
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}
