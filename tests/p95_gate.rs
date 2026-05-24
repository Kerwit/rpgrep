//! Gate de latencia P95 (Capa C, complemento al bench Criterion).
//!
//! BLUEPRINT §1 establece P95 < 150 ms end-to-end sobre 100k chunks.
//! Este test mide 50 queries sobre un corpus sintético de 100k chunks
//! y FALLA si el P95 supera el umbral. Marcado `#[ignore]` porque
//! construir 100k chunks (HNSW build + IndexStore) tarda decenas de
//! segundos: se activa explícitamente en CI con
//!   cargo test --test p95_gate -- --ignored --nocapture
//!
//! Embeddings sintéticos (vectores random 384-dim normalizados). Mide
//! exclusivamente Xor + HNSW + QUBO, igual que el bench Criterion. La
//! latencia real end-to-end suma además el embedding de la query
//! (~10-30 ms en CPU para una query corta vía MiniLM L6 v2).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use rpgrep::chunk::Chunk;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::index::hnsw::HnswIndex;
use rpgrep::index::store::IndexStore;
use rpgrep::search::qubo::{QuboProblem, SimulatedAnnealer};

const DIM: usize = 384;
const CHUNKS_PER_FILE: usize = 40;
const CORPUS_SIZE: usize = 100_000;
const QUERIES: usize = 50;
const P95_BUDGET_MS: u128 = 150;

// ---------------------------------------------------------------------------
// SYNCED con benches/pipeline.rs — mantén ambas versiones en sincronía.
// Duplicado intencionalmente: los integration tests no pueden importar
// módulos desde `benches/` ni viceversa, y exportar estas helpers desde
// `src/` contaminaría el código de producción.
// ---------------------------------------------------------------------------

fn unit_vector(rng: &mut ChaCha8Rng) -> Vec<f32> {
    let raw: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    raw.into_iter().map(|x| x / norm).collect()
}

fn build_synthetic_store(n_chunks: usize, seed: u64) -> IndexStore {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_files = (n_chunks / CHUNKS_PER_FILE).max(1);

    let mut chunks: Vec<Chunk> = Vec::with_capacity(n_chunks);
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n_chunks);
    let mut ids: Vec<u64> = Vec::with_capacity(n_chunks);
    let mut bloom = FileBloomIndex::new();

    for f in 0..n_files {
        let path = PathBuf::from(format!("synth/file_{f:06}.rs"));
        let mut content = String::new();
        let chunks_in_file = if f == n_files - 1 {
            n_chunks - chunks.len()
        } else {
            CHUNKS_PER_FILE
        };
        for c in 0..chunks_in_file {
            let global_idx = chunks.len();
            let text = format!(
                "fn fn_{global_idx:06}() {{\n    let var_unique_{global_idx:06} = compute_{f:04}_{c:04}();\n    handler_shared();\n}}\n"
            );
            content.push_str(&text);
            chunks.push(Chunk {
                id: global_idx as u64,
                file: path.clone(),
                start_line: c * 4 + 1,
                end_line: c * 4 + 4,
                text,
            });
            vectors.push(unit_vector(&mut rng));
            ids.push(global_idx as u64);
        }
        bloom.add_file(path, &content);
    }

    let hnsw = HnswIndex::build(vectors, ids);
    IndexStore {
        chunks,
        bloom,
        hnsw,
    }
}

fn pipeline_post_embed(
    store: &IndexStore,
    query_text: &str,
    qvec: &[f32],
    budget: usize,
    topk: usize,
) -> usize {
    let candidate_files = store.bloom.candidates(query_text);
    let candidate_set: HashSet<_> = candidate_files.into_iter().collect();

    let hnsw_results = store.hnsw.search(qvec, topk.max(10));
    let chunks_by_id: HashMap<u64, &Chunk> = store.chunks.iter().map(|c| (c.id, c)).collect();

    let filtered: Vec<(&Chunk, f32)> = hnsw_results
        .into_iter()
        .filter_map(|(id, dist)| {
            let c = *chunks_by_id.get(&id)?;
            let pass = candidate_set.is_empty() || candidate_set.contains(&c.file);
            if pass {
                Some((c, 1.0 - dist))
            } else {
                None
            }
        })
        .collect();

    if filtered.is_empty() {
        return 0;
    }

    let relevance: Vec<f32> = filtered.iter().map(|(_, s)| *s).collect();
    let tokens: Vec<usize> = filtered.iter().map(|(c, _)| c.token_estimate()).collect();
    let similarity = vec![vec![0.0_f32; filtered.len()]; filtered.len()];

    let problem = QuboProblem {
        relevance,
        similarity,
        tokens,
        budget,
        lambda: 0.5,
        mu: 0.001,
    };
    let solver = SimulatedAnnealer::default();
    let assignment = solver.solve(&problem);
    assignment.iter().filter(|x| **x).count()
}

fn percentile(samples: &mut [Duration], p: f64) -> Duration {
    samples.sort();
    let idx = ((samples.len() as f64) * p).ceil() as usize - 1;
    samples[idx.min(samples.len() - 1)]
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "construye corpus de 100k chunks; lento (~30-60s). Ejecutar con --ignored en CI"]
fn pipeline_post_embed_p95_under_150ms_at_100k_chunks() {
    eprintln!("[p95_gate] construyendo corpus sintético de {CORPUS_SIZE} chunks…");
    let t0 = Instant::now();
    let store = build_synthetic_store(CORPUS_SIZE, 0xE5);
    eprintln!(
        "[p95_gate] corpus listo en {:.2}s ({} chunks, {} archivos)",
        t0.elapsed().as_secs_f64(),
        store.chunks.len(),
        store.bloom.len()
    );

    let mut rng = ChaCha8Rng::seed_from_u64(0xF6);
    let queries: Vec<(String, Vec<f32>)> = (0..QUERIES)
        .map(|i| {
            let qtext = format!("var_unique_{:06} handler_shared compute_{:04}", i * 7, i);
            (qtext, unit_vector(&mut rng))
        })
        .collect();

    // Warm-up (no contar): primeras 3 queries calientan caches.
    for (qt, qv) in queries.iter().take(3) {
        let _ = pipeline_post_embed(&store, qt, qv, 4000, 50);
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(queries.len());
    for (qt, qv) in &queries {
        let t = Instant::now();
        let _ = std::hint::black_box(pipeline_post_embed(&store, qt, qv, 4000, 50));
        samples.push(t.elapsed());
    }

    let p50 = percentile(&mut samples.clone(), 0.50);
    let p95 = percentile(&mut samples.clone(), 0.95);
    let p99 = percentile(&mut samples.clone(), 0.99);

    eprintln!(
        "[p95_gate] {} queries — P50={:>6.1}ms  P95={:>6.1}ms  P99={:>6.1}ms  (gate: P95 < {}ms)",
        samples.len(),
        p50.as_secs_f64() * 1000.0,
        p95.as_secs_f64() * 1000.0,
        p99.as_secs_f64() * 1000.0,
        P95_BUDGET_MS
    );

    assert!(
        p95.as_millis() < P95_BUDGET_MS,
        "P95 = {} ms > gate {} ms — regresión de latencia",
        p95.as_millis(),
        P95_BUDGET_MS
    );
}
