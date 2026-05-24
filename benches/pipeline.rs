//! Benchmarks de latencia del pipeline (Capa C).
//!
//! Usan **embeddings sintéticos** (vectores random normalizados a 384 dims
//! = MiniLM L6 v2). NO miden la latencia del `Embedder` (fase ONNX), miden
//! exclusivamente el pipeline post-embedding: Xor → HNSW → QUBO.
//!
//! Justificación: la fase de embedding es la única red-dependent y la única
//! que requiere el modelo ONNX descargado. Aislarla permite mediciones
//! reproducibles y offline. La latencia real end-to-end = bench + tiempo
//! de `Embedder::embed(query)` (~10-30 ms para una query corta en CPU).
//!
//! El gate P95 < 150 ms (BLUEPRINT §1) se verifica en `tests/p95_gate.rs`
//! sobre 100k chunks; aquí Criterion reporta P50/P95/P99 sobre 1k y 10k.

use std::collections::HashSet;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use rpgrep::chunk::Chunk;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::index::hnsw::HnswIndex;
use rpgrep::index::store::IndexStore;
use rpgrep::search::qubo::{QuboProblem, SimulatedAnnealer};

const DIM: usize = 384;
const CHUNKS_PER_FILE: usize = 40;

fn unit_vector(rng: &mut ChaCha8Rng) -> Vec<f32> {
    let raw: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    raw.into_iter().map(|x| x / norm).collect()
}

/// Construye un IndexStore sintético reproducible: `n_chunks` chunks
/// repartidos entre `n_chunks/CHUNKS_PER_FILE` archivos. Embeddings random
/// normalizados. Tokens reales para que el Xor filter sea representativo.
pub fn build_synthetic_store(n_chunks: usize, seed: u64) -> IndexStore {
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
                id: global_idx as u64, // ID denso sintético
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

/// Replica la lógica de `SearchPipeline::search` (pipeline.rs:31-96) PERO
/// recibe el query-vector ya calculado, evitando el `Embedder`. Mide
/// exclusivamente Xor + HNSW + QUBO.
pub fn pipeline_post_embed(
    store: &IndexStore,
    query_text: &str,
    qvec: &[f32],
    budget: usize,
    topk: usize,
) -> usize {
    let candidate_files = store.bloom.candidates(query_text);
    let candidate_set: HashSet<_> = candidate_files.into_iter().collect();

    let hnsw_results = store.hnsw.search(qvec, topk.max(10));
    let chunks_by_id: std::collections::HashMap<u64, &Chunk> =
        store.chunks.iter().map(|c| (c.id, c)).collect();

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

// ---------------------------------------------------------------------------
// Benches Criterion
// ---------------------------------------------------------------------------

fn bench_xor_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_candidates");
    for &n in &[1_000usize, 10_000] {
        let store = build_synthetic_store(n, 0xA1);
        let q = "var_unique_000007 handler_shared";
        group.throughput(Throughput::Elements(store.bloom.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| std::hint::black_box(store.bloom.candidates(std::hint::black_box(q))));
        });
    }
    group.finish();
}

fn bench_hnsw_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_topk");
    let mut rng = ChaCha8Rng::seed_from_u64(0xB2);
    let qvec = unit_vector(&mut rng);
    for &n in &[1_000usize, 10_000] {
        let store = build_synthetic_store(n, 0xB2);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| std::hint::black_box(store.hnsw.search(std::hint::black_box(&qvec), 50)));
        });
    }
    group.finish();
}

fn bench_qubo_anneal(c: &mut Criterion) {
    let mut group = c.benchmark_group("qubo_anneal");
    for &n in &[50usize, 100, 200] {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC3 + n as u64);
        let relevance: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();
        let tokens: Vec<usize> = (0..n).map(|i| 50 + (i * 7) % 80).collect();
        let mut similarity = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let s = rng.gen::<f32>() * 0.3;
                similarity[i][j] = s;
                similarity[j][i] = s;
            }
        }
        let problem = QuboProblem {
            relevance,
            similarity,
            tokens,
            budget: 600,
            lambda: 0.5,
            mu: 0.001,
        };
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let solver = SimulatedAnnealer::default();
            b.iter(|| std::hint::black_box(solver.solve(std::hint::black_box(&problem))));
        });
    }
    group.finish();
}

fn bench_pipeline_post_embed(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_post_embed");
    let mut rng = ChaCha8Rng::seed_from_u64(0xD4);
    let qvec = unit_vector(&mut rng);
    let qtext = "var_unique_000007 handler_shared";

    for &n in &[1_000usize, 10_000] {
        let store = build_synthetic_store(n, 0xD4);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(pipeline_post_embed(
                    std::hint::black_box(&store),
                    qtext,
                    &qvec,
                    4000,
                    50,
                ))
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_xor_candidates, bench_hnsw_topk, bench_qubo_anneal, bench_pipeline_post_embed
}
criterion_main!(benches);
