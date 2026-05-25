//! Benchmarks de latencia del pipeline (Capa C).
//!
//! Pipeline 100% probabilístico: Xor + BM25 + MinHash + QUBO. Cero
//! embeddings, cero red, cero descargas. El bench mide la latencia
//! end-to-end real — no queda ninguna fase neuronal que aislar.
//!
//! El gate P95 < 150 ms (BLUEPRINT §1) se verifica en `tests/p95_gate.rs`
//! sobre 100k chunks; aquí Criterion reporta P50/P95/P99 sobre 1k y 10k
//! para cada etapa por separado y para el pipeline completo.

use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use rpgrep::chunk::Chunk;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::index::bm25::Bm25Index;
use rpgrep::index::minhash::MinHash;
use rpgrep::index::store::{IndexStore, MmappedStore};
use rpgrep::search::qubo::{QuboProblem, SimulatedAnnealer};
use rpgrep::SearchPipeline;

const CHUNKS_PER_FILE: usize = 40;

/// Construye un IndexStore sintético reproducible. Tokens reales
/// (`var_unique_X`, `handler_shared`, `compute_Y_Z`, `filler_R`) para que
/// BM25 y MinHash tengan distribución no trivial.
///
/// SYNCED con `tests/p95_gate.rs::build_synthetic_store`.
pub fn build_synthetic_store(n_chunks: usize, seed: u64) -> IndexStore {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_files = (n_chunks / CHUNKS_PER_FILE).max(1);

    let mut chunks: Vec<Chunk> = Vec::with_capacity(n_chunks);
    let mut bloom = FileBloomIndex::new();

    for f in 0..n_files {
        let file_str = format!("synth/file_{f:06}.rs");
        let mut content = String::new();
        let chunks_in_file = if f == n_files - 1 {
            n_chunks - chunks.len()
        } else {
            CHUNKS_PER_FILE
        };
        for c in 0..chunks_in_file {
            let global_idx = chunks.len();
            let filler: u32 = rng.gen_range(0..10_000);
            let text = format!(
                "fn fn_{global_idx:06}() {{\n    let var_unique_{global_idx:06} = compute_{f:04}_{c:04}();\n    handler_shared(filler_{filler:05});\n}}\n"
            );
            content.push_str(&text);
            chunks.push(Chunk {
                id: global_idx as u64,
                file: file_str.clone(),
                start_line: c * 4 + 1,
                end_line: c * 4 + 4,
                text,
            });
        }
        bloom.add_file(file_str, &content);
    }

    let bm25 = Bm25Index::build(&chunks);
    let minhash: HashMap<u64, MinHash> = chunks
        .iter()
        .map(|c| (c.id, MinHash::from_text(&c.text)))
        .collect();

    IndexStore {
        chunks,
        bloom,
        bm25,
        minhash,
    }
}

// ---------------------------------------------------------------------------
// Benches por etapa
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

fn bench_bm25_topn(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_topn");
    let q = "var_unique_000007 handler_shared compute_0000";
    for &n in &[1_000usize, 10_000] {
        let store = build_synthetic_store(n, 0xB2);
        let all_ids: Vec<u64> = store.chunks.iter().map(|c| c.id).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(store.bm25.top_n(
                    std::hint::black_box(q),
                    std::hint::black_box(&all_ids),
                    50,
                ))
            });
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

fn bench_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load");
    for &n in &[1_000usize, 10_000] {
        let store = build_synthetic_store(n, 0xE5);

        // v0.2 — rkyv + mmap (formato RPGRP002 vía IndexStore::save).
        let dir_rkyv = tempfile::tempdir().expect("tempdir rkyv");
        store.save(dir_rkyv.path()).expect("save rkyv");
        let rkyv_bytes = std::fs::metadata(dir_rkyv.path().join("rpgrep.idx"))
            .expect("rkyv stat")
            .len();

        // v0.1 baseline — bincode uniforme sobre la misma IndexStore.
        // No usa IndexStore::load (magic distinto): replica el load v0.1
        // como `fs::read + bincode::deserialize`.
        let dir_bin = tempfile::tempdir().expect("tempdir bincode");
        let bincode_bytes_buf = bincode::serialize(&store).expect("bincode serialize");
        let bin_path = dir_bin.path().join("rpgrep.idx");
        std::fs::write(&bin_path, &bincode_bytes_buf).expect("bincode write");
        let bincode_bytes = bincode_bytes_buf.len() as u64;

        eprintln!(
            "[bench_load] n={n:>5}  rkyv={:>9} B  bincode={:>9} B  ratio={:.2}x",
            rkyv_bytes,
            bincode_bytes,
            (rkyv_bytes as f64) / (bincode_bytes as f64)
        );

        // v0.2.4 — zero-copy real: MmappedStore.open mapea el archivo
        // pero NO deserializa el payload. La diferencia con `IndexStore::load`
        // mide exactamente el coste del rkyv::from_bytes (que reconstruye
        // HashMaps + Vecs en heap).
        group.bench_with_input(BenchmarkId::new("rkyv_mmap_zerocopy", n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(MmappedStore::open(dir_rkyv.path()).expect("open mmap"))
            });
        });

        group.bench_with_input(BenchmarkId::new("rkyv_v02_owned", n), &n, |b, _| {
            b.iter(|| std::hint::black_box(IndexStore::load(dir_rkyv.path()).expect("load rkyv")));
        });

        group.bench_with_input(BenchmarkId::new("bincode_v01", n), &n, |b, _| {
            b.iter(|| {
                let bytes = std::fs::read(&bin_path).expect("read bincode");
                let s: IndexStore = bincode::deserialize(&bytes).expect("deserialize bincode");
                std::hint::black_box(s)
            });
        });
    }
    group.finish();
}

fn bench_pipeline_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_e2e");
    let qtext = "var_unique_000007 handler_shared compute_0000";
    for &n in &[1_000usize, 10_000] {
        let store = build_synthetic_store(n, 0xD4);
        let pipeline = SearchPipeline::from_store(store);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(pipeline.search(std::hint::black_box(qtext), 4000, 50))
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_xor_candidates, bench_bm25_topn, bench_qubo_anneal, bench_pipeline_e2e, bench_load
}
criterion_main!(benches);
