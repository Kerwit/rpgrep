//! Gate de latencia P95 (Capa C, complemento al bench Criterion).
//!
//! BLUEPRINT §1 establece P95 < 150 ms end-to-end sobre 100k chunks.
//! Este test mide 50 queries sobre un corpus sintético de 100k chunks
//! y FALLA si el P95 supera el umbral. Marcado `#[ignore]` porque
//! construir 100k chunks tarda decenas de segundos: se activa
//! explícitamente con
//!   cargo test --release --test p95_gate -- --ignored --nocapture
//!
//! IMPORTANTE: `--release` es obligatorio. En modo debug, BM25, MinHash y
//! simulated annealing son órdenes de magnitud más lentos, saturan CPU
//! durante minutos y el gate P95 < 150 ms pierde su sentido (mide tiempos
//! que no representan producción).
//!
//! Sin embeddings: el pipeline es 100% probabilístico (Xor + BM25 +
//! MinHash + QUBO). El test mide la latencia end-to-end real — no queda
//! ninguna fase "post-embed" que aislar.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use rpgrep::chunk::Chunk;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::index::bm25::Bm25Index;
use rpgrep::index::minhash::MinHash;
use rpgrep::index::store::IndexStore;
use rpgrep::SearchPipeline;

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

pub fn build_synthetic_store(n_chunks: usize, seed: u64) -> IndexStore {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_files = (n_chunks / CHUNKS_PER_FILE).max(1);

    let mut chunks: Vec<Chunk> = Vec::with_capacity(n_chunks);
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
            // Token raro `var_unique_X` + token compartido + entropía controlada
            // por el RNG (filler_R) para que BM25 tenga distribución no trivial.
            let filler: u32 = rng.gen_range(0..10_000);
            let text = format!(
                "fn fn_{global_idx:06}() {{\n    let var_unique_{global_idx:06} = compute_{f:04}_{c:04}();\n    handler_shared(filler_{filler:05});\n}}\n"
            );
            content.push_str(&text);
            chunks.push(Chunk {
                id: global_idx as u64,
                file: path.clone(),
                start_line: c * 4 + 1,
                end_line: c * 4 + 4,
                text,
            });
        }
        bloom.add_file(path, &content);
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

fn percentile(samples: &mut [Duration], p: f64) -> Duration {
    samples.sort();
    let idx = ((samples.len() as f64) * p).ceil() as usize - 1;
    samples[idx.min(samples.len() - 1)]
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "construye corpus de 100k chunks; lento (~30-60s en --release; minutos en debug). Ejecutar: cargo test --release --test p95_gate -- --ignored"]
fn pipeline_p95_under_150ms_at_100k_chunks() {
    eprintln!("[p95_gate] construyendo corpus sintético de {CORPUS_SIZE} chunks…");
    let t0 = Instant::now();
    let store = build_synthetic_store(CORPUS_SIZE, 0xE5);
    eprintln!(
        "[p95_gate] corpus listo en {:.2}s ({} chunks, {} archivos)",
        t0.elapsed().as_secs_f64(),
        store.chunks.len(),
        store.bloom.len()
    );

    let pipeline = SearchPipeline::from_store(store);

    let queries: Vec<String> = (0..QUERIES)
        .map(|i| format!("var_unique_{:06} handler_shared compute_{:04}", i * 7, i))
        .collect();

    // Warm-up (no cuenta): primeras 3 queries calientan caches.
    for q in queries.iter().take(3) {
        let _ = pipeline.search(q, 4000, 50).expect("search warm-up");
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(queries.len());
    for q in &queries {
        let t = Instant::now();
        let _ = std::hint::black_box(pipeline.search(q, 4000, 50).expect("search"));
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
