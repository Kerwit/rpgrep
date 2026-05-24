//! Orquestación del pipeline probabilístico de búsqueda.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::chunk::Chunk;
use crate::embed::Embedder;
use crate::index::store::IndexStore;
use crate::search::qubo::{QuboProblem, SimulatedAnnealer};
use crate::Result;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
}

pub struct SearchPipeline {
    store: IndexStore,
    embedder: Embedder,
}

impl SearchPipeline {
    pub fn load(dir: &Path) -> Result<Self> {
        let store = IndexStore::load(dir)?;
        let embedder = Embedder::new()?;
        Ok(Self { store, embedder })
    }

    /// Pipeline completo: [A] Xor → [B] HNSW → [D] QUBO.
    pub fn search(&self, query: &str, budget: usize, topk: usize) -> Result<Vec<SearchResult>> {
        // [A] Pre-screen con Xor filter
        let candidate_files = self.store.bloom.candidates(query);
        let candidate_set: HashSet<_> = candidate_files.into_iter().collect();

        // [B] HNSW retrieval
        let qvec = self.embedder.embed(query)?;
        let hnsw_results = self.store.hnsw.search(&qvec, topk.max(10));

        let chunks_by_id: HashMap<u64, &Chunk> =
            self.store.chunks.iter().map(|c| (c.id, c)).collect();

        // Intersección: chunks que están en HNSW top-K Y cuyo archivo pasó el Xor filter.
        // (Si candidate_set está vacío, conservamos todos los del HNSW.)
        let filtered: Vec<(&Chunk, f32)> = hnsw_results
            .into_iter()
            .filter_map(|(id, dist)| {
                let c = *chunks_by_id.get(&id)?;
                let pass = candidate_set.is_empty() || candidate_set.contains(&c.file);
                if pass {
                    // similitud coseno ≈ 1 - distancia
                    Some((c, 1.0 - dist))
                } else {
                    None
                }
            })
            .collect();

        if filtered.is_empty() {
            return Ok(vec![]);
        }

        // [D] QUBO + Simulated Annealing
        let relevance: Vec<f32> = filtered.iter().map(|(_, s)| *s).collect();
        let tokens: Vec<usize> = filtered.iter().map(|(c, _)| c.token_estimate()).collect();
        let similarity = trigram_similarity_matrix(&filtered);

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

        let mut results: Vec<SearchResult> = filtered
            .into_iter()
            .enumerate()
            .filter(|(i, _)| assignment[*i])
            .map(|(_, (c, s))| SearchResult {
                chunk: c.clone(),
                score: s,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }
}

/// Matriz de similitud por Jaccard de trigramas.
/// Es un proxy barato para la redundancia entre chunks.
///
/// v0.2: sustituir por similitud coseno entre los embeddings persistidos
///       (ya disponibles, eliminamos esta aproximación).
fn trigram_similarity_matrix(items: &[(&Chunk, f32)]) -> Vec<Vec<f32>> {
    let n = items.len();
    let mut sim = vec![vec![0.0_f32; n]; n];
    let sets: Vec<HashSet<[u8; 3]>> = items.iter().map(|(c, _)| trigram_set(&c.text)).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let inter = sets[i].intersection(&sets[j]).count() as f32;
            let union = sets[i].union(&sets[j]).count().max(1) as f32;
            let j_sim = inter / union;
            sim[i][j] = j_sim;
            sim[j][i] = j_sim;
        }
    }
    sim
}

fn trigram_set(text: &str) -> HashSet<[u8; 3]> {
    let bytes = text.as_bytes();
    let mut s = HashSet::new();
    if bytes.len() < 3 {
        return s;
    }
    for w in bytes.windows(3) {
        s.insert([w[0], w[1], w[2]]);
    }
    s
}
