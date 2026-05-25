//! Orquestación del pipeline probabilístico de búsqueda.
//!
//!   [A] Xor filter pre-screen   → archivos candidatos (cero falsos negativos)
//!   [B] BM25 scoring            → top-N chunks por relevancia probabilística rᵢ
//!   [C] MinHash Jaccard         → matriz de redundancia sᵢⱼ
//!   [D] QUBO + Simulated SA     → selección óptima bajo budget de tokens
//!
//! Cero pesos pre-entrenados, cero red, cero descargas: todo el pipeline
//! corre sobre estructuras matemáticas puras (Xor, BM25, MinHash, Metropolis).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::chunk::Chunk;
use crate::index::minhash::MinHash;
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
}

impl SearchPipeline {
    pub fn load(dir: &Path) -> Result<Self> {
        let store = IndexStore::load(dir)?;
        Ok(Self { store })
    }

    pub fn from_store(store: IndexStore) -> Self {
        Self { store }
    }

    /// Pipeline completo: [A] Xor → [B] BM25 → [C] MinHash → [D] QUBO.
    pub fn search(&self, query: &str, budget: usize, topk: usize) -> Result<Vec<SearchResult>> {
        // [A] Pre-screen con Xor filter sobre archivos.
        let candidate_files = self.store.bloom.candidates(query);
        let candidate_set: HashSet<_> = candidate_files.into_iter().collect();

        // chunk_ids elegibles: si `candidate_set` está vacío (query sin tokens
        // ≥3 chars), conservamos TODOS los chunks → preserva R3.
        let candidate_ids: Vec<u64> = if candidate_set.is_empty() {
            self.store.chunks.iter().map(|c| c.id).collect()
        } else {
            self.store
                .chunks
                .iter()
                .filter(|c| candidate_set.contains(&c.file))
                .map(|c| c.id)
                .collect()
        };

        if candidate_ids.is_empty() {
            return Ok(vec![]);
        }

        // [B] BM25 scoring → top-N por relevancia.
        let topn = self
            .store
            .bm25
            .top_n(query, &candidate_ids, topk.max(10));
        if topn.is_empty() {
            return Ok(vec![]);
        }

        let chunks_by_id: HashMap<u64, &Chunk> =
            self.store.chunks.iter().map(|c| (c.id, c)).collect();

        let filtered: Vec<(&Chunk, f32)> = topn
            .into_iter()
            .filter_map(|(id, score)| chunks_by_id.get(&id).map(|c| (*c, score)))
            .collect();
        if filtered.is_empty() {
            return Ok(vec![]);
        }

        // Normalizar relevancia a [0, 1] dividiendo por el máximo del batch.
        // Imprescindible: BM25 produce scores no acotados; sin normalizar el
        // término λ·sᵢⱼ del QUBO (∈ [0, λ]) se vuelve irrelevante frente a rᵢ.
        let max_score = filtered
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        let relevance: Vec<f32> = filtered.iter().map(|(_, s)| *s / max_score).collect();
        let tokens: Vec<usize> = filtered.iter().map(|(c, _)| c.token_estimate()).collect();

        // [C] Matriz sᵢⱼ vía MinHash Jaccard.
        let similarity = minhash_similarity_matrix(&self.store.minhash, &filtered);

        // [D] QUBO + Simulated Annealing.
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
                score: s / max_score,
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

/// Construye la matriz sᵢⱼ ∈ [0, 1] a partir de las firmas MinHash
/// pre-calculadas. Si alguna firma falta (chunk huérfano), su fila/columna
/// queda en 0 — equivale a "asumir no-redundancia" y deja la decisión al
/// término de relevancia. Es preferible a abortar.
fn minhash_similarity_matrix(
    sigs: &HashMap<u64, MinHash>,
    items: &[(&Chunk, f32)],
) -> Vec<Vec<f32>> {
    let n = items.len();
    let mut sim = vec![vec![0.0_f32; n]; n];
    let resolved: Vec<Option<&MinHash>> = items.iter().map(|(c, _)| sigs.get(&c.id)).collect();

    for i in 0..n {
        let si = match resolved[i] {
            Some(s) => s,
            None => continue,
        };
        for j in (i + 1)..n {
            let sj = match resolved[j] {
                Some(s) => s,
                None => continue,
            };
            let j_sim = si.jaccard(sj);
            sim[i][j] = j_sim;
            sim[j][i] = j_sim;
        }
    }
    sim
}
