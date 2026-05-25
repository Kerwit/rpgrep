//! BM25 (Robertson & Walker 1994) — sustituto probabilístico de embeddings densos.
//!
//! Provee `rᵢ` (relevancia chunk↔query) del QUBO sin pesos pre-entrenados.
//! Fundamento: modelo probabilístico de relevancia con saturación de TF y
//! normalización por longitud de documento.
//!
//!   score(q, d) = Σ_{t ∈ q} IDF(t) · ( tf(t,d) · (k1+1) )
//!                            / ( tf(t,d) + k1 · (1 - b + b · |d|/avgdl) )
//!
//!   IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )   (variante Lucene)
//!
//! Parámetros canónicos: k1 = 1.5 (saturación de TF), b = 0.75 (peso de
//! normalización por longitud). Tokenización ≥3 chars, lowercase, hash a u64
//! — coherente con `bloom::unique_token_hashes` para que el filtro Xor y
//! BM25 hablen del mismo vocabulario.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::chunk::Chunk;

const K1: f32 = 1.5;
const B: f32 = 0.75;
const MIN_TOKEN_LEN: usize = 3;

#[derive(Default, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Bm25Index {
    /// chunk_id → (token_hash → frecuencia de ese token en el chunk).
    pub term_freq: HashMap<u64, HashMap<u64, u32>>,
    /// token_hash → número de chunks distintos que contienen el token.
    pub doc_freq: HashMap<u64, u32>,
    /// chunk_id → longitud del chunk en tokens (con repeticiones).
    pub doc_len: HashMap<u64, u32>,
    pub avg_doc_len: f32,
    pub n_docs: u32,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construye el índice de una sola pasada y calcula `avg_doc_len`.
    pub fn build(chunks: &[Chunk]) -> Self {
        let mut idx = Self::default();
        for c in chunks {
            idx.add_chunk(c);
        }
        idx.finalize();
        idx
    }

    /// Añade un chunk. Llamar `finalize()` al terminar de añadir todos.
    pub fn add_chunk(&mut self, chunk: &Chunk) {
        let mut tf: HashMap<u64, u32> = HashMap::new();
        let mut total = 0_u32;
        for token in tokenize(&chunk.text) {
            *tf.entry(token).or_insert(0) += 1;
            total += 1;
        }
        if total == 0 {
            return;
        }
        for token in tf.keys() {
            *self.doc_freq.entry(*token).or_insert(0) += 1;
        }
        self.term_freq.insert(chunk.id, tf);
        self.doc_len.insert(chunk.id, total);
        self.n_docs += 1;
    }

    /// Calcula `avg_doc_len`. Idempotente.
    pub fn finalize(&mut self) {
        if self.n_docs == 0 {
            self.avg_doc_len = 0.0;
            return;
        }
        let total: u64 = self.doc_len.values().map(|&v| v as u64).sum();
        self.avg_doc_len = total as f32 / self.n_docs as f32;
    }

    /// Score BM25 del chunk para la query. 0 si el chunk no está indexado
    /// o si ningún token de la query aparece en el chunk.
    pub fn score(&self, query: &str, chunk_id: u64) -> f32 {
        let dl = match self.doc_len.get(&chunk_id) {
            Some(&v) => v as f32,
            None => return 0.0,
        };
        let tf_map = match self.term_freq.get(&chunk_id) {
            Some(m) => m,
            None => return 0.0,
        };
        let avgdl = self.avg_doc_len.max(1.0);
        let norm = 1.0 - B + B * (dl / avgdl);

        let mut score = 0.0_f32;
        let mut seen = std::collections::HashSet::new();
        for token in tokenize(query) {
            if !seen.insert(token) {
                continue;
            }
            let tf = match tf_map.get(&token) {
                Some(&v) => v as f32,
                None => continue,
            };
            let df = self.doc_freq.get(&token).copied().unwrap_or(0) as f32;
            let idf = ((self.n_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            score += idf * (tf * (K1 + 1.0)) / (tf + K1 * norm);
        }
        score
    }

    /// Top-N chunks por score BM25.
    ///
    /// Si `candidates` está vacío, escanea **todos** los chunks indexados
    /// — esto preserva la garantía R3 del pipeline: pre-screen vacío no
    /// debe interpretarse como "filtra todo".
    pub fn top_n(&self, query: &str, candidates: &[u64], n: usize) -> Vec<(u64, f32)> {
        if self.n_docs == 0 || n == 0 {
            return Vec::new();
        }
        let q_tokens: Vec<u64> = {
            let mut seen = std::collections::HashSet::new();
            tokenize(query).filter(|t| seen.insert(*t)).collect()
        };
        if q_tokens.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(u64, f32)> = if candidates.is_empty() {
            self.doc_len
                .keys()
                .filter_map(|&id| {
                    let s = self.score(query, id);
                    if s > 0.0 {
                        Some((id, s))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            candidates
                .iter()
                .filter_map(|&id| {
                    let s = self.score(query, id);
                    if s > 0.0 {
                        Some((id, s))
                    } else {
                        None
                    }
                })
                .collect()
        };

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }
}

/// Score BM25 del chunk para la query, operando sobre `&ArchivedBm25Index`
/// (zero-copy: los HashMap viven en mmap).
///
/// Réplica exacta de la matemática de `Bm25Index::score`. La diferencia
/// está en la indirección: en lugar de `HashMap<u64, ...>`, consultamos
/// `ArchivedHashMap<Archived<u64>, ...>` con claves `u64_le::from_native`.
pub fn score_archived(arch: &ArchivedBm25Index, query: &str, chunk_id: u64) -> f32 {
    use rkyv::rend::u64_le;
    let key = u64_le::from_native(chunk_id);

    let dl = match arch.doc_len.get(&key) {
        Some(v) => v.to_native() as f32,
        None => return 0.0,
    };
    let tf_map = match arch.term_freq.get(&key) {
        Some(m) => m,
        None => return 0.0,
    };
    let avgdl = arch.avg_doc_len.to_native().max(1.0);
    let norm = 1.0 - B + B * (dl / avgdl);
    let n_docs = arch.n_docs.to_native() as f32;

    let mut score = 0.0_f32;
    let mut seen = std::collections::HashSet::new();
    for token in tokenize(query) {
        if !seen.insert(token) {
            continue;
        }
        let token_key = u64_le::from_native(token);
        let tf = match tf_map.get(&token_key) {
            Some(v) => v.to_native() as f32,
            None => continue,
        };
        let df = arch
            .doc_freq
            .get(&token_key)
            .map(|v| v.to_native())
            .unwrap_or(0) as f32;
        let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
        score += idf * (tf * (K1 + 1.0)) / (tf + K1 * norm);
    }
    score
}

/// Top-N chunks por score BM25 sobre el archived. Misma semántica que
/// `Bm25Index::top_n`: si `candidates` está vacío, escanea todos los
/// `doc_len` (preserva R3).
pub fn top_n_archived(
    arch: &ArchivedBm25Index,
    query: &str,
    candidates: &[u64],
    n: usize,
) -> Vec<(u64, f32)> {
    use rkyv::rend::u64_le;

    if arch.n_docs.to_native() == 0 || n == 0 {
        return Vec::new();
    }
    let q_tokens: Vec<u64> = {
        let mut seen = std::collections::HashSet::new();
        tokenize(query).filter(|t| seen.insert(*t)).collect()
    };
    if q_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<(u64, f32)> = if candidates.is_empty() {
        arch.doc_len
            .keys()
            .filter_map(|k| {
                let id = k.to_native();
                let s = score_archived(arch, query, id);
                if s > 0.0 {
                    Some((id, s))
                } else {
                    None
                }
            })
            .collect()
    } else {
        candidates
            .iter()
            .filter_map(|&id| {
                // Verifica que el chunk existe en el archived antes de scorear:
                // evita malgastar tokenize→lookup cuando candidates contiene
                // ids que no están en el índice (defensa por construcción).
                if arch.doc_len.get(&u64_le::from_native(id)).is_none() {
                    return None;
                }
                let s = score_archived(arch, query, id);
                if s > 0.0 {
                    Some((id, s))
                } else {
                    None
                }
            })
            .collect()
    };

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
}

/// Tokenizador compartido: separa por chars no `[A-Za-z0-9_]`, filtra
/// tokens < MIN_TOKEN_LEN, lowercase, hash a u64 con `DefaultHasher`.
/// **Devuelve duplicados** — BM25 los necesita para `tf`.
pub(crate) fn tokenize(text: &str) -> impl Iterator<Item = u64> + '_ {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= MIN_TOKEN_LEN)
        .map(|t| {
            let mut h = DefaultHasher::new();
            t.to_lowercase().hash(&mut h);
            h.finish()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_chunk(id: u64, text: &str) -> Chunk {
        Chunk {
            id,
            file: format!("c{id}.rs"),
            start_line: 1,
            end_line: 1,
            text: text.to_string(),
        }
    }

    #[test]
    fn empty_index_returns_zero_and_no_topn() {
        let idx = Bm25Index::new();
        assert_eq!(idx.score("anything", 42), 0.0);
        assert!(idx.top_n("anything", &[], 10).is_empty());
    }

    #[test]
    fn score_zero_when_no_query_term_in_chunk() {
        let idx = Bm25Index::build(&[
            make_chunk(1, "validate user input checker"),
            make_chunk(2, "render html template engine"),
        ]);
        assert_eq!(idx.score("rocket science", 1), 0.0);
    }

    #[test]
    fn score_positive_when_token_overlaps() {
        let idx = Bm25Index::build(&[
            make_chunk(1, "validate user input checker"),
            make_chunk(2, "render html template engine"),
        ]);
        assert!(idx.score("validate input", 1) > 0.0);
        assert_eq!(idx.score("validate input", 2), 0.0);
    }

    #[test]
    fn idf_rewards_rare_terms_over_common_ones() {
        // "common" aparece en TODOS los chunks; "rare" en uno solo.
        // Score para una query con sólo "rare" debe superar al de "common".
        let chunks: Vec<Chunk> = (0..10)
            .map(|i| {
                if i == 0 {
                    make_chunk(0, "rare common common common")
                } else {
                    make_chunk(i as u64, "common common common")
                }
            })
            .collect();
        let idx = Bm25Index::build(&chunks);
        let s_rare = idx.score("rare", 0);
        let s_common = idx.score("common", 0);
        assert!(
            s_rare > s_common,
            "IDF debe pesar más el término raro: rare={s_rare} vs common={s_common}"
        );
    }

    #[test]
    fn longer_doc_penalized_for_same_absolute_tf() {
        // dos chunks con el mismo TF absoluto pero distinta longitud:
        // el más largo debe puntuar menos por normalización b·|d|/avgdl.
        let short = make_chunk(1, "validate validate validate");
        let mut long_text = String::from("validate validate validate ");
        for i in 0..40 {
            long_text.push_str(&format!("filler_{i} "));
        }
        let long = make_chunk(2, &long_text);
        let idx = Bm25Index::build(&[short, long]);

        let s_short = idx.score("validate", 1);
        let s_long = idx.score("validate", 2);
        assert!(
            s_short > s_long,
            "normalización por longitud debe favorecer al chunk corto"
        );
    }

    #[test]
    fn top_n_sorted_descending_and_truncated() {
        let chunks: Vec<Chunk> = vec![
            make_chunk(10, "alpha beta gamma"),
            make_chunk(20, "alpha alpha beta"),
            make_chunk(30, "alpha alpha alpha"),
            make_chunk(40, "delta epsilon"),
        ];
        let idx = Bm25Index::build(&chunks);
        let top = idx.top_n("alpha", &[], 2);
        assert_eq!(top.len(), 2);
        assert!(top[0].1 >= top[1].1, "no ordenado descendente");
        // 30 tiene la mayor TF de "alpha"; debería estar primero.
        assert_eq!(top[0].0, 30);
    }

    #[test]
    fn top_n_respects_candidate_filter() {
        let idx = Bm25Index::build(&[
            make_chunk(1, "alpha alpha"),
            make_chunk(2, "alpha"),
            make_chunk(3, "alpha alpha alpha"),
        ]);
        // Restringimos al chunk con menor score; debe ganar a falta de competidores.
        let top = idx.top_n("alpha", &[2], 10);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 2);
    }

    #[test]
    fn archived_top_n_matches_owned() {
        use rkyv::rancor::Error as RkyvError;

        let chunks: Vec<Chunk> = vec![
            make_chunk(10, "alpha beta gamma"),
            make_chunk(20, "alpha alpha beta"),
            make_chunk(30, "alpha alpha alpha"),
            make_chunk(40, "delta epsilon"),
        ];
        let idx = Bm25Index::build(&chunks);
        let bytes = rkyv::to_bytes::<RkyvError>(&idx).unwrap();
        let arch = rkyv::access::<ArchivedBm25Index, RkyvError>(&bytes).unwrap();

        // 1) candidates vacío (path "todos los doc_len.keys()").
        let owned_empty = idx.top_n("alpha", &[], 3);
        let arch_empty = top_n_archived(arch, "alpha", &[], 3);
        assert_eq!(owned_empty.len(), arch_empty.len(), "len empty mismatch");
        for (a, b) in owned_empty.iter().zip(arch_empty.iter()) {
            assert_eq!(a.0, b.0, "[empty] id mismatch {:?} vs {:?}", a, b);
            assert!(
                (a.1 - b.1).abs() < 1e-6,
                "[empty] score {} vs {}",
                a.1,
                b.1
            );
        }

        // 2) candidates explícitos (path productivo del pipeline). Repetir
        // con subsets variados para detectar divergencias en el lookup
        // archived por chunk_id concreto.
        let queries_and_candidates: Vec<(&str, Vec<u64>)> = vec![
            ("alpha", vec![10, 20, 30, 40]),
            ("alpha", vec![40, 30, 20, 10]),
            ("alpha", vec![10, 30]),
            ("alpha beta", vec![10, 20, 30, 40]),
            ("nonexistent_token", vec![10, 20, 30, 40]),
            ("alpha", vec![999, 40, 10]),
        ];
        for (q, cands) in queries_and_candidates {
            let owned = idx.top_n(q, &cands, 10);
            let arch = top_n_archived(arch, q, &cands, 10);
            assert_eq!(
                owned.len(),
                arch.len(),
                "len mismatch q={q:?} cands={cands:?}: owned={owned:?} arch={arch:?}"
            );
            for (a, b) in owned.iter().zip(arch.iter()) {
                assert_eq!(
                    a.0, b.0,
                    "id mismatch q={q:?} cands={cands:?}: owned={:?} arch={:?}",
                    a, b
                );
                assert!(
                    (a.1 - b.1).abs() < 1e-6,
                    "score mismatch q={q:?} cands={cands:?}: owned={} arch={}",
                    a.1,
                    b.1
                );
            }
        }
    }

    #[test]
    fn rebuild_is_deterministic() {
        let chunks = vec![
            make_chunk(1, "fn handle_connection request response"),
            make_chunk(2, "struct Config name path"),
            make_chunk(3, "impl handler for connection"),
        ];
        let a = Bm25Index::build(&chunks);
        let b = Bm25Index::build(&chunks);
        for c in &chunks {
            assert_eq!(
                a.score("handle connection", c.id).to_bits(),
                b.score("handle connection", c.id).to_bits(),
                "BM25 no determinista para chunk {}",
                c.id
            );
        }
    }
}
