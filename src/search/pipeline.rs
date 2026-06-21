//! Orquestación del pipeline probabilístico de búsqueda.
//!
//!   `[A]` Xor filter pre-screen   → archivos candidatos (cero falsos negativos)
//!   `[B]` BM25 scoring            → top-N chunks por relevancia probabilística rᵢ
//!   `[C]` MinHash Jaccard         → matriz de redundancia sᵢⱼ
//!   `[D]` QUBO + Simulated SA     → selección óptima bajo budget de tokens
//!
//! Cero pesos pre-entrenados, cero red, cero descargas: todo el pipeline
//! corre sobre estructuras matemáticas puras (Xor, BM25, MinHash, Metropolis).
//!
//! **Zero-copy real**: `SearchPipeline` opera sobre `&ArchivedPayload`
//! (mmap-eado por `MmappedStore`). Sólo se materializan `Chunk` owned
//! en los `SearchResult` devueltos al caller — el resto del pipeline
//! lee directamente del bytestream archived.

use std::collections::HashSet;
use std::path::Path;

use rkyv::rancor::Error as RkyvError;
use rkyv::rend::u64_le;

use crate::chunk::Chunk;
use crate::index::bloom::candidates_archived;
use crate::index::bm25::top_n_archived;
use crate::index::minhash::archived_jaccard;
use crate::index::store::{ArchivedPayload, IndexStore, MmappedStore};
use crate::search::qubo::{QuboProblem, SimulatedAnnealer};
use crate::Result;
use crate::RpgrepError;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
}

/// Backing del pipeline: el `&ArchivedPayload` debe vivir mientras viva
/// el `SearchPipeline`. Dos casos:
///
/// - `Mmap`: persistencia real, lo que usa `load(dir)`. Cero copia.
/// - `Buffer`: bytes rkyv en heap, lo que usa `from_store(store)`. Una
///   copia única (la serialización a bytes) para que tests/benches que
///   construyen el índice en memoria puedan usar el mismo path zero-copy.
///
/// El buffer guarda `AlignedVec<16>` (alineamiento que rkyv exige para
/// `access_unchecked`). **Convertir a `Vec<u8>` con `into_vec()` rompe
/// el alineamiento** porque `malloc` solo garantiza 8 bytes en muchos
/// allocators — el bytestream luego se lee desalineado y produce valores
/// corruptos sin que `access` checked lo detecte siempre.
enum Backing {
    Mmap(MmappedStore),
    Buffer(rkyv::util::AlignedVec<16>),
}

impl Backing {
    fn payload(&self) -> &ArchivedPayload {
        match self {
            Backing::Mmap(store) => store.payload(),
            // SAFETY: los bytes fueron generados por `rkyv::to_bytes` y
            // validados con `rkyv::access` en `from_store`. Mientras
            // `self` viva, los bytes viven en el mismo allocation
            // **alineado a 16 bytes** (AlignedVec garantiza alignment).
            Backing::Buffer(buf) => unsafe {
                rkyv::access_unchecked::<ArchivedPayload>(buf.as_slice())
            },
        }
    }
}

pub struct SearchPipeline {
    backing: Backing,
}

impl SearchPipeline {
    /// Carga el índice persistido en `dir`. Path productivo: `Mmap` →
    /// zero-copy real, ninguna copia del payload.
    pub fn load(dir: &Path) -> Result<Self> {
        let store = MmappedStore::open(dir)?;
        Ok(Self {
            backing: Backing::Mmap(store),
        })
    }

    /// Construye un `SearchPipeline` desde un `IndexStore` owned. Usado
    /// por tests y benches que arman el índice en memoria. Paga una
    /// única serialización a bytes para alimentar el mismo path
    /// `&ArchivedPayload` que `load`.
    pub fn from_store(store: IndexStore) -> Self {
        let payload = store.to_payload();
        let bytes = rkyv::to_bytes::<RkyvError>(&payload)
            .expect("rkyv serialize en from_store no debería fallar");
        // Validación checkeada one-shot. Si falla, el bytestream que acabamos
        // de generar es inválido — eso es un bug del crate, no del usuario.
        rkyv::access::<ArchivedPayload, RkyvError>(&bytes).expect("rkyv access en from_store");
        Self {
            backing: Backing::Buffer(bytes),
        }
    }

    /// Variante checked de `from_store` para tests que prefieren un
    /// `Result` explícito en lugar de `expect`.
    pub fn try_from_store(store: IndexStore) -> Result<Self> {
        let payload = store.to_payload();
        let bytes = rkyv::to_bytes::<RkyvError>(&payload)
            .map_err(|e| RpgrepError::Persist(format!("rkyv serialize: {e}")))?;
        rkyv::access::<ArchivedPayload, RkyvError>(&bytes)
            .map_err(|e| RpgrepError::Persist(format!("rkyv access: {e}")))?;
        Ok(Self {
            backing: Backing::Buffer(bytes),
        })
    }

    /// Pipeline completo: `[A]` Xor → `[B]` BM25 → `[C]` MinHash → `[D]` QUBO.
    pub fn search(&self, query: &str, budget: usize, topk: usize) -> Result<Vec<SearchResult>> {
        let payload = self.backing.payload();

        // [A] Pre-screen con Xor filter sobre archivos. Opera directamente
        // sobre el HashMap archived (zero-copy).
        let candidate_set: HashSet<String> = candidates_archived(&payload.bloom_filters, query)
            .into_iter()
            .collect();

        // chunk_ids elegibles: si `candidate_set` está vacío (query sin tokens
        // ≥3 chars), conservamos TODOS los chunks → preserva R3.
        let candidate_ids: Vec<u64> = if candidate_set.is_empty() {
            payload.chunks.iter().map(|c| c.id.to_native()).collect()
        } else {
            payload
                .chunks
                .iter()
                .filter(|c| candidate_set.contains(c.file.as_str()))
                .map(|c| c.id.to_native())
                .collect()
        };

        rank(payload, query, budget, topk, candidate_ids)
    }

    /// Variante de `search` restringida a un conjunto de ficheros candidatos
    /// (p. ej. salida de `rg -l` o `ast-grep`). Salta el pre-screen Xor `[A]`
    /// y aplica EXACTAMENTE el mismo pipeline `[B]` BM25 → `[C]` MinHash →
    /// `[D]` QUBO sobre los chunks cuyo fichero ∈ `files`.
    ///
    /// A diferencia de `search`, si `files` está vacío devuelve `Ok(vec![])`
    /// (NO conserva todo el índice). Los ficheros ausentes del índice se
    /// ignoran en silencio.
    pub fn select(
        &self,
        query: &str,
        budget: usize,
        topk: usize,
        files: &HashSet<String>,
    ) -> Result<Vec<SearchResult>> {
        if files.is_empty() {
            return Ok(vec![]);
        }

        let payload = self.backing.payload();

        let candidate_ids: Vec<u64> = payload
            .chunks
            .iter()
            .filter(|c| files.contains(c.file.as_str()))
            .map(|c| c.id.to_native())
            .collect();

        rank(payload, query, budget, topk, candidate_ids)
    }
}

/// Cuerpo común del pipeline tras la resolución de `candidate_ids`:
/// `[B]` BM25 → `[C]` MinHash → `[D]` QUBO. Compartido por `search`
/// (pre-screen Xor) y `select` (restricción por fichero).
fn rank(
    payload: &ArchivedPayload,
    query: &str,
    budget: usize,
    topk: usize,
    candidate_ids: Vec<u64>,
) -> Result<Vec<SearchResult>> {
    if candidate_ids.is_empty() {
        return Ok(vec![]);
    }

    // [B] BM25 scoring → top-N por relevancia, vía archived.
    let topn = top_n_archived(&payload.bm25, query, &candidate_ids, topk.max(10));
    if topn.is_empty() {
        return Ok(vec![]);
    }

    // Resolución id → posición dentro de `payload.chunks` (sin copiar).
    // Para 100k chunks, esto es un single-pass que construye el índice
    // de lookup una vez por query.
    let id_to_pos: std::collections::HashMap<u64, usize> = payload
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id.to_native(), i))
        .collect();

    let filtered: Vec<(usize, f32)> = topn
        .into_iter()
        .filter_map(|(id, score)| id_to_pos.get(&id).map(|&pos| (pos, score)))
        .collect();
    if filtered.is_empty() {
        return Ok(vec![]);
    }

    // Normalizar relevancia a [0, 1] dividiendo por el máximo del batch.
    let max_score = filtered
        .iter()
        .map(|(_, s)| *s)
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let relevance: Vec<f32> = filtered.iter().map(|(_, s)| *s / max_score).collect();
    let tokens: Vec<usize> = filtered
        .iter()
        .map(|(pos, _)| {
            // token_estimate sobre el archived: text es ArchivedString.
            let text_len = payload.chunks[*pos].text.as_str().len();
            (text_len / 4).max(1)
        })
        .collect();

    // [C] Matriz sᵢⱼ vía MinHash Jaccard sobre archived.
    let similarity = archived_similarity_matrix(payload, &filtered);

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

    // Materializamos los Chunk seleccionados — única copia owned del
    // pipeline. Para `topk` chunks (típicamente ≤ 50), el coste es
    // despreciable frente a evitar copiar los 100k chunks completos.
    let mut results: Vec<SearchResult> = filtered
        .into_iter()
        .enumerate()
        .filter(|(i, _)| assignment[*i])
        .map(|(_, (pos, s))| {
            let arch_chunk = &payload.chunks[pos];
            SearchResult {
                chunk: chunk_from_archived(arch_chunk),
                score: s / max_score,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

/// Construye la matriz sᵢⱼ ∈ [0, 1] consultando las firmas MinHash del
/// archived (mmap). Si alguna firma falta, su fila/columna queda en 0 —
/// equivalente a "asumir no-redundancia".
fn archived_similarity_matrix(payload: &ArchivedPayload, items: &[(usize, f32)]) -> Vec<Vec<f32>> {
    let n = items.len();
    let mut sim = vec![vec![0.0_f32; n]; n];

    let resolved: Vec<Option<&rkyv::Archived<crate::index::minhash::MinHash>>> = items
        .iter()
        .map(|(pos, _)| {
            let id = payload.chunks[*pos].id.to_native();
            payload.minhash.get(&u64_le::from_native(id))
        })
        .collect();

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
            let j_sim = archived_jaccard(si, sj);
            sim[i][j] = j_sim;
            sim[j][i] = j_sim;
        }
    }
    sim
}

/// Materializa un `Chunk` owned desde un `ArchivedChunk`. Solo se llama
/// para los chunks que el pipeline devuelve al caller — todo el resto
/// del trabajo sucede sobre las vistas archived.
fn chunk_from_archived(arch: &rkyv::Archived<Chunk>) -> Chunk {
    Chunk {
        id: arch.id.to_native(),
        file: arch.file.as_str().to_string(),
        start_line: arch.start_line.to_native() as usize,
        end_line: arch.end_line.to_native() as usize,
        text: arch.text.as_str().to_string(),
    }
}
