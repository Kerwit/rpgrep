//! rpgrep — Búsqueda semántica de código mediante pipeline probabilístico clásico.
//!
//! Pipeline:
//!   [A] Xor filter pre-screen   → descarta archivos sin tokens relevantes (O(1))
//!   [B] HNSW retrieval          → top-K aproximado sobre embeddings
//!   [C] (Opcional) Re-rank      → cross-encoder ligero (v0.2)
//!   [D] QUBO + Simulated Annealing → selección óptima bajo budget de tokens
//!
//! Equivalente al Hamiltoniano de Ising que resolvería un p-bit físico,
//! aquí ejecutado sobre CPU mediante muestreo Metropolis.

pub mod chunk;
pub mod embed;
pub mod index;
pub mod search;

pub use search::pipeline::{SearchPipeline, SearchResult};

#[derive(Debug, thiserror::Error)]
pub enum RpgrepError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("Embedding: {0}")]
    Embedding(String),

    #[error("Index: {0}")]
    Index(String),

    #[error("Serialization: {0}")]
    Serde(#[from] bincode::Error),
}

pub type Result<T> = std::result::Result<T, RpgrepError>;
