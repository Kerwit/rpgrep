//! rpgrep — Búsqueda probabilística de código sin modelos de lenguaje.
//!
//! Pipeline 100% matemático (cero pesos pre-entrenados, cero descargas):
//!   `[A]` Xor filter pre-screen   → descarta archivos sin tokens (O(1), cero FN)
//!   `[B]` BM25 scoring            → relevancia probabilística rᵢ (Robertson 1994)
//!   `[C]` MinHash signatures      → similitud Jaccard sᵢⱼ insesgada (Broder 1997)
//!   `[D]` QUBO + Simulated Annealing → selección óptima bajo budget de tokens
//!
//! El Hamiltoniano de Ising de `[D]` es exactamente lo que un p-bit / annealer
//! cuántico resolvería físicamente. Aquí lo ejecutamos sobre CPU vía
//! muestreo Metropolis: misma matemática, distinto sustrato.

pub mod chunk;
pub mod index;
pub mod search;
// `serve` usa Unix domain sockets (std::os::unix::net) → solo Unix.
#[cfg(unix)]
pub mod serve;

pub use search::pipeline::{SearchPipeline, SearchResult};

#[derive(Debug, thiserror::Error)]
pub enum RpgrepError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("Index: {0}")]
    Index(String),

    #[error("Persist: {0}")]
    Persist(String),
}

pub type Result<T> = std::result::Result<T, RpgrepError>;
