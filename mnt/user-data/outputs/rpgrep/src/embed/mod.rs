//! Generación de embeddings vía fastembed (ONNX MiniLM por defecto).
//!
//! NOTA: la API de `fastembed` ha evolucionado; estas llamadas asumen
//! la familia v4. Si tu versión difiere, ajusta `try_new`/`embed`.

use crate::{Result, RpgrepError};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Carga MiniLM-L6-v2 (~80 MB, 384-dim). Descarga la primera vez.
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2),
        )
        .map_err(|e| RpgrepError::Embedding(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(texts, None)
            .map_err(|e| RpgrepError::Embedding(e.to_string()))
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed_batch(vec![text.to_string()])?;
        v.pop()
            .ok_or_else(|| RpgrepError::Embedding("empty embedding".into()))
    }

    pub const fn dim() -> usize {
        384
    }
}
