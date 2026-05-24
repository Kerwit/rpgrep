//! Embeddings vía `fastembed` (MiniLM L6 v2, ONNX, CPU).

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::{Result, RpgrepError};

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .map_err(|e| RpgrepError::Embedding(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self
            .model
            .embed(vec![text], None)
            .map_err(|e| RpgrepError::Embedding(e.to_string()))?;
        out.pop()
            .ok_or_else(|| RpgrepError::Embedding("empty embedding output".into()))
    }

    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(texts, None)
            .map_err(|e| RpgrepError::Embedding(e.to_string()))
    }
}
