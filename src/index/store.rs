//! Persistencia del índice completo (chunks + bloom + HNSW) con bincode.
//!
//! v0.2: migrar a `rkyv` + `memmap2` para carga zero-copy.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::chunk::Chunk;
use crate::index::bloom::FileBloomIndex;
use crate::index::hnsw::HnswIndex;
use crate::Result;

#[derive(Serialize, Deserialize)]
pub struct IndexStore {
    pub chunks: Vec<Chunk>,
    pub bloom: FileBloomIndex,
    pub hnsw: HnswIndex,
}

impl IndexStore {
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("rpgrep.idx");
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("rpgrep.idx");
        let bytes = std::fs::read(path)?;
        let store: Self = bincode::deserialize(&bytes)?;
        Ok(store)
    }
}
