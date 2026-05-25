//! Filtros probabilísticos por archivo para descarte O(1).
//!
//! Usamos Xor filters (Graf & Lemire 2020): más compactos que Bloom para
//! sets estáticos, 0 falsos negativos, ~0.39% falsos positivos a 9 bits/elem.
//!
//! El runtime mantiene `Xor8` en memoria (query-hot, sin conversión). En
//! la frontera de persistencia se convierte a `ArchivableXor8` — un
//! struct rkyv-derivable que copia byte a byte los 3 campos públicos de
//! `Xor8` (`seed`, `block_length`, `fingerprints`). La ida y vuelta es
//! O(1) además del move del buffer porque `Vec::into_boxed_slice` y
//! `Box::into_vec` no copian.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use xorf::{Filter, Xor8};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct FileBloomIndex {
    /// Un filtro por archivo, indexado por su path como String (lo que
    /// archive rkyv puede representar). La key es lo que `Chunk.file`
    /// también almacena → R6 (vocabularios alineados).
    pub filters: HashMap<String, Xor8>,
}

/// Representación rkyv-derivable de `xorf::Xor8`. Espejo de sus 3 campos
/// públicos; ida/vuelta sin copia significativa (move del `Vec<u8>` ↔
/// `Box<[u8]>`).
#[derive(Default, Clone, Debug, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ArchivableXor8 {
    pub seed: u64,
    pub block_length: u64,
    pub fingerprints: Vec<u8>,
}

impl ArchivableXor8 {
    pub fn from_xor8(x: Xor8) -> Self {
        Self {
            seed: x.seed,
            block_length: x.block_length as u64,
            fingerprints: x.fingerprints.into_vec(),
        }
    }

    pub fn into_xor8(self) -> Xor8 {
        Xor8 {
            seed: self.seed,
            block_length: self.block_length as usize,
            fingerprints: self.fingerprints.into_boxed_slice(),
        }
    }
}

impl FileBloomIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construye un filtro con los identificadores únicos del archivo.
    pub fn add_file(&mut self, path: impl Into<String>, content: &str) {
        let tokens: Vec<u64> = unique_token_hashes(content);
        if !tokens.is_empty() {
            let filter = Xor8::from(&tokens);
            self.filters.insert(path.into(), filter);
        }
    }

    /// Archivos que *podrían* contener al menos un token de la query.
    /// Garantía: si un archivo NO aparece aquí, seguro NO contiene los términos.
    pub fn candidates(&self, query: &str) -> Vec<String> {
        let qhashes = unique_token_hashes(query);
        if qhashes.is_empty() {
            return self.filters.keys().cloned().collect();
        }

        self.filters
            .iter()
            .filter(|(_, f)| qhashes.iter().any(|t| f.contains(t)))
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Drena el bloom en su representación rkyv-archivable. Move-only;
    /// los buffers de fingerprints viajan sin copia.
    pub fn into_archivable(self) -> HashMap<String, ArchivableXor8> {
        self.filters
            .into_iter()
            .map(|(p, x)| (p, ArchivableXor8::from_xor8(x)))
            .collect()
    }

    /// Reconstrucción inversa: dato deserializado por rkyv → runtime.
    pub fn from_archivable(map: HashMap<String, ArchivableXor8>) -> Self {
        Self {
            filters: map.into_iter().map(|(p, a)| (p, a.into_xor8())).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.filters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

fn unique_token_hashes(text: &str) -> Vec<u64> {
    let mut seen = std::collections::HashSet::new();
    for token in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if token.len() >= 3 {
            let mut h = DefaultHasher::new();
            token.to_lowercase().hash(&mut h);
            seen.insert(h.finish());
        }
    }
    seen.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_screen_zero_false_negatives() {
        let mut idx = FileBloomIndex::new();
        idx.add_file("a.rs", "fn handle_connection() { do_thing(); }");
        idx.add_file("b.rs", "struct Config;");

        let cands = idx.candidates("handle_connection");
        assert!(cands.iter().any(|s| s == "a.rs"));
    }

    #[test]
    fn archivable_round_trip_preserves_contains() {
        let mut idx = FileBloomIndex::new();
        idx.add_file("a.rs", "fn handle_connection() { do_thing(); }");
        idx.add_file("b.rs", "struct Config; fn make_thing() {}");

        let cands_before = idx.candidates("handle_connection");
        // Ida y vuelta por la representación archivable.
        let archivable = idx.into_archivable();
        let idx2 = FileBloomIndex::from_archivable(archivable);
        let cands_after = idx2.candidates("handle_connection");
        assert_eq!(cands_before, cands_after);
        assert!(cands_after.iter().any(|s| s == "a.rs"));
    }
}
