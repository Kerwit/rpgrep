//! Filtros probabilísticos por archivo para descarte O(1).
//!
//! Usamos Xor filters (Graf & Lemire 2020): más compactos que Bloom para
//! sets estáticos, 0 falsos negativos, ~0.39% falsos positivos a 9 bits/elem.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use xorf::{Filter, Xor8};

#[derive(Default, Serialize, Deserialize)]
pub struct FileBloomIndex {
    /// Un filtro por archivo, indexado por path.
    pub filters: HashMap<PathBuf, Xor8>,
}

impl FileBloomIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construye un filtro con los identificadores únicos del archivo.
    pub fn add_file(&mut self, path: PathBuf, content: &str) {
        let tokens: Vec<u64> = unique_token_hashes(content);
        if !tokens.is_empty() {
            let filter = Xor8::from(&tokens);
            self.filters.insert(path, filter);
        }
    }

    /// Archivos que *podrían* contener al menos un token de la query.
    /// Garantía: si un archivo NO aparece aquí, seguro NO contiene los términos.
    pub fn candidates(&self, query: &str) -> Vec<PathBuf> {
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
        idx.add_file(
            PathBuf::from("a.rs"),
            "fn handle_connection() { do_thing(); }",
        );
        idx.add_file(PathBuf::from("b.rs"), "struct Config;");

        let cands = idx.candidates("handle_connection");
        assert!(cands.contains(&PathBuf::from("a.rs")));
    }
}
