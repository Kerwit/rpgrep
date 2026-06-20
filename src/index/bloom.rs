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
#[derive(
    Default, Clone, Debug, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
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

/// `Xor8::contains(key)` reimplementado directamente sobre el `Archived`
/// del filtro (zero-copy real: opera sobre los slices mmap-eados sin
/// reconstruir `Xor8`).
///
/// Algoritmo idéntico al de `xorf 0.11` (`xor_contains_impl!` en
/// `prelude/xor.rs`): mix murmur3 + 3 hashes rotacionales + reduce
/// Lemire + XOR de fingerprints. Verificado en
/// `archived_xor_contains_matches_runtime`.
pub fn xor_contains_archived(arch: &ArchivedArchivableXor8, key: u64) -> bool {
    let seed: u64 = arch.seed.to_native();
    let block_length: usize = arch.block_length.to_native() as usize;

    // mix(key, seed) — murmur3 finalization mix sobre key + seed.
    let h = mix64(key.wrapping_add(seed));
    let fp: u8 = (h ^ (h >> 32)) as u8;

    // h_b = reduce(rotl64(h, b * 21), block_length) para b ∈ {0, 1, 2}.
    let h0 = reduce_lemire(h.rotate_left(0), block_length);
    let h1 = reduce_lemire(h.rotate_left(21), block_length);
    let h2 = reduce_lemire(h.rotate_left(42), block_length);

    let fps: &[u8] = arch.fingerprints.as_slice();
    fp == fps[h0] ^ fps[h1 + block_length] ^ fps[h2 + 2 * block_length]
}

/// Garantía equivalente a `FileBloomIndex::candidates` pero sobre el
/// `ArchivedHashMap<ArchivedString, ArchivableXor8>` mmap-eado.
///
/// Devuelve `Vec<String>` para que `pipeline` compare contra `Chunk.file`
/// (que viene como `&ArchivedString` también, comparable a `&str`).
pub fn candidates_archived(
    filters: &rkyv::collections::swiss_table::ArchivedHashMap<
        rkyv::string::ArchivedString,
        ArchivedArchivableXor8,
    >,
    query: &str,
) -> Vec<String> {
    let qhashes = unique_token_hashes(query);
    if qhashes.is_empty() {
        return filters.keys().map(|k| k.as_str().to_string()).collect();
    }

    filters
        .iter()
        .filter(|(_, f)| qhashes.iter().any(|t| xor_contains_archived(f, *t)))
        .map(|(p, _)| p.as_str().to_string())
        .collect()
}

// Réplica local de las primitivas hashing/reduce de xorf — el algoritmo
// es público (Xor Filters, Graf & Lemire 2020) y `xorf` las expone como
// macros internas. Replicar aquí preserva R3 (zero false negatives) sobre
// el path zero-copy sin abrir la abstracción de `xorf::Filter`.
#[inline]
fn mix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^= k >> 33;
    k
}

#[inline]
fn reduce_lemire(hash: u64, n: usize) -> usize {
    (((hash as u32) as u64 * n as u64) >> 32) as usize
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

    /// Paridad `candidates_archived` ↔ `FileBloomIndex::candidates` para
    /// queries arbitrarias sobre un set de archivos. Si difieren, el
    /// pre-screen zero-copy filtra distinto al runtime y el pipeline
    /// produce un recall distinto.
    #[test]
    fn candidates_archived_matches_owned_for_diverse_queries() {
        use rkyv::rancor::Error as RkyvError;

        let mut idx = FileBloomIndex::new();
        idx.add_file(
            "src/handler.rs",
            "pub fn handle_connection(request: Request, response: Response) -> Result<(), Error> { validate_user(&request); }",
        );
        idx.add_file(
            "src/auth.rs",
            "fn validate_user(req: &Request) -> bool { check_token(&req.token) }",
        );
        idx.add_file(
            "src/config.rs",
            "pub struct Config { pub host: String, pub port: u16 } impl Config { pub fn load() -> Self { todo!() } }",
        );
        idx.add_file(
            "src/empty.rs",
            "// just a comment without enough alphanumerics ab cd",
        );

        let archivable = idx.clone().into_archivable();
        let bytes = rkyv::to_bytes::<RkyvError>(&archivable).unwrap();
        let archived = rkyv::access::<
            rkyv::collections::swiss_table::ArchivedHashMap<
                rkyv::string::ArchivedString,
                ArchivedArchivableXor8,
            >,
            RkyvError,
        >(&bytes)
        .unwrap();

        let queries = [
            "handle_connection",
            "validate_user",
            "Config load",
            "nonexistent_token_xyz",
            "handle request",
            "ab", // <3 chars → empty tokens → debe devolver todos
            "",
            "request response validate",
        ];

        for q in queries {
            let mut owned: Vec<String> = idx.candidates(q);
            let mut arch: Vec<String> = candidates_archived(archived, q);
            owned.sort();
            arch.sort();
            assert_eq!(owned, arch, "divergencia en query {q:?}");
        }
    }

    /// R3 sobre el path zero-copy: `xor_contains_archived` debe coincidir
    /// con `Xor8::contains` para todos los tokens del archivo original.
    /// Si difieren, generaríamos falsos negativos sobre el archived.
    #[test]
    fn archived_xor_contains_matches_runtime() {
        use rkyv::rancor::Error as RkyvError;
        use xorf::Filter;

        let mut idx = FileBloomIndex::new();
        // Vocabulario suficientemente diverso para que un xor8 estable se
        // construya sin reseed prolongado.
        idx.add_file(
            "a.rs",
            "fn handle_connection(request, response) { do_thing(); validate_user(); }",
        );
        let archivable = idx.clone().into_archivable();
        let xor = idx.filters.get("a.rs").unwrap();

        let tokens: Vec<u64> = unique_token_hashes(
            "handle_connection request response do_thing validate_user struct missing",
        );

        // Archive el HashMap completo y reinterpreta como ArchivedHashMap.
        let bytes = rkyv::to_bytes::<RkyvError>(&archivable).unwrap();
        let archived = rkyv::access::<
            rkyv::collections::swiss_table::ArchivedHashMap<
                rkyv::string::ArchivedString,
                ArchivedArchivableXor8,
            >,
            RkyvError,
        >(&bytes)
        .unwrap();
        let arch_xor = archived
            .iter()
            .find(|(k, _)| k.as_str() == "a.rs")
            .map(|(_, v)| v)
            .unwrap();

        for t in &tokens {
            assert_eq!(
                xor.contains(t),
                xor_contains_archived(arch_xor, *t),
                "Divergencia archived↔runtime para token {t}"
            );
        }
    }
}
