//! Persistencia del índice completo (chunks + bloom + bm25 + minhash) con bincode.
//!
//! v0.2: migrar a `rkyv` + `memmap2` para carga zero-copy.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::chunk::{chunk_file, Chunk};
use crate::index::bloom::FileBloomIndex;
use crate::index::bm25::Bm25Index;
use crate::index::minhash::MinHash;
use crate::Result;

#[derive(Default, Serialize, Deserialize)]
pub struct IndexStore {
    pub chunks: Vec<Chunk>,
    pub bloom: FileBloomIndex,
    pub bm25: Bm25Index,
    /// chunk_id → firma MinHash. Pre-calculado en build, evita re-tokenizar en search.
    pub minhash: HashMap<u64, MinHash>,
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

    /// Construye un índice recorriendo `root` y procesando archivos cuya
    /// extensión esté en `extensions` (sin punto: `&["rs", "md"]`). Si
    /// `extensions` está vacío, indexa **todos** los archivos legibles.
    ///
    /// Errores de I/O en archivos sueltos NO abortan la indexación: se
    /// loguean por stderr y se continúa. Solo errores en la persistencia
    /// final o en la creación del directorio raíz propagan `Err`.
    pub fn from_dir(
        root: &Path,
        extensions: &[&str],
        lines_per_chunk: usize,
        overlap: usize,
    ) -> Result<Self> {
        let files = discover_files(root, extensions);

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut bloom = FileBloomIndex::new();

        for f in &files {
            match chunk_file(f, lines_per_chunk, overlap) {
                Ok(cs) if !cs.is_empty() => {
                    match std::fs::read_to_string(f) {
                        Ok(content) => {
                            bloom.add_file(f.clone(), &content);
                            chunks.extend(cs);
                        }
                        Err(e) => eprintln!("[index] omito {}: {e}", f.display()),
                    }
                }
                Ok(_) => {} // archivo vacío: nada que indexar
                Err(e) => eprintln!("[index] omito {}: {e}", f.display()),
            }
        }

        let bm25 = Bm25Index::build(&chunks);
        let minhash: HashMap<u64, MinHash> = chunks
            .iter()
            .map(|c| (c.id, MinHash::from_text(&c.text)))
            .collect();

        Ok(Self {
            chunks,
            bloom,
            bm25,
            minhash,
        })
    }

    /// Número de archivos únicos indexados (deducido del Xor filter).
    pub fn n_files(&self) -> usize {
        self.bloom.len()
    }
}

fn discover_files(root: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            if extensions.is_empty() {
                return true;
            }
            match p.extension().and_then(|x| x.to_str()) {
                Some(ext) => extensions.iter().any(|allowed| allowed.eq_ignore_ascii_case(ext)),
                None => false,
            }
        })
        .collect();
    out.sort();
    out
}
