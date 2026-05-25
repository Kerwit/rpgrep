//! Persistencia del índice completo.
//!
//! Formato v0.2 (`RPGRP002`):
//!
//! ```text
//!   [0..8)     magic = b"RPGRP002"
//!   [8..16)    u64 LE: longitud del bloque rkyv
//!   [16..24)   u64 LE: longitud del bloque bincode (bloom)
//!   [24..24+rk_len)               : rkyv archive (chunks + bm25 + minhash)
//!   [24+rk_len..+bc_len)          : bincode encoding de FileBloomIndex
//! ```
//!
//! La carga usa `memmap2::Mmap` para evitar la allocation de un `Vec<u8>`
//! del tamaño del índice. rkyv deserializa en owned para mantener la API
//! `load() -> IndexStore` estable; la transición a zero-copy real
//! (devolver `&ArchivedIndexStore`) queda como tarea separada de v0.2
//! que requiere refactor de `SearchPipeline`.
//!
//! `bloom` (HashMap<PathBuf, xorf::Xor8>) se serializa con bincode porque
//! `xorf::Xor8` no implementa rkyv en v0.11; ocupa <3 % del archivo en
//! corpus grandes, así que se acepta el coste hasta que se reemplace por
//! un Xor filter propio rkyv-derivable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rkyv::rancor::Error as RkyvError;
use walkdir::WalkDir;

use crate::chunk::{chunk_file, Chunk};
use crate::index::bloom::FileBloomIndex;
use crate::index::bm25::Bm25Index;
use crate::index::minhash::MinHash;
use crate::{Result, RpgrepError};

const MAGIC: &[u8; 8] = b"RPGRP002";
const HEADER_LEN: usize = 24;

#[derive(Default, Serialize, Deserialize)]
pub struct IndexStore {
    pub chunks: Vec<Chunk>,
    pub bloom: FileBloomIndex,
    pub bm25: Bm25Index,
    /// chunk_id → firma MinHash. Pre-calculado en build, evita re-tokenizar en search.
    pub minhash: HashMap<u64, MinHash>,
}

/// Subconjunto rkyv-archivable del índice. `bloom` queda fuera (ver doc del módulo).
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct Payload {
    chunks: Vec<Chunk>,
    bm25: Bm25Index,
    minhash: HashMap<u64, MinHash>,
}

impl IndexStore {
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("rpgrep.idx");

        let payload = Payload {
            chunks: self.chunks.clone(),
            bm25: self.bm25.clone(),
            minhash: self.minhash.clone(),
        };
        let rkyv_bytes = rkyv::to_bytes::<RkyvError>(&payload)
            .map_err(|e| RpgrepError::Persist(format!("rkyv serialize: {e}")))?;
        let bloom_bytes = bincode::serialize(&self.bloom)?;

        let mut file = File::create(&path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(rkyv_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&(bloom_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&rkyv_bytes)?;
        file.write_all(&bloom_bytes)?;
        file.flush()?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("rpgrep.idx");
        let file = File::open(&path)?;
        // SAFETY: mmap requiere que el archivo no sea modificado durante
        // la vida del map. El proceso solo lee el índice tras `save`; en
        // condiciones normales no hay escritores concurrentes.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_LEN {
            return Err(RpgrepError::Persist(format!(
                "índice {} truncado: {} bytes (mínimo {})",
                path.display(),
                mmap.len(),
                HEADER_LEN
            )));
        }
        if &mmap[0..8] != MAGIC {
            return Err(RpgrepError::Persist(format!(
                "magic inválido en {}: esperaba {:?}, encontrado {:?}",
                path.display(),
                std::str::from_utf8(MAGIC).unwrap_or("?"),
                std::str::from_utf8(&mmap[0..8]).unwrap_or("?")
            )));
        }

        let rkyv_len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let bloom_len = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let total = HEADER_LEN.saturating_add(rkyv_len).saturating_add(bloom_len);
        if mmap.len() < total {
            return Err(RpgrepError::Persist(format!(
                "índice {} truncado: cabecera anuncia {} bytes, archivo tiene {}",
                path.display(),
                total,
                mmap.len()
            )));
        }

        let rkyv_section = &mmap[HEADER_LEN..HEADER_LEN + rkyv_len];
        let bloom_section = &mmap[HEADER_LEN + rkyv_len..HEADER_LEN + rkyv_len + bloom_len];

        let payload: Payload = rkyv::from_bytes::<Payload, RkyvError>(rkyv_section)
            .map_err(|e| RpgrepError::Persist(format!("rkyv deserialize: {e}")))?;
        let bloom: FileBloomIndex = bincode::deserialize(bloom_section)?;

        Ok(Self {
            chunks: payload.chunks,
            bloom,
            bm25: payload.bm25,
            minhash: payload.minhash,
        })
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
