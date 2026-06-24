//! Persistencia del índice completo.
//!
//! Formato v0.2 (`RPGRP003`):
//!
//! ```text
//!   [0..8)     magic = b"RPGRP003"
//!   [8..16)    u64 LE: longitud del bloque rkyv
//!   [16..16+rk_len)   : rkyv archive (chunks + bloom + bm25 + minhash)
//! ```
//!
//! Dos APIs de carga:
//!
//! - `IndexStore::load(dir)` deserializa a tipos owned. Útil para
//!   construcción incremental o tests; paga la copia completa.
//! - `MmappedStore::open(dir)` mantiene el archivo `mmap`-eado y expone
//!   `&ArchivedPayload` (zero-copy real). Es lo que usa `SearchPipeline`
//!   en producción. El sistema operativo pagina bajo demanda; nunca se
//!   reserva un `Vec<u8>` del tamaño del índice.
//!
//! El bloom (xorf::Xor8) viaja dentro del archive rkyv vía `ArchivableXor8`
//! (espejo de los 3 campos públicos de Xor8: seed/block_length/fingerprints).
//! Sobre el archived, `bloom::xor_contains_archived` reimplementa
//! `Xor8::contains` directamente sobre los slices mmap-eados.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rkyv::rancor::Error as RkyvError;
use walkdir::WalkDir;

use crate::chunk::{chunk_file, Chunk};
use crate::index::bloom::{ArchivableXor8, FileBloomIndex};
use crate::index::bm25::Bm25Index;
use crate::index::minhash::MinHash;
use crate::{Result, RpgrepError};

const MAGIC: &[u8; 8] = b"RPGRP003";
const HEADER_LEN: usize = 16;

#[derive(Default, Serialize, Deserialize)]
pub struct IndexStore {
    pub chunks: Vec<Chunk>,
    pub bloom: FileBloomIndex,
    pub bm25: Bm25Index,
    /// chunk_id → firma MinHash. Pre-calculado en build, evita re-tokenizar en search.
    pub minhash: HashMap<u64, MinHash>,
}

/// Representación rkyv-archivable del índice completo. `bloom_filters`
/// es la forma serializable de `FileBloomIndex` (cada `Xor8` mapeado a
/// `ArchivableXor8`, sus 3 campos públicos).
///
/// Público (en el crate) porque `MmappedStore::payload()` devuelve
/// `&ArchivedPayload`, el tipo derivado por `rkyv::Archive` aquí, y
/// `SearchPipeline` lo consume directamente.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Payload {
    pub chunks: Vec<Chunk>,
    pub bloom_filters: HashMap<String, ArchivableXor8>,
    pub bm25: Bm25Index,
    pub minhash: HashMap<u64, MinHash>,
}

impl IndexStore {
    /// Construye el `Payload` rkyv-archivable a partir del store owned.
    /// Reusado por `save` y por `SearchPipeline::from_store` (camino de
    /// tests/benches que necesitan exponer `&ArchivedPayload` sin pasar
    /// por disco).
    pub fn to_payload(&self) -> Payload {
        Payload {
            chunks: self.chunks.clone(),
            bloom_filters: self.bloom.clone().into_archivable(),
            bm25: self.bm25.clone(),
            minhash: self.minhash.clone(),
        }
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("rpgrep.idx");

        // Clone porque save(&self,...) no consume. Para 100k chunks el
        // coste dominante es minhash (~100 MB de Vec<u64>); save es
        // one-shot así que la pasada de memoria es aceptable.
        let payload = self.to_payload();
        let rkyv_bytes = rkyv::to_bytes::<RkyvError>(&payload)
            .map_err(|e| RpgrepError::Persist(format!("rkyv serialize: {e}")))?;

        let mut file = File::create(&path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(rkyv_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&rkyv_bytes)?;
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
        let total = HEADER_LEN.saturating_add(rkyv_len);
        if mmap.len() < total {
            return Err(RpgrepError::Persist(format!(
                "índice {} truncado: cabecera anuncia {} bytes, archivo tiene {}",
                path.display(),
                total,
                mmap.len()
            )));
        }

        let rkyv_section = &mmap[HEADER_LEN..HEADER_LEN + rkyv_len];
        let payload: Payload = rkyv::from_bytes::<Payload, RkyvError>(rkyv_section)
            .map_err(|e| RpgrepError::Persist(format!("rkyv deserialize: {e}")))?;

        Ok(Self {
            chunks: payload.chunks,
            bloom: FileBloomIndex::from_archivable(payload.bloom_filters),
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
        Self::from_dir_with_excludes(root, extensions, lines_per_chunk, overlap, &[])
    }

    /// Igual que [`from_dir`](Self::from_dir) pero además descarta los
    /// ficheros cuya ruta relativa (o nombre) case con cualquiera de los
    /// patrones `excludes`. Glob simple: nombre exacto y `*` como comodín
    /// (p. ej. `SUMMARIES.md`, `*.lock`, `gen_*`). Exclusión explícita,
    /// sin `.gitignore` — coherente con la denylist estática.
    pub fn from_dir_with_excludes(
        root: &Path,
        extensions: &[&str],
        lines_per_chunk: usize,
        overlap: usize,
        excludes: &[String],
    ) -> Result<Self> {
        if !root.is_dir() {
            return Err(RpgrepError::Index(format!(
                "la ruta {} no es un directorio existente",
                root.display()
            )));
        }
        let files = discover_files(root, extensions, excludes);

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut bloom = FileBloomIndex::new();

        for f in &files {
            match chunk_file(f, lines_per_chunk, overlap) {
                Ok(cs) if !cs.is_empty() => match std::fs::read_to_string(f) {
                    Ok(content) => {
                        bloom.add_file(f.to_string_lossy().into_owned(), &content);
                        chunks.extend(cs);
                    }
                    Err(e) => eprintln!("[index] omito {}: {e}", f.display()),
                },
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

/// Vista zero-copy del índice: `Mmap` viva + acceso `&ArchivedPayload`.
///
/// `MmappedStore` es lo que usa `SearchPipeline` en producción. Validamos
/// el header igual que `IndexStore::load`, pero NO deserializamos a tipos
/// owned: `payload()` reinterpreta el slice rkyv-archive directamente y
/// devuelve `&ArchivedPayload`, cuyo tiempo de vida queda atado al
/// `MmappedStore`.
///
/// La validación con `rkyv::access` se hace una sola vez en `open` y
/// queda registrada en `validated: ()` para que `payload()` pueda usar
/// `access_unchecked` en el hot path sin re-revalidar el bytestream.
pub struct MmappedStore {
    mmap: Mmap,
    rkyv_offset: usize,
    rkyv_len: usize,
}

impl MmappedStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("rpgrep.idx");
        let file = File::open(&path)?;
        // SAFETY: ver `IndexStore::load`. Mismas condiciones (sin escritores
        // concurrentes durante la vida del map).
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
        let total = HEADER_LEN.saturating_add(rkyv_len);
        if mmap.len() < total {
            return Err(RpgrepError::Persist(format!(
                "índice {} truncado: cabecera anuncia {} bytes, archivo tiene {}",
                path.display(),
                total,
                mmap.len()
            )));
        }

        // Validación rkyv (CheckBytes). Costosa pero one-shot.
        let rkyv_section = &mmap[HEADER_LEN..HEADER_LEN + rkyv_len];
        rkyv::access::<ArchivedPayload, RkyvError>(rkyv_section)
            .map_err(|e| RpgrepError::Persist(format!("rkyv access: {e}")))?;

        Ok(Self {
            mmap,
            rkyv_offset: HEADER_LEN,
            rkyv_len,
        })
    }

    /// Devuelve la vista `&ArchivedPayload` mapeada en memoria.
    ///
    /// SAFETY: el bytestream fue validado en `open()` con
    /// `rkyv::access`. Reusar `access_unchecked` aquí evita revalidar
    /// en cada query (validación es O(n) sobre el archive entero).
    pub fn payload(&self) -> &ArchivedPayload {
        let bytes = &self.mmap[self.rkyv_offset..self.rkyv_offset + self.rkyv_len];
        // SAFETY: validado en open(). Mientras `self` vive, `mmap` vive,
        // y los bytes son los mismos que validamos en open().
        unsafe { rkyv::access_unchecked::<ArchivedPayload>(bytes) }
    }
}

/// Directorios que NUNCA se indexan: dependencias vendoreadas, artefactos de
/// build, metadatos de VCS y el propio índice. Lista explícita y determinista
/// ("Nada al azar"): no dependemos de `.gitignore` ni de heurísticas.
const EXCLUDED_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    ".hg",
    ".svn",
    "target",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    ".rpgrep",
    ".cache",
    "vendor",
    "__pycache__",
    ".venv",
    ".dart_tool",
    "Pods",
    ".gradle",
];

/// `true` si la entrada es un directorio de la denylist. Usado con
/// `filter_entry` para PODAR el subárbol completo (no se desciende en él).
fn is_excluded_dir(entry: &walkdir::DirEntry) -> bool {
    entry.file_type().is_dir()
        && entry
            .file_name()
            .to_str()
            .map(|n| EXCLUDED_DIRS.contains(&n))
            .unwrap_or(false)
}

fn discover_files(root: &Path, extensions: &[&str], excludes: &[String]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !is_excluded_dir(e)) // poda node_modules/, target/, … sin descender
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            if extensions.is_empty() {
                return true;
            }
            match p.extension().and_then(|x| x.to_str()) {
                Some(ext) => extensions
                    .iter()
                    .any(|allowed| allowed.eq_ignore_ascii_case(ext)),
                None => false,
            }
        })
        .filter(|p| !is_excluded_path(p, root, excludes))
        .collect();
    out.sort();
    out
}

/// `true` si el fichero `p` debe descartarse por `--exclude`. Casa cada
/// patrón contra la ruta relativa a `root` y contra el nombre del fichero,
/// de modo que `--exclude SUMMARIES.md` excluye el fichero esté donde esté.
fn is_excluded_path(p: &Path, root: &Path, excludes: &[String]) -> bool {
    if excludes.is_empty() {
        return false;
    }
    let rel = p.strip_prefix(root).unwrap_or(p);
    let rel_str = rel.to_string_lossy();
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    excludes
        .iter()
        .any(|pat| glob_match(&rel_str, pat) || glob_match(&name, pat))
}

/// Glob mínimo: `*` casa cualquier secuencia (incluida vacía); el resto de
/// caracteres son literales. Sin `?` ni clases — suficiente para nombres y
/// sufijos/prefijos (`*.lock`, `gen_*`, `SUMMARIES.md`).
fn glob_match(haystack: &str, pattern: &str) -> bool {
    if !pattern.contains('*') {
        return haystack == pattern;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        let is_first = i == 0;
        let is_last = i == parts.len() - 1;
        match haystack[pos..].find(part) {
            Some(off) => {
                // El primer segmento debe anclar al inicio si el patrón no
                // empieza por `*`.
                if is_first && off != 0 {
                    return false;
                }
                pos += off + part.len();
            }
            None => return false,
        }
        // El último segmento debe anclar al final si el patrón no acaba en `*`.
        if is_last && !haystack.ends_with(part) {
            return false;
        }
    }
    true
}
