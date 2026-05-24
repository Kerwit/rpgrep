//! Segmentación de archivos en unidades semánticas básicas.
//!
//! v0.1: line-based con solapamiento.
//! v0.2: AST-aware mediante tree-sitter.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: u64,
    pub file: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

impl Chunk {
    /// Estimación rápida de tokens (~4 chars/token para código).
    pub fn token_estimate(&self) -> usize {
        (self.text.len() / 4).max(1)
    }
}

/// Divide un archivo en chunks por ventanas de líneas con solapamiento.
/// IDs derivados de hash(path + start_line) → estables ante re-indexación.
pub fn chunk_file(
    path: &Path,
    lines_per_chunk: usize,
    overlap: usize,
) -> std::io::Result<Vec<Chunk>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(vec![]);
    }

    let mut chunks = Vec::new();
    let stride = lines_per_chunk.saturating_sub(overlap).max(1);

    let mut i = 0;
    while i < lines.len() {
        let end = (i + lines_per_chunk).min(lines.len());
        let text = lines[i..end].join("\n");
        chunks.push(Chunk {
            id: chunk_id(path, i),
            file: path.to_path_buf(),
            start_line: i + 1,
            end_line: end,
            text,
        });
        if end == lines.len() {
            break;
        }
        i += stride;
    }
    Ok(chunks)
}

fn chunk_id(path: &Path, start: usize) -> u64 {
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    start.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn chunks_overlap_correctly() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let body: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        f.write_all(body.as_bytes()).unwrap();

        let chunks = chunk_file(f.path(), 20, 5).unwrap();
        assert!(!chunks.is_empty());
        // Stride = 15 → al menos 6 chunks para 100 líneas
        assert!(chunks.len() >= 6);
        // El primer chunk arranca en línea 1
        assert_eq!(chunks[0].start_line, 1);
    }
}
