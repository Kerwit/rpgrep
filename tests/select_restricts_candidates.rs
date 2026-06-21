//! `select` re-rankea SOLO los chunks cuyo fichero pertenece al conjunto
//! candidato (salida de `rg -l` / `ast-grep`), aplicando el mismo pipeline
//! BM25 → MinHash → QUBO que `search` pero sin el pre-screen Xor.

use std::collections::HashSet;

use rpgrep::chunk::Chunk;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::index::bm25::Bm25Index;
use rpgrep::index::minhash::MinHash;
use rpgrep::index::store::IndexStore;
use rpgrep::SearchPipeline;

/// Índice sintético con 3 chunks en `auth.rs`, `render.rs` y `session.rs`.
/// `auth.rs` y `session.rs` contienen el vocabulario de la query; `render.rs`
/// es deliberadamente ajeno.
fn build_store() -> IndexStore {
    let files = [
        (
            "auth.rs",
            "fn authenticate(session: Session, token: Token) -> bool {\n    verify(session, token)\n}\n",
        ),
        (
            "render.rs",
            "fn draw(frame: Frame, pixels: Buffer) {\n    blit(frame, pixels);\n}\n",
        ),
        (
            "session.rs",
            "fn open_session(token: Token) -> Session {\n    Session::new(token)\n}\n",
        ),
    ];

    let mut chunks: Vec<Chunk> = Vec::with_capacity(files.len());
    let mut bloom = FileBloomIndex::new();

    for (i, (file, text)) in files.iter().enumerate() {
        bloom.add_file((*file).to_string(), text);
        chunks.push(Chunk {
            id: i as u64,
            file: (*file).to_string(),
            start_line: 1,
            end_line: 3,
            text: (*text).to_string(),
        });
    }

    let bm25 = Bm25Index::build(&chunks);
    let minhash = chunks
        .iter()
        .map(|c| (c.id, MinHash::from_text(&c.text)))
        .collect();

    IndexStore {
        chunks,
        bloom,
        bm25,
        minhash,
    }
}

fn set(files: &[&str]) -> HashSet<String> {
    files.iter().map(|s| s.to_string()).collect()
}

const QUERY: &str = "authenticate session token";

#[test]
fn select_only_returns_files_in_set() {
    let pipeline = SearchPipeline::from_store(build_store());
    let files = set(&["auth.rs"]);

    let results = pipeline.select(QUERY, 4000, 50, &files).unwrap();

    assert!(
        !results.is_empty(),
        "select debería devolver al menos un chunk de auth.rs"
    );
    for r in &results {
        assert!(
            files.contains(&r.chunk.file),
            "select devolvió un fichero fuera del set: {}",
            r.chunk.file
        );
    }
}

#[test]
fn select_empty_set_returns_empty() {
    let pipeline = SearchPipeline::from_store(build_store());
    let results = pipeline.select(QUERY, 4000, 50, &HashSet::new()).unwrap();
    assert!(
        results.is_empty(),
        "set vacío debe devolver vacío (no conserva el índice)"
    );
}

#[test]
fn select_never_returns_files_outside_set() {
    let pipeline = SearchPipeline::from_store(build_store());
    let files = set(&["auth.rs", "session.rs"]);

    let results = pipeline.select(QUERY, 4000, 50, &files).unwrap();

    for r in &results {
        assert_ne!(
            r.chunk.file, "render.rs",
            "render.rs no está en el set y no debe aparecer"
        );
        assert!(files.contains(&r.chunk.file));
    }
}
