//! Tests de invariantes del pipeline (Capa A).
//!
//! Todos los tests aquí son **offline y deterministas** — no requieren
//! red, no descargan modelos, no dependen del `Embedder` real. Verifican
//! propiedades formales del pipeline (R2/R3/R4/R5) sobre datos sintéticos.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rpgrep::chunk::chunk_file;
use rpgrep::index::bloom::FileBloomIndex;
use rpgrep::search::qubo::{QuboProblem, SimulatedAnnealer};

// ---------------------------------------------------------------------------
// Helpers de fixture (no exportados; uso interno de los tests).
// ---------------------------------------------------------------------------

/// Construye un `FileBloomIndex` con `n` archivos sintéticos, cada uno con
/// un token único `tok_{i}` (≥3 chars) más algunos tokens compartidos para
/// generar interferencia realista.
fn build_synthetic_bloom(n: usize) -> (FileBloomIndex, Vec<(PathBuf, String)>) {
    let mut idx = FileBloomIndex::new();
    let mut catalog = Vec::with_capacity(n);
    for i in 0..n {
        let path = PathBuf::from(format!("synthetic_{i}.rs"));
        let unique = format!("tok_unique_marker_{i:06}");
        let content = format!(
            "fn handler_{i}() {{\n    let common_shared_helper = {unique}();\n    do_work();\n}}\n"
        );
        idx.add_file(path.clone(), &content);
        catalog.push((path, unique));
    }
    (idx, catalog)
}

// ---------------------------------------------------------------------------
// A1 — R3 escalado: Xor filter cero falsos negativos sobre corpus 1k archivos.
// ---------------------------------------------------------------------------

#[test]
fn xor_filter_zero_false_negatives_on_random_corpus() {
    let (idx, catalog) = build_synthetic_bloom(1_000);
    assert_eq!(idx.len(), 1_000, "todos los archivos deben tener filtro");

    for (path, unique_token) in &catalog {
        let cands = idx.candidates(unique_token);
        assert!(
            cands.contains(path),
            "falso negativo en pre-screen: token `{unique_token}` no recupera `{}`",
            path.display()
        );
    }
}

// ---------------------------------------------------------------------------
// A2 — R4: chunk_id estable ante reindexación.
// ---------------------------------------------------------------------------

#[test]
fn chunk_id_is_stable_across_reindex() {
    let mut f = tempfile::NamedTempFile::new().expect("crear tempfile");
    let body: String = (1..=120).map(|i| format!("line_marker_{i}\n")).collect();
    f.write_all(body.as_bytes()).expect("escribir cuerpo");

    let pass_one = chunk_file(f.path(), 40, 8).expect("primer chunking");
    let pass_two = chunk_file(f.path(), 40, 8).expect("segundo chunking");

    assert_eq!(pass_one.len(), pass_two.len(), "mismo conteo de chunks");
    for (a, b) in pass_one.iter().zip(pass_two.iter()) {
        assert_eq!(a.id, b.id, "chunk_id divergente en mismo offset");
        assert_eq!(a.start_line, b.start_line);
        assert_eq!(a.end_line, b.end_line);
    }
}

// ---------------------------------------------------------------------------
// A3 — R3: contrato "query sin tokens ≥3 chars → candidates devuelve TODOS".
//
// Este es el contrato sobre el que `SearchPipeline::search` (pipeline.rs:33-49)
// apoya su garantía de "candidate_set vacío conserva todos los chunks del HNSW".
// Si este contrato se rompe en `FileBloomIndex::candidates`, la garantía de
// alto nivel también se rompe. Testar la garantía a nivel del pipeline
// completo requeriría `Embedder` (red); aquí testeamos el invariante base
// que sostiene la lógica de pipeline.rs:49.
// ---------------------------------------------------------------------------

#[test]
fn empty_candidates_does_not_filter_all() {
    let (idx, _catalog) = build_synthetic_bloom(50);
    let total = idx.len();

    // Query sin ningún token de ≥3 chars: el pre-screen DEBE devolver todos
    // los archivos (no debe interpretarse como "filtra todo").
    let cands: HashSet<_> = idx.candidates("a b c").into_iter().collect();
    assert_eq!(
        cands.len(),
        total,
        "query degenerada debe preservar todos los archivos (R3); recibí {} de {}",
        cands.len(),
        total
    );

    // Caso límite: query completamente vacía.
    let cands_empty: HashSet<_> = idx.candidates("").into_iter().collect();
    assert_eq!(
        cands_empty.len(),
        total,
        "query vacía debe preservar todos los archivos"
    );
}

// ---------------------------------------------------------------------------
// A4 — R2: el SimulatedAnnealer con seed fija es exactamente reproducible.
// ---------------------------------------------------------------------------

#[test]
fn annealer_is_deterministic_with_fixed_seed() {
    let n = 30;
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
    let relevance: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();
    let tokens: Vec<usize> = (0..n).map(|i| 50 + (i * 7) % 80).collect();
    let mut similarity = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = rng.gen::<f32>() * 0.3;
            similarity[i][j] = s;
            similarity[j][i] = s;
        }
    }

    let problem = QuboProblem {
        relevance: relevance.clone(),
        similarity: similarity.clone(),
        tokens: tokens.clone(),
        budget: 600,
        lambda: 0.5,
        mu: 0.001,
    };

    let solver_a = SimulatedAnnealer::default();
    let solver_b = SimulatedAnnealer::default();
    let assignment_a = solver_a.solve(&problem);
    let assignment_b = solver_b.solve(&problem);

    assert_eq!(
        assignment_a, assignment_b,
        "el annealer con misma seed debe producir asignación idéntica"
    );

    // Refuerzo: la energía final también es exactamente la misma.
    let e_a = problem.energy(&assignment_a);
    let e_b = problem.energy(&assignment_b);
    assert_eq!(
        e_a.to_bits(),
        e_b.to_bits(),
        "energía final divergente bit a bit"
    );
}

// ---------------------------------------------------------------------------
// A5 — R5: budget es penalización suave; overflow leve se permite si la
// energía global baja.
//
// Construimos un caso donde tomar AMBOS chunks excede el budget en 50 tokens,
// pero la ganancia de relevancia (-1.6) supera la penalización μ·overflow²
// (0.0001·2500 = 0.25). El solver debe descubrir x = [true, true].
// ---------------------------------------------------------------------------

#[test]
fn budget_overflow_is_tolerated_when_energy_drops() {
    let problem = QuboProblem {
        relevance: vec![1.0, 0.6],
        similarity: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        tokens: vec![100, 50],
        budget: 100,
        lambda: 1.0,
        mu: 0.0001,
    };

    // Sanidad del modelo: tomar ambos da menor energía que tomar solo el primero.
    let only_first = vec![true, false];
    let both = vec![true, true];
    assert!(
        problem.energy(&both) < problem.energy(&only_first),
        "modelo mal construido: tomar ambos no reduce energía"
    );

    let solver = SimulatedAnnealer::default();
    let assignment = solver.solve(&problem);
    assert_eq!(
        assignment,
        vec![true, true],
        "budget es soft-penalty: con μ pequeño y ganancia alta, overflow debe aceptarse"
    );
}
