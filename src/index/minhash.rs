//! MinHash signatures (Broder 1997) — estimador insesgado de Jaccard.
//!
//! Sustituye la similitud coseno densa para `sᵢⱼ` del QUBO sin embeddings.
//! Dada una firma de K hashes por chunk:
//!
//!   `Ĵ(A, B) = |{ i : sig(A)[i] = sig(B)[i] }| / K`
//!
//! Estimador insesgado de la similitud Jaccard real; varianza = J·(1-J)/K.
//! Con K=128, error típico ~ 0.04 — suficiente para alimentar el término de
//! redundancia del QUBO, donde la matemática solo necesita un orden parcial
//! razonable entre pares.
//!
//! Hashes generados con seeds 0..K sobre cada token (lowercased, ≥3 chars).
//! Determinista bit a bit para un mismo `(text, k)`.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const DEFAULT_K: usize = 128;
const MIN_TOKEN_LEN: usize = 3;

#[derive(
    Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct MinHash {
    /// K hashes mínimos. `sig[i]` = min sobre todos los tokens del
    /// `Hash64(i || token_lowercased)`.
    pub sig: Vec<u64>,
}

impl MinHash {
    /// Firma con K = DEFAULT_K (128).
    pub fn from_text(text: &str) -> Self {
        Self::from_text_with_k(text, DEFAULT_K)
    }

    pub fn from_text_with_k(text: &str, k: usize) -> Self {
        debug_assert!(k > 0, "K debe ser > 0");
        let mut sig = vec![u64::MAX; k];

        for token in text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| t.len() >= MIN_TOKEN_LEN)
        {
            let lower = token.to_lowercase();
            for (i, slot) in sig.iter_mut().enumerate() {
                let mut h = DefaultHasher::new();
                (i as u64).hash(&mut h);
                lower.hash(&mut h);
                let v = h.finish();
                if v < *slot {
                    *slot = v;
                }
            }
        }
        Self { sig }
    }

    /// Estimador Jaccard ∈ [0, 1]. Si ambas firmas son "vacías" (no se vio
    /// ningún token), devuelve 0 — convención para que el QUBO no penalice
    /// chunks degenerados.
    pub fn jaccard(&self, other: &Self) -> f32 {
        let k = self.sig.len().min(other.sig.len());
        if k == 0 {
            return 0.0;
        }
        // Si ambas firmas siguen siendo todo u64::MAX, no había tokens.
        // Devolver 0 evita un "1.0" espurio de chunks vacíos.
        let both_empty =
            self.sig.iter().all(|&v| v == u64::MAX) && other.sig.iter().all(|&v| v == u64::MAX);
        if both_empty {
            return 0.0;
        }
        let matches = (0..k).filter(|&i| self.sig[i] == other.sig[i]).count();
        matches as f32 / k as f32
    }

    pub fn k(&self) -> usize {
        self.sig.len()
    }
}

/// Estimador Jaccard ∈ [0, 1] sobre dos firmas archived. Misma matemática
/// que `MinHash::jaccard`, comparando `u64_le` elemento a elemento.
pub fn archived_jaccard(a: &ArchivedMinHash, b: &ArchivedMinHash) -> f32 {
    let k = a.sig.len().min(b.sig.len());
    if k == 0 {
        return 0.0;
    }
    let sentinel = u64::MAX;
    let both_empty = a.sig.iter().all(|v| v.to_native() == sentinel)
        && b.sig.iter().all(|v| v.to_native() == sentinel);
    if both_empty {
        return 0.0;
    }
    let matches = (0..k)
        .filter(|&i| a.sig[i].to_native() == b.sig[i].to_native())
        .count();
    matches as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_yields_jaccard_one() {
        let a = MinHash::from_text("fn validate_input(user, payload) -> Result<()>");
        let b = MinHash::from_text("fn validate_input(user, payload) -> Result<()>");
        assert_eq!(a.jaccard(&b), 1.0);
    }

    #[test]
    fn disjoint_vocabularies_yield_low_jaccard() {
        let a = MinHash::from_text("alpha bravo charlie delta echo foxtrot");
        let b = MinHash::from_text("uniform victor whiskey xray yankee zulu");
        // Vocabularios completamente disjuntos: estimador ≈ 0 con varianza
        // pequeña; pedimos < 0.05 con K=128.
        let j = a.jaccard(&b);
        assert!(j < 0.05, "esperaba ~0, recibí {j}");
    }

    #[test]
    fn jaccard_decreases_when_adding_disjoint_tokens() {
        let base = "alpha bravo charlie delta";
        let a = MinHash::from_text(base);
        let b1 = MinHash::from_text(&format!("{base} echo"));
        let b2 = MinHash::from_text(&format!("{base} echo foxtrot golf hotel india juliet"));
        let j1 = a.jaccard(&b1);
        let j2 = a.jaccard(&b2);
        assert!(
            j1 >= j2,
            "añadir tokens disjuntos debe reducir o mantener Jaccard: j1={j1} j2={j2}"
        );
    }

    #[test]
    fn empty_text_signature_yields_zero_jaccard() {
        let a = MinHash::from_text("");
        let b = MinHash::from_text("");
        assert_eq!(a.jaccard(&b), 0.0);

        // Empty vs non-empty también 0: ningún hash coincide porque uno es u64::MAX.
        let c = MinHash::from_text("alpha bravo");
        assert_eq!(a.jaccard(&c), 0.0);
    }

    #[test]
    fn signature_is_deterministic() {
        let a = MinHash::from_text("handle_connection request response config");
        let b = MinHash::from_text("handle_connection request response config");
        assert_eq!(a.sig, b.sig);
    }

    #[test]
    fn jaccard_estimator_within_expected_error() {
        // Construimos sets con Jaccard real conocido y verificamos error
        // < 3·sqrt(J(1-J)/K) ≈ 3σ (criterio empírico generoso).
        let a_tokens = "alpha bravo charlie delta echo foxtrot golf hotel india juliet";
        let b_tokens = "alpha bravo charlie delta echo papa quebec romeo sierra tango";
        // |A ∩ B| = 5, |A ∪ B| = 15 → Jaccard real = 5/15 ≈ 0.333
        let a = MinHash::from_text(a_tokens);
        let b = MinHash::from_text(b_tokens);
        let est = a.jaccard(&b);
        let real = 5.0_f32 / 15.0;
        let sigma = (real * (1.0 - real) / DEFAULT_K as f32).sqrt();
        assert!(
            (est - real).abs() < 3.0 * sigma + 0.05,
            "Jaccard estimado {est} fuera de 3σ ({:.3}) del real {real}",
            3.0 * sigma
        );
    }

    #[test]
    fn custom_k_smaller_signature() {
        let a = MinHash::from_text_with_k("alpha bravo charlie", 32);
        assert_eq!(a.k(), 32);
    }

    #[test]
    fn archived_jaccard_matches_owned() {
        use rkyv::rancor::Error as RkyvError;

        let pairs = [
            ("alpha bravo charlie", "alpha bravo charlie"),
            ("alpha bravo", "uniform victor whiskey"),
            ("", "alpha bravo"),
            ("", ""),
            (
                "alpha bravo charlie delta echo foxtrot",
                "alpha bravo papa quebec",
            ),
        ];

        for (a_text, b_text) in pairs {
            let a = MinHash::from_text(a_text);
            let b = MinHash::from_text(b_text);
            let owned = a.jaccard(&b);

            let a_bytes = rkyv::to_bytes::<RkyvError>(&a).unwrap();
            let b_bytes = rkyv::to_bytes::<RkyvError>(&b).unwrap();
            let a_arch = rkyv::access::<ArchivedMinHash, RkyvError>(&a_bytes).unwrap();
            let b_arch = rkyv::access::<ArchivedMinHash, RkyvError>(&b_bytes).unwrap();
            let archived = archived_jaccard(a_arch, b_arch);

            assert!(
                (owned - archived).abs() < 1e-6,
                "divergencia owned={owned} arch={archived} para ({a_text:?}, {b_text:?})"
            );
        }
    }
}
