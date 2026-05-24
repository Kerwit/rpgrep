//! HNSW (Hierarchical Navigable Small Worlds) sobre embeddings.
//!
//! Usamos `instant-distance`. Distancia coseno entre vectores normalizados.

use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddedPoint(pub Vec<f32>);

impl Point for EmbeddedPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Distancia coseno: 1 - dot(a,b) / (|a||b|).
        let dot: f32 = self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum();
        let na: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let denom = (na * nb).max(1e-12);
        1.0 - (dot / denom)
    }
}

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    pub map: HnswMap<EmbeddedPoint, u64>, // u64 = chunk_id
}

impl HnswIndex {
    pub fn build(points: Vec<Vec<f32>>, ids: Vec<u64>) -> Self {
        let points: Vec<EmbeddedPoint> = points.into_iter().map(EmbeddedPoint).collect();
        let map = Builder::default().ef_construction(200).build(points, ids);
        Self { map }
    }

    /// Top-K vecinos aproximados; devuelve (chunk_id, distancia_coseno).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut search = Search::default();
        let q = EmbeddedPoint(query.to_vec());
        self.map
            .search(&q, &mut search)
            .take(k)
            .map(|item| (*item.value, item.distance))
            .collect()
    }
}
