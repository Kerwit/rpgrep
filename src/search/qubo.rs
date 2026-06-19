//! QUBO + Simulated Annealing para selección óptima de chunks.
//!
//! Formulación:
//!   min  E(x) = -Σᵢ rᵢ·xᵢ  +  λ·Σᵢⱼ sᵢⱼ·xᵢxⱼ  +  μ·(Σᵢ tᵢ·xᵢ - B)²
//!   s.a. xᵢ ∈ {0, 1}
//!
//! - rᵢ = relevancia del chunk i (similitud coseno query↔chunk)
//! - sᵢⱼ = similitud entre chunks i,j (penalización por redundancia)
//! - tᵢ = tokens del chunk i, B = budget
//!
//! Este Hamiltoniano es exactamente lo que un p-bit / annealer cuántico
//! resolvería físicamente por relajación térmica. Aquí lo simulamos sobre
//! CPU con criterio de Metropolis — misma matemática, distinto sustrato.

use rand::prelude::*;
use rand_distr::{Distribution, Uniform};

pub struct QuboProblem {
    pub relevance: Vec<f32>,
    pub similarity: Vec<Vec<f32>>,
    pub tokens: Vec<usize>,
    pub budget: usize,
    pub lambda: f32, // peso de redundancia
    pub mu: f32,     // peso de penalización de budget
}

impl QuboProblem {
    pub fn energy(&self, x: &[bool]) -> f32 {
        let mut e: f32 = 0.0;

        // Lineal: -relevancia
        for (i, &xi) in x.iter().enumerate() {
            if xi {
                e -= self.relevance[i];
            }
        }

        // Cuadrático: redundancia
        for i in 0..x.len() {
            if !x[i] {
                continue;
            }
            for j in (i + 1)..x.len() {
                if x[j] {
                    e += self.lambda * self.similarity[i][j];
                }
            }
        }

        // Penalización suave de budget
        let used: i64 = x
            .iter()
            .zip(&self.tokens)
            .map(|(&xi, &t)| if xi { t as i64 } else { 0 })
            .sum();
        let overflow = (used - self.budget as i64).max(0) as f32;
        e += self.mu * overflow * overflow;

        e
    }
}

pub struct SimulatedAnnealer {
    pub initial_temp: f32,
    pub final_temp: f32,
    pub steps: usize,
    pub seed: u64,
}

impl Default for SimulatedAnnealer {
    fn default() -> Self {
        Self {
            initial_temp: 5.0,
            final_temp: 0.01,
            steps: 5_000,
            seed: 0xC0DE_F00D,
        }
    }
}

impl SimulatedAnnealer {
    /// Resuelve el QUBO; devuelve la asignación binaria óptima encontrada.
    pub fn solve(&self, problem: &QuboProblem) -> Vec<bool> {
        let n = problem.relevance.len();
        if n == 0 {
            return vec![];
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let uniform = Uniform::new(0.0_f32, 1.0_f32).expect("rango uniforme válido [0,1)");

        // Inicialización greedy: top-K por relevancia hasta llenar budget.
        let mut x = greedy_init(problem);
        let mut best_x = x.clone();
        let mut best_e = problem.energy(&x);
        let mut cur_e = best_e;

        // Schedule geométrico de temperatura.
        let alpha = (self.final_temp / self.initial_temp).powf(1.0 / self.steps.max(1) as f32);
        let mut temp = self.initial_temp;

        for _ in 0..self.steps {
            // Movimiento: flip de un bit aleatorio.
            let idx = rng.random_range(0..n);
            x[idx] = !x[idx];
            let new_e = problem.energy(&x);
            let delta = new_e - cur_e;

            // Criterio de Metropolis: acepta siempre si mejora,
            // probabilísticamente si empeora.
            if delta < 0.0 || uniform.sample(&mut rng) < (-delta / temp).exp() {
                cur_e = new_e;
                if cur_e < best_e {
                    best_e = cur_e;
                    best_x = x.clone();
                }
            } else {
                x[idx] = !x[idx]; // revertir
            }

            temp *= alpha;
        }

        best_x
    }
}

fn greedy_init(problem: &QuboProblem) -> Vec<bool> {
    let n = problem.relevance.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        problem.relevance[b]
            .partial_cmp(&problem.relevance[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut x = vec![false; n];
    let mut used = 0_usize;
    for i in order {
        if used + problem.tokens[i] <= problem.budget {
            x[i] = true;
            used += problem.tokens[i];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_budget_approximately() {
        let n = 5;
        let problem = QuboProblem {
            relevance: vec![1.0, 0.8, 0.6, 0.9, 0.5],
            similarity: vec![vec![0.0; n]; n],
            tokens: vec![100, 200, 150, 250, 100],
            budget: 400,
            lambda: 1.0,
            mu: 0.01,
        };
        let solver = SimulatedAnnealer::default();
        let x = solver.solve(&problem);
        let used: usize = x
            .iter()
            .zip(&problem.tokens)
            .map(|(&xi, &t)| if xi { t } else { 0 })
            .sum();
        assert!(used <= problem.budget + 50, "overflow excesivo: {used}");
    }

    #[test]
    fn prefers_diverse_chunks_over_redundant_ones() {
        // 3 chunks: A y B casi idénticos (alta sim), C distinto.
        // Esperamos seleccionar {A,C} o {B,C}, NUNCA {A,B}.
        let n = 3;
        let mut similarity = vec![vec![0.0_f32; n]; n];
        similarity[0][1] = 0.95;
        similarity[1][0] = 0.95;

        let problem = QuboProblem {
            relevance: vec![0.9, 0.85, 0.7],
            similarity,
            tokens: vec![100, 100, 100],
            budget: 250,
            lambda: 2.0,
            mu: 0.01,
        };
        let solver = SimulatedAnnealer::default();
        let x = solver.solve(&problem);
        let picked_both_ab = x[0] && x[1];
        assert!(
            !picked_both_ab,
            "no debería tomar A y B redundantes a la vez"
        );
        assert!(x[2], "C debería estar seleccionado por diversidad");
    }
}
