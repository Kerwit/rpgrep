# Validación y benchmark comparativo — `rpgrep`

Documento de la suite de validación end-to-end definida por el Súper Prompt
de `validacion_y_benchmark_comparativo.md`. Cumple dos funciones:

1. **Validar `rpgrep` como buscador de contexto semántico** (invariantes del
   pipeline + latencia objetivo P95 < 150 ms @ 100k chunks, según
   `BLUEPRINT.md` §1).
2. **Producir un análisis comparativo reproducible** entre `rpgrep`, `grep`
   (GNU), `ripgrep` (`rg`) y `ast-grep` (`sg`) sobre un corpus controlado.

---

## 1. Propósito y diferencia semántica entre las herramientas

Las tres herramientas que comparamos **NO son intercambiables**. Cada una
responde a una pregunta distinta:

| Herramienta | Pregunta que responde                                              | Tipo de output           |
|-------------|--------------------------------------------------------------------|--------------------------|
| `grep`/`rg` | "¿Dónde aparece exactamente la subcadena X?"                       | Lista de **matches**     |
| `sg`        | "¿Dónde aparece la estructura sintáctica Y (AST pattern)?"         | Lista de **matches**     |
| `rpgrep`    | "Dame el **mejor paquete de contexto** para entender Z."           | **Bundle** óptimo bajo budget |

`rpgrep` devuelve un *bundle* con tres propiedades simultáneas:
- **Relevante** (similitud semántica query↔chunk vía embeddings),
- **Diverso** (penaliza redundancia entre chunks seleccionados),
- **Dentro de budget** (`Σ tokens(chunk) ≤ B` como penalización suave).

Comparar "quién encuentra más líneas" sería tramposo: `grep` siempre ganará
en recall sintáctico literal y `rpgrep` siempre ganará en utilidad semántica.
Esta validación mide **ejes complementarios**, no un ganador absoluto.

---

## 2. Metodología del corpus dorado

**Corpus indexable**: `src/` del propio repositorio `rpgrep` (10 archivos
Rust, ~600 líneas, ~30-40 chunks con `chunk_file(40, 8)`).

**Corpus dorado**: 25 pares `query ↔ expected_substrings` curados manualmente
en [tests/fixtures/golden_corpus.tsv](../tests/fixtures/golden_corpus.tsv).

Formato TSV (3 columnas separadas por tab):

```
query                                  | expected_substrings (CSV) | expected_files (CSV, informativo)
```

**Definición de "chunk relevante"**: un chunk se considera relevante para
una query si su `chunk.text.to_lowercase()` contiene al menos una de las
`expected_substrings` (también lowercase, plain string match).

**Justificación**: criterio determinista y reproducible sin etiquetado
ID-por-ID, robusto a cambios de chunking. Si refactorizas `src/` y rompes
una etiqueta, **actualiza el TSV** (cambio de etiqueta, no del test).

---

## 3. Definición formal de las métricas

### 3.1 Recall@K (semántica)
> ¿Cuánta proporción de los chunks relevantes está en el top-K que devuelve
> `rpgrep`?

```
Recall@K(q) = |TopK(q) ∩ Relevant(q)| / |Relevant(q)|
```

- `TopK(q)`: primeros K resultados del pipeline (post-QUBO, ordenados por
  score descendente).
- `Relevant(q)`: set de chunks del corpus que cumplen la regla de §2.
- Si `|Relevant(q)| == 0`, la query se descarta del cálculo (no contribuye).

K=5 por defecto.

### 3.2 MRR — Mean Reciprocal Rank
> Posición media (recíproca) del **primer** chunk relevante en el top-K.

```
MRR(q) = 1 / rank_first_relevant(q)    si existe alguno, 0 en otro caso
MRR    = mean_q( MRR(q) )
```

Rank 1-indexado. Si ningún chunk del top-K es relevante, contribuye 0.

### 3.3 Diversity@K
> Cuán heterogéneo es el bundle devuelto, medido como **complemento de la
> similitud media intra-resultados** vía Jaccard de trigramas.

```
Diversity@K(q) = 1 - mean_{i<j} J( trigrams(top_i.text), trigrams(top_j.text) )
```

donde `J(A, B) = |A ∩ B| / max(|A ∪ B|, 1)`.

Coherente con el término `λ · sᵢⱼ · xᵢxⱼ` del QUBO (`pipeline.rs:104`,
`qubo.rs:6`): mide lo mismo que el solver penaliza.

### 3.4 P95 de latencia
> Percentil 95 del tiempo end-to-end del pipeline **post-embedding** (Xor →
> HNSW → QUBO) sobre 50 queries con vectores aleatorios, corpus sintético
> de 100k chunks, 384 dims (= MiniLM L6 v2).

```
samples = [latency(query_i) for i in 1..50]
P95 = ceil(0.95 · 50) -ésimo valor de sort(samples)
gate: P95 < 150 ms   (BLUEPRINT §1)
```

**Importante**: el bench **no** mide la latencia del `Embedder`
(`fastembed`/ONNX) — esa fase añade ~10-30 ms para queries cortas en CPU.
El gate aplica al pipeline post-embedding, que es la parte donde `rpgrep`
tiene control directo.

---

## 4. Cómo reproducir

### 4.1 Capa A — Invariantes del pipeline (rápido, offline)

```bash
cargo test --test pipeline_invariants
```

5 tests deterministas, sin red, sin descarga de modelo. Deben pasar todos.

### 4.2 Capa B — Calidad semántica (descarga modelo ONNX la 1ª vez)

```bash
cargo test --test semantic_quality -- --ignored --nocapture
```

Marcado `#[ignore]` por defecto. Imprime las métricas por query y la media.
Asserts contra **pisos laxos** (`Recall@5 ≥ 0.30`, `MRR ≥ 0.20`,
`Diversity@5 ≥ 0.40`) — pensados para detectar regresión brutal, no para
fijar rendimiento absoluto.

### 4.3 Capa C — Benches y gate de latencia

Reportes Criterion (P50/P95/P99 sobre 1k y 10k):
```bash
cargo bench --bench pipeline
```

Salida HTML en `target/criterion/`. **No falla** si P95 > 150 ms; solo reporta.

Gate duro sobre 100k chunks:
```bash
cargo test --test p95_gate -- --ignored --nocapture
```

Marcado `#[ignore]` porque construir un HNSW de 100k×384 tarda
~30–60 s en CPU de portátil. **Falla** si P95 > 150 ms.

### 4.4 Capa D — Harness comparativo

```bash
./scripts/bench_compare.sh | column -ts $'\t'
```

Requiere `python3` y `cargo`. Si `rg` o `sg` faltan, sus columnas se
rellenan con `WARN` y el script continúa (exit 0).

Construye automáticamente:
- `target/release/rpgrep` (vía `cargo build --release`),
- Índice demo en `.rpgrep-bench/` (vía
  `cargo run --release --example build_demo_index -- src .rpgrep-bench`,
  workaround R6 — ver §6).

Variables de entorno:
- `CORPUS=src` (directorio a indexar/buscar)
- `ITERATIONS=5` (muestras por query, reporta mediana)
- `INDEX_DIR=.rpgrep-bench` (donde persiste el índice demo)

---

## 5. Resultados

> ⏳ **Pendiente de ejecución**. Tras correr las suites de §4.2, §4.3 y §4.4
> en una máquina de referencia, completar las tablas siguientes con valores
> reales. La estructura del documento NO debe cambiar.

### 5.1 Capa B — Calidad semántica (media sobre 25 queries)

| Métrica       | Valor medido | Piso CI | Notas |
|---------------|--------------|---------|-------|
| Recall@5      | _pendiente_  | 0.30    | Sobre `src/` del repo |
| MRR           | _pendiente_  | 0.20    |       |
| Diversity@5   | _pendiente_  | 0.40    |       |
| Queries evaluadas | _pendiente_ / 25 | ≥15 | El resto descartado por `|Relevant|=0` |

### 5.2 Capa C — Latencias del pipeline post-embedding

| Etapa               | Tamaño  | P50 (ms) | P95 (ms) | P99 (ms) |
|---------------------|---------|----------|----------|----------|
| xor_candidates      | 10k     | _pend_   | _pend_   | _pend_   |
| hnsw_topk           | 10k     | _pend_   | _pend_   | _pend_   |
| qubo_anneal         | n=200   | _pend_   | _pend_   | _pend_   |
| pipeline_post_embed | 10k     | _pend_   | _pend_   | _pend_   |
| **gate (100k)**     | **100k**| _pend_   | _pend_   | _pend_   |

Gate: **P95 < 150 ms @ 100k chunks** (BLUEPRINT §1). Si falla, documentar
la regresión aquí con timestamp y hash del commit.

### 5.3 Capa D — Tabla comparativa (extracto de `bench_compare.sh`)

Pegar aquí la salida completa de `./scripts/bench_compare.sh | column -ts $'\t'`.
Las latencias son **medianas** sobre `ITERATIONS=5`. Columnas:

| Columna             | Significado                                                  |
|---------------------|--------------------------------------------------------------|
| `query`             | Texto de la query (corpus dorado)                            |
| `grep_matches`      | Líneas con match del primer token (`grep -rnE`)              |
| `rg_matches`        | Suma de matches reportados por `rg --count-matches`          |
| `sg_matches`        | Líneas con match estructural de `sg run -p <token> -l rust`  |
| `rpgrep_n`          | Nº de chunks en el bundle final tras QUBO                    |
| `rpgrep_top1_score` | Score (similitud coseno) del chunk top-1                     |
| `*_ms`              | Mediana de latencia en ms (`ITERATIONS=5`)                   |

```
[pegar salida real aquí]
```

**Caveat sobre la comparación**: `grep`/`rg`/`sg` reciben el **primer token**
de cada query (no son semánticos: no entienden frases). Comparar
`grep_matches` con `rpgrep_n` mide ortogonalmente: matches literales vs.
tamaño del bundle semántico. No es una comparación de "calidad", es de
**comportamiento**.

---

## 6. Limitaciones conocidas

### 6.1 R6 — Comando `Index` no cableado
`cli.rs:70` lanza `anyhow::bail!`. El binario `rpgrep` no puede construir
un índice por sí mismo todavía. **Workaround**:
[examples/build_demo_index.rs](../examples/build_demo_index.rs) replica los
pasos del `TODO` en `cli.rs:64-70` para indexar un directorio.

Esto es **deuda**: cuando `cli.rs::Commands::Index` se cablee, eliminar el
example y reescribir `bench_compare.sh` para usar `rpgrep index` directamente.

### 6.2 Embeddings sintéticos en benches y gate
[benches/pipeline.rs](../benches/pipeline.rs) y
[tests/p95_gate.rs](../tests/p95_gate.rs) usan vectores random normalizados
a 384 dims **en lugar de** embeddings reales de MiniLM. Razón:

- Generar 100k embeddings reales tarda ~minutos en CPU; incompatible con
  ejecución repetida en CI.
- La fase de embedding es independiente del pipeline (`Embedder::embed`
  no toca el HNSW ni el Xor filter). Aislarla es honesto.

**Consecuencia**: el bench/gate mide latencia del pipeline **post-embed**.
La latencia real end-to-end suma además ~10-30 ms del embedder por query.

### 6.3 Tests semánticos requieren red la primera vez
[tests/semantic_quality.rs](../tests/semantic_quality.rs) y
[examples/build_demo_index.rs](../examples/build_demo_index.rs) llaman a
`Embedder::new()`, que descarga MiniLM L6 v2 (~80 MB) en `~/.cache` (vía
`fastembed`). En CI offline, marcar la red como dependencia o pre-cachear
el modelo.

### 6.4 Duplicación intencional bench ↔ gate test
Las helpers `build_synthetic_store` y `pipeline_post_embed` están
duplicadas entre [benches/pipeline.rs](../benches/pipeline.rs:33) y
[tests/p95_gate.rs](../tests/p95_gate.rs:36). Razón: los integration tests
no pueden importar módulos del directorio `benches/` ni viceversa, y
exponerlos desde `src/` contaminaría el código de producción. Si cambias
una versión, **sincroniza la otra**. Marcadas con comentario `SYNCED`.

### 6.5 Corpus dorado pequeño
25 pares es un piso, no un techo. Para mediciones más robustas, ampliar a
≥100 pares con criterio humano. Cada par cuesta ~2 minutos de etiquetado;
multiplicar por número objetivo.

### 6.6 R7 — Features de v0.2 NO testeadas
Esta suite **no** valida tree-sitter chunking, re-rank con cross-encoder,
modo watch, modo serve, ni migración a `rkyv`. Cuando esas features
aterricen, añadir tests específicos; no antes.

---

## Apéndice — Resumen del árbol de archivos de la suite

```
Cargo.toml                          MODIFICADO (criterion, rand_chacha, [[bench]])
tests/pipeline_invariants.rs        Capa A — 5 tests offline (R2/R3/R4/R5)
tests/semantic_quality.rs           Capa B — Recall@K, MRR, Diversity@K [ignore]
tests/p95_gate.rs                   Capa C — gate P95 < 150 ms @ 100k [ignore]
tests/fixtures/golden_corpus.tsv    25 pares query↔relevantes
benches/pipeline.rs                 Capa C — Criterion (xor / hnsw / qubo / e2e)
examples/build_demo_index.rs        Workaround R6 — constructor de índice
scripts/bench_compare.sh            Capa D — harness comparativo
docs/VALIDATION.md                  Este documento
```

Cero modificaciones a `src/`. La validación NO contamina producción.
