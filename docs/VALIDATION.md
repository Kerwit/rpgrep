# Validación y benchmark comparativo — `rpgrep`

Documento de la suite de validación end-to-end. Cumple dos funciones:

1. **Validar `rpgrep` como buscador probabilístico de contexto** (invariantes
   del pipeline + latencia objetivo P95 < 150 ms @ 100k chunks, según
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
- **Relevante** (BM25 query↔chunk: modelo probabilístico de relevancia),
- **Diverso** (MinHash Jaccard penaliza redundancia entre chunks seleccionados),
- **Dentro de budget** (`Σ tokens(chunk) ≤ B` como penalización suave en el QUBO).

Comparar "quién encuentra más líneas" sería tramposo: `grep` siempre ganará
en recall sintáctico literal y `rpgrep` siempre ganará en utilidad como
*bundle bajo budget*. Esta validación mide **ejes complementarios**, no un
ganador absoluto.

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

### 3.1 Recall@K
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
> Percentil 95 del tiempo end-to-end del pipeline (Xor → BM25 → MinHash →
> QUBO) sobre 50 queries de texto, corpus sintético de 100k chunks.

```
samples = [latency(query_i) for i in 1..50]
P95 = ceil(0.95 · 50) -ésimo valor de sort(samples)
gate: P95 < 150 ms   (BLUEPRINT §1)
```

Sin embeddings ni red: el gate mide el pipeline completo tal como lo ve el
usuario final. No queda ninguna fase neuronal que aislar.

---

## 4. Cómo reproducir

### 4.1 Capa A — Invariantes del pipeline (rápido, offline)

```bash
cargo test --test pipeline_invariants
```

5 tests deterministas, sin red, sin descarga de modelo. Deben pasar todos.

### 4.2 Capa B — Calidad del retrieval (sin red, sin descargas)

```bash
cargo test --test semantic_quality -- --nocapture
```

Imprime las métricas por query y la media. Asserts contra **pisos laxos**
(`Recall@5 ≥ 0.30`, `MRR ≥ 0.20`, `Diversity@5 ≥ 0.40`) — pensados para
detectar regresión brutal, no para fijar rendimiento absoluto.

### 4.3 Capa C — Benches y gate de latencia

Reportes Criterion (P50/P95/P99 sobre 1k y 10k):
```bash
cargo bench --bench pipeline
```

Salida HTML en `target/criterion/`. **No falla** si P95 > 150 ms; solo reporta.

Gate duro sobre 100k chunks:
```bash
cargo test --release --test p95_gate -- --ignored --nocapture
```

Marcado `#[ignore]` porque construir 100k chunks (BM25 + MinHash + Xor)
tarda ~30–60 s en CPU de portátil. **Falla** si P95 > 150 ms.

> ⚠️ `--release` es **obligatorio**. En modo debug, BM25, MinHash y
> simulated annealing son órdenes de magnitud más lentos: saturan CPU
> durante minutos y el gate de P95 < 150 ms pierde su sentido porque
> mide tiempos que no representan producción.

### 4.4 Capa D — Harness comparativo

```bash
./scripts/bench_compare.sh | column -ts $'\t'
```

Requiere `python3` y `cargo`. Si `rg` o `sg` faltan, sus columnas se
rellenan con `WARN` y el script continúa (exit 0).

Construye automáticamente:
- `target/release/rpgrep` (vía `cargo build --release`),
- Índice en `.rpgrep-bench/` vía `rpgrep index "$CORPUS" --out .rpgrep-bench`
  (sin descargas: el pipeline es 100% probabilístico).

Variables de entorno:
- `CORPUS=src` (directorio a indexar/buscar)
- `ITERATIONS=5` (muestras por query, reporta mediana)
- `INDEX_DIR=.rpgrep-bench` (donde persiste el índice demo)

---

## 5. Resultados

> §5.1 y §5.2 reflejan ejecución de referencia del 2026-05-25 en `--release`.
> §5.3 (harness comparativo de Capa D) sigue pendiente — completar tras
> correr `./scripts/bench_compare.sh` en máquina de referencia.

### 5.1 Capa B — Calidad semántica (media sobre 25 queries)

| Métrica       | Valor medido | Piso CI | Notas |
|---------------|--------------|---------|-------|
| Recall@5      | **0.369**    | 0.30    | Sobre `src/` del repo |
| MRR           | **1.000**    | 0.20    | Top-1 siempre relevante en las 25 queries |
| Diversity@5   | **0.855**    | 0.40    |       |
| Queries evaluadas | **25** / 25 | ≥15 | `skipped=0` (todas tienen `|Relevant|≥1`) |

Ejecución de referencia (2026-05-25, `--release`): `cargo test --release --test
semantic_quality -- --nocapture`, tiempo total 0.05 s.

### 5.2 Capa C — Latencias por etapa del pipeline

Etapas intermedias (`cargo bench --bench pipeline`, 2026-05-25, `--release`):

| Etapa               | Tamaño  | Media (ms) | IC95 inf | IC95 sup |
|---------------------|---------|------------|----------|----------|
| xor_candidates      | 1k      | 0.00128    | 0.00127  | 0.00129  |
| xor_candidates      | 10k     | 0.00894    | 0.00891  | 0.00897  |
| bm25_topn           | 1k      | 0.316      | 0.314    | 0.317    |
| bm25_topn           | 10k     | 3.179      | 3.164    | 3.197    |
| qubo_anneal         | n=50    | 1.725      | 1.720    | 1.729    |
| qubo_anneal         | n=100   | 2.313      | 2.301    | 2.325    |
| qubo_anneal         | n=200   | 3.832      | 3.824    | 3.841    |
| pipeline_e2e        | 1k      | 2.454      | 2.444    | 2.467    |
| pipeline_e2e        | 10k     | 5.706      | 5.684    | 5.738    |

> Criterion reporta **media ± IC95 bootstrap** sobre `≥20` muestras, no
> percentiles de la distribución de latencias. Para P50/P95/P99 reales
> ver la fila gate (50 queries individuales medidas con `Instant::now`).

Carga del índice (`bench_load`, comparativa formato `RPGRP002` rkyv+mmap
vs baseline bincode v0.1, mismo `IndexStore` sintético):

| Tamaño | Formato        | Media       | IC95 inf | IC95 sup | Tamaño on-disk |
|--------|----------------|-------------|----------|----------|----------------|
| 1k     | bincode v0.1   | 472 µs      | 458 µs   | 494 µs   | 1.36 MB        |
| 1k     | rkyv v0.2      | **446 µs**  | 441 µs   | 452 µs   | 1.47 MB        |
| 10k    | bincode v0.1   | 7.89 ms     | 7.68 ms  | 8.12 ms  | 13.53 MB       |
| 10k    | rkyv v0.2      | **5.87 ms** | 5.65 ms  | 6.16 ms  | 14.67 MB       |

Speedup load: **1.06×** @ 1k, **1.34×** @ 10k — la ventaja relativa
crece con el corpus (el coste fijo de bincode parse domina a tamaño
pequeño; a tamaño grande, rkyv evita la pasada de parseo entera). El
formato rkyv ocupa **~8–9 % más en disco** (alignment + metadata de
archive) — coste aceptable a cambio de carga ~34 % más rápida @ 10k.
La medición es **warm-cache** (page cache de OS caliente entre
iteraciones); el comportamiento cold-cache no está caracterizado.

Gate de latencia (`cargo test --release --test p95_gate -- --ignored`):

| Etapa               | Tamaño  | P50 (ms) | P95 (ms) | P99 (ms) |
|---------------------|---------|----------|----------|----------|
| **gate v0.1 (bincode)** | **100k** | 50.0 | 52.4     | 56.8     |
| **gate v0.2 (rkyv)**    | **100k** | 47.8 | **50.2** | 50.4     |

Gate: **P95 < 150 ms @ 100k chunks** (BLUEPRINT §1). Si falla, documentar
la regresión aquí con timestamp y hash del commit.

Ejecución de referencia (2026-05-25, `--release`, corpus sintético seed=0xE5,
50 queries tras 3 de warm-up, budget=4000 tokens, top_n=50): el gate
**PASA con holgura ~3×** (P95=50.2 ms vs gate 150 ms). Construcción del
corpus de 100k chunks en ~1.3 s; ejecución total del test ~3.9 s. La
mejora P95 (52.4 → 50.2 ms) es marginal: el gate construye el corpus en
memoria y no toca `save`/`load`, así que la migración de persistencia no
puede mover esta métrica significativamente — el beneficio real de rkyv
aparece en operaciones de carga repetidas (CLI `search` sobre índices
grandes), pendiente de microbench dedicado.

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
| `rpgrep_top1_score` | Relevancia BM25 normalizada a [0,1] del chunk top-1          |
| `*_ms`              | Mediana de latencia en ms (`ITERATIONS=5`)                   |

Ejecución de referencia (2026-05-25, `CORPUS=src`, `ITERATIONS=5`,
`grep` BSD/macOS, `rg`=ripgrep 15.1.0, `sg`=ast-grep 0.42.1, índice
generado en `.rpgrep-bench/` sobre 12 archivos / 44 chunks):

```
query                                     grep_matches  rg_matches  sg_matches  rpgrep_n  rpgrep_top1_score  grep_ms  rg_ms  sg_ms  rpgrep_ms
simulated annealing temperature schedule  0             0           0           6         1.000              8.5      11.0   17.8   8.0
xor filter zero false negatives           1             1           0           7         1.000              7.5      11.0   17.0   8.0
qubo problem energy hamiltonian           2             2           2           8         1.000              7.8      10.2   17.5   7.8
budget penalty overflow soft constraint   20            20          5           5         1.000              7.5      10.8   17.2   8.7
estimate tokens per chunk                 2             2           0           10        1.000              8.7      11.2   17.5   9.0
bm25 term frequency document length       10            12          7           10        1.000              8.6      11.6   18.6   9.0
persist index to bincode file             7             7           0           7         1.000              9.1      9.9    15.3   7.7
walk directory tree for indexing          1             1           0           8         1.000              7.7      9.4    16.0   7.7
cli search subcommand definition          5             5           5           5         1.000              7.6      11.7   16.7   7.4
minhash jaccard similarity estimator      10            12          6           10        1.000              7.6      11.0   16.1   7.2
greedy initialization heuristic           3             3           0           1         1.000              7.9      9.6    16.2   6.6
chunk overlap stride calculation          111           120         11          6         1.000              7.5      10.6   16.2   8.2
custom error type with thiserror          1             1           0           3         1.000              7.7      10.3   16.3   7.7
bm25 idf inverse document frequency       10            12          7           7         1.000              6.8      10.1   17.1   8.4
hash unique tokens for bloom filter       38            41          4           8         1.000              7.3      10.4   16.9   8.2
search pipeline orchestrator entry point  6             6           4           8         1.000              7.6      10.7   16.0   7.4
metropolis acceptance criterion           0             0           0           4         1.000              7.0      9.9    16.2   7.2
random seed for reproducible solver       0             0           0           7         1.000              7.4      9.8    16.0   8.0
line based file chunking                  37            40          0           5         1.000              8.0      11.2   17.2   8.2
bm25 top n scoring with candidates        10            12          7           7         1.000              8.1      10.8   17.8   8.4
relevance score in qubo formulation       10            10          2           9         1.000              8.3      10.7   15.9   7.8
minhash signature with k hashes           10            12          6           7         1.000              7.1      9.5    15.9   8.6
clap derive parser for cli                2             2           2           5         1.000              6.9      11.1   15.8   8.0
anyhow context for error chain            2             2           2           10        1.000              7.0      10.3   15.9   8.5
search result struct definition           6             6           4           8         1.000              6.6      10.4   16.6   7.5
```

Medianas agregadas sobre las 25 queries (ms):

| Herramienta       | Mediana (ms) | Rango     | Observación                                          |
|-------------------|--------------|-----------|------------------------------------------------------|
| `grep -rnE`       | ~7.6         | 6.6–9.1   | Búsqueda literal del primer token (BSD/macOS)        |
| `rg`              | ~10.7        | 9.4–11.7  | Ripgrep 15.1.0; el paralelismo no amortiza en 12 archivos |
| `sg run -l rust`  | ~16.6        | 15.3–18.6 | Matcher estructural ast-grep 0.42.1                  |
| `rpgrep search`   | ~7.9         | 6.6–9.0   | Pipeline completo Xor+BM25+MinHash+QUBO              |

> **Lectura honesta**: en un corpus de 12 archivos `rpgrep` empata con
> `grep` (~8 ms) mientras devuelve un bundle ranqueado de 5–10 chunks
> bajo budget, no líneas planas. La paridad esperada con `grep` sube en
> corpus grandes (el gate §5.2 muestra ~52 ms P95 sobre 100k chunks);
> `grep` escalará linealmente con el tamaño del repo, `rpgrep` con el
> tamaño del top-k tras el descarte Xor → la ventaja relativa de
> `rpgrep` aparece con más archivos, no con menos.
>
> **Caveat sobre `rg_matches` vs `grep_matches`**: `rg --count-matches`
> reporta nº total de matches (varios por línea cuentan), `grep -rnE | wc -l`
> cuenta líneas distintas con al menos un match. Por eso `rg_matches ≥
> grep_matches` (ej. `bm25` 12 vs 10, `chunk` 120 vs 111). Ambas
> métricas son válidas en su definición, no son intercambiables.

> `rpgrep_top1_score=1.000` en todas las filas: BM25 se normaliza
> dividiendo por el máximo del batch, por construcción el top-1 satura el
> rango. No es un sesgo, es la definición del normalizador (ver
> [src/search/pipeline.rs](../src/search/pipeline.rs)).

> Bugs corregidos en `scripts/bench_compare.sh` durante esta ejecución:
> (1) `$INDEX_DIR…` con ellipsis Unicode pegada → `${INDEX_DIR}…` con
> braces (set -u rechazaba el identificador compuesto); (2) heredoc
> Python con interpolación `r'''$cmd'''` rompía si `$cmd` terminaba en
> `'…'` → migrado a heredoc quoted + variables de entorno `CMD`/`ITERS`.

**Caveat sobre la comparación**: `grep`/`rg`/`sg` reciben el **primer token**
de cada query (no son semánticos: no entienden frases). Comparar
`grep_matches` con `rpgrep_n` mide ortogonalmente: matches literales vs.
tamaño del bundle semántico. No es una comparación de "calidad", es de
**comportamiento**.

---

## 6. Limitaciones conocidas

### 6.1 R6 — Comando `Index` cableado (RESUELTO)
El subcomando `rpgrep index` ahora construye el índice directamente vía
`IndexStore::from_dir`. El antiguo workaround
[examples/build_demo_index.rs](../examples/build_demo_index.rs) se mantiene
como ejemplo programático mínimo, pero ya no es necesario para el harness.

### 6.2 Corpus sintético textual en benches y gate
[benches/pipeline.rs](../benches/pipeline.rs) y
[tests/p95_gate.rs](../tests/p95_gate.rs) generan chunks de **texto
sintético reproducible** (tokens `var_unique_X`, `handler_shared`,
`compute_Y_Z`, `filler_R`) con seed fija. BM25 y MinHash se construyen
sobre ese texto exactamente como en producción: el gate mide el pipeline
end-to-end real, no una simulación.

### 6.3 Sin red (RESUELTO)
Tras la eliminación de `fastembed`/MiniLM, **ninguna parte de la suite
requiere red, ni descarga ni cache de modelos**. El test de calidad
(Capa B) ya no está marcado `#[ignore]`.

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
Esta suite **no** valida tree-sitter chunking, modo watch, modo serve, ni
migración a `rkyv`. Cuando esas features aterricen, añadir tests
específicos; no antes.

---

## Apéndice — Resumen del árbol de archivos de la suite

```
Cargo.toml                          dev-deps de validación (criterion, rand_chacha, [[bench]])
tests/pipeline_invariants.rs        Capa A — 5 tests offline (R2/R3/R4/R5)
tests/semantic_quality.rs           Capa B — Recall@K, MRR, Diversity@K (sin red)
tests/p95_gate.rs                   Capa C — gate P95 < 150 ms @ 100k [ignore]
tests/fixtures/golden_corpus.tsv    25 pares query↔relevantes
benches/pipeline.rs                 Capa C — Criterion (xor / bm25 / qubo / e2e)
examples/build_demo_index.rs        Ejemplo programático de IndexStore::from_dir
scripts/bench_compare.sh            Capa D — harness comparativo
docs/VALIDATION.md                  Este documento
```

La validación NO contamina producción.
