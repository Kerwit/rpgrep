# rpgrep

Búsqueda probabilística de código sin modelos de lenguaje
(**Xor filter → BM25 → MinHash → QUBO + Simulated Annealing**).

Sustituto de `grep` que produce un *bundle* de contexto óptimo bajo budget
de tokens: relevante, diverso y matemáticamente puro. El término final del
pipeline es un Hamiltoniano de Ising — exactamente lo que un p-bit /
annealer cuántico resolvería físicamente, aquí ejecutado sobre CPU con
muestreo Metropolis. Cero hardware especial, cero pesos pre-entrenados,
cero descargas, cero red.

## Pipeline

```
Query del usuario
   │
   ▼
[A] Xor filter pre-screen          ~0.1  ms   cero falsos negativos
   │
   ▼
[B] BM25 scoring (rᵢ)              ~1-10 ms   relevancia probabilística
   │
   ▼
[C] MinHash Jaccard (sᵢⱼ)          ~1-5  ms   redundancia estimada
   │
   ▼
[D] QUBO + Simulated Annealing     ~10-30 ms  selección óptima bajo budget
   │
   ▼
Contexto óptimo  (relevante + diverso + dentro de presupuesto)
```

Latencia P95 objetivo: **<150 ms** end-to-end sobre 100k chunks.

## Uso

### CLI

```bash
# Indexar un directorio (por defecto: archivos .rs)
rpgrep index ./mi-proyecto --out .rpgrep

# Indexar varias extensiones
rpgrep index ./mi-proyecto --ext rs,md,toml --out .rpgrep

# Buscar con presupuesto de 4000 tokens
rpgrep search "manejo de errores en conexiones" --budget 4000

# Estadísticas del índice
rpgrep stats --index .rpgrep
```

### Como crate

```toml
[dependencies]
rpgrep = { path = "../rpgrep" }
```

```rust
use rpgrep::SearchPipeline;

let pipeline = SearchPipeline::load(".rpgrep".as_ref())?;
let results = pipeline.search("query", 4000, 50)?;
for r in results {
    println!("{}:{}-{}  {:.3}", r.chunk.file.display(),
             r.chunk.start_line, r.chunk.end_line, r.score);
}
```

## Arquitectura

| Módulo                 | Responsabilidad                                       |
|------------------------|-------------------------------------------------------|
| `chunk/`               | Segmentación line-based con solapamiento, IDs estables|
| `index/bloom.rs`       | Xor filter por archivo (Graf & Lemire 2020)           |
| `index/bm25.rs`        | BM25 puro Rust (Robertson 1994) — provee `rᵢ`         |
| `index/minhash.rs`     | MinHash signatures (Broder 1997) — provee `sᵢⱼ`       |
| `index/store.rs`       | Persistencia rkyv + mmap (`RPGRP003`) + `IndexStore::from_dir` |
| `search/qubo.rs`       | Simulated Annealing puro Rust (Metropolis)            |
| `search/pipeline.rs`   | Orquestador A→B→C→D                                   |

Cero crates de ML, cero ONNX runtime, cero archivos de modelo. Solo
matemática clásica: hashing aleatorizado, modelo probabilístico de
relevancia, estimador insesgado de Jaccard, optimización combinatoria
vía relajación térmica simulada.

## Estado: v0.1 — funcional end-to-end

**Implementado y testeado:**
- ✅ QUBO + Simulated Annealing puro Rust con seed fija (R2)
- ✅ Xor filter por archivo con test de zero-false-negative (R3)
- ✅ BM25 con tests de IDF, normalización por longitud, top-N filtrado
- ✅ MinHash con tests de identidad, disjunción, error estadístico
- ✅ Chunking por líneas con solapamiento e IDs estables (R4)
- ✅ Persistencia rkyv + memmap2 (`RPGRP003`); load ~1.35× más rápido que bincode @ 10k chunks
- ✅ CLI con `clap`: `index` / `search` / `stats` totalmente cableados
- ✅ Pipeline orquestador completo (Xor → BM25 → MinHash → QUBO)
- ✅ Test de calidad sobre corpus dorado: **MRR=1.000, Recall@5=0.37**

**Hoja de ruta v0.2:**
- AST-aware chunking con `tree-sitter`
- Modo `watch` con `notify` (re-indexación incremental)
- Modo `serve` con Unix socket
- ⏳ Zero-copy real: `IndexStore::load_archived() -> &ArchivedIndexStore`
  (refactor de `SearchPipeline` para operar sobre tipos archivados;
  el formato `RPGRP003` ya soporta esto, solo falta el frontal)

## Verificación rápida

```bash
cargo check                                   # compilación
cargo test --lib                              # tests unitarios (BM25, MinHash, QUBO, Bloom, Chunk)
cargo test --test pipeline_invariants         # invariantes (Capa A)
cargo test --test semantic_quality            # calidad sobre corpus dorado (Capa B)
cargo test --release --test p95_gate -- --ignored  # gate de latencia (Capa C)
```

## Bibliografía

- Graf & Lemire 2020 — *Xor filters: Faster and smaller than Bloom filters*
- Robertson & Walker 1994 — *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval* (BM25)
- Broder 1997 — *On the resemblance and containment of documents* (MinHash)
- Kirkpatrick, Gelatt, Vecchi 1983 — *Optimization by Simulated Annealing*

## Licencia

MIT OR Apache-2.0
