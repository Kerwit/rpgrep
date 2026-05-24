# rpgrep

Búsqueda semántica de código mediante pipeline probabilístico clásico
(**Bloom/Xor → HNSW → QUBO + Simulated Annealing**).

Sustituto de `grep` que comprende significado en lugar de coincidencias léxicas.
Equivalente al Hamiltoniano de Ising que resolvería un p-bit físico, ejecutado
sobre CPU mediante muestreo Metropolis. Cero hardware especial.

## Pipeline

```
Query del usuario
   │
   ▼
[A] Xor filter pre-screen          ~0.1  ms   (cero falsos negativos)
   │
   ▼
[B] HNSW retrieval (top-K)         ~10-30 ms  (ANN aproximado)
   │
   ▼
[D] QUBO + Simulated Annealing     ~30   ms   (selección óptima bajo budget)
   │
   ▼
Contexto óptimo  (relevante + diverso + dentro de presupuesto)
```

Latencia P95 objetivo: **<150 ms** end-to-end sobre 100k chunks.

## Uso

### CLI

```bash
# Indexar un directorio
rpgrep index ./mi-proyecto --out .rpgrep

# Buscar con presupuesto de 4000 tokens
rpgrep search "manejo de errores en conexiones HTTP" --budget 4000

# Estadísticas del índice
rpgrep stats
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

| Módulo                 | Responsabilidad                                |
|------------------------|------------------------------------------------|
| `chunk/`               | Segmentación line-based con solapamiento       |
| `embed/`               | Embeddings vía `fastembed` (MiniLM L6 v2)      |
| `index/bloom.rs`       | Xor filter por archivo (Graf & Lemire 2020)    |
| `index/hnsw.rs`        | ANN con `instant-distance`, distancia coseno   |
| `index/store.rs`       | Persistencia con `bincode`                     |
| `search/qubo.rs`       | Simulated Annealing puro Rust (Metropolis)     |
| `search/pipeline.rs`   | Orquestador A→B→D                              |

## Estado: v0.1 — scaffold

**Implementado y funcional:**
- ✅ QUBO + Simulated Annealing con tests (incluye test de diversidad)
- ✅ Xor filter por archivo con test de zero-false-negative
- ✅ Chunking por líneas con solapamiento y test
- ✅ Persistencia con bincode
- ✅ Estructura CLI completa con `clap`
- ✅ Pipeline orquestador completo

**Pendiente de cablear** (sin trabajo conceptual, solo glue):
- ⏳ `Commands::Index` en `src/cli.rs` — el TODO marcado describe los pasos exactos
- ⏳ Verificar versiones actuales de `fastembed` e `instant-distance` (APIs evolucionan)
- ⏳ Tests de integración end-to-end

**Hoja de ruta v0.2:**
- Cross-encoder re-ranking (paso [C] del pipeline)
- AST-aware chunking con `tree-sitter`
- Modo `watch` con `notify` (re-indexación incremental)
- Modo `serve` con Unix socket (integración con Sidecar Kerwit)
- Migración a `rkyv` + `memmap2` para carga zero-copy
- Sustituir similitud por trigramas con similitud coseno de embeddings persistidos

## Verificación rápida

```bash
cd rpgrep
cargo check                                # compilación
cargo test --lib search::qubo              # tests del solver QUBO
cargo test --lib index::bloom              # tests del Xor filter
```

## Bibliografía

- Indyk & Motwani 1998 — *Approximate nearest neighbors via LSH*
- Malkov & Yashunin 2016 — *HNSW: Efficient and robust ANN*
- Jégou, Douze, Schmid 2011 — *Product Quantization*
- Graf & Lemire 2020 — *Xor filters: Faster and smaller than Bloom filters*
- Kirkpatrick, Gelatt, Vecchi 1983 — *Optimization by Simulated Annealing*

## Licencia

MIT OR Apache-2.0
