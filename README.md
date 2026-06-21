# rpgrep

Búsqueda probabilística de código sin modelos de lenguaje
(**Xor filter → BM25 → MinHash → QUBO + Simulated Annealing**).

Sustituto de `grep` que produce un *bundle* de contexto óptimo bajo budget
de tokens: relevante, diverso y matemáticamente puro. El término final del
pipeline es un Hamiltoniano de Ising — exactamente lo que un p-bit /
annealer cuántico resolvería físicamente, aquí ejecutado sobre CPU con
muestreo Metropolis. Cero hardware especial, cero pesos pre-entrenados,
cero descargas, cero red.

**Estado actual: v0.2.6 — catálogo extendido de lenguajes (8 con AST chunking).**

## Pipeline

```
Query del usuario
   │
   ▼
[A] Xor filter pre-screen (archived) ~0.1  ms   cero falsos negativos
   │
   ▼
[B] BM25 scoring (rᵢ) (archived)     ~1-10 ms   relevancia probabilística
   │
   ▼
[C] MinHash Jaccard (sᵢⱼ) (archived) ~1-5  ms   redundancia estimada
   │
   ▼
[D] QUBO + Simulated Annealing       ~10-30 ms  selección óptima bajo budget
   │
   ▼
Contexto óptimo  (relevante + diverso + dentro de presupuesto)
```

**Latencia P95 medida**: 53.8 ms end-to-end @ 100k chunks (gate 150 ms).
**Load zero-copy**: 62 µs @ 1k chunks, 455 µs @ 10k chunks (mmap directo
sobre `&ArchivedPayload`, 6.9×–13.6× sobre `IndexStore::load` owned).

## Uso

### CLI

```bash
# Indexar un directorio (default: .rs). AST chunking automático
# para Rust/Python/JavaScript/TypeScript/TSX/Dart/C/Go; line-based fallback
# para el resto. Error si la ruta no es un directorio existente.
rpgrep index ./mi-proyecto --out .rpgrep

# Indexar varias extensiones
rpgrep index ./mi-proyecto --ext rs,py,js,ts,tsx,c,dart,go,md --out .rpgrep

# Buscar con presupuesto de 4000 tokens
rpgrep search "manejo de errores en conexiones" --budget 4000

# Estadísticas del índice
rpgrep stats --index .rpgrep

# Re-indexar automáticamente ante cambios (debounce 500 ms default)
rpgrep watch ./mi-proyecto --out .rpgrep --debounce-ms 500

# Servidor de queries vía Unix socket JSON-line (sesiones largas)
rpgrep serve --index .rpgrep
echo '{"query":"validate user input","budget":4000,"topk":5}' \
  | nc -U .rpgrep/rpgrep.sock
```

### Como crate

```toml
[dependencies]
rpgrep = { path = "../rpgrep" }
```

```rust
use rpgrep::SearchPipeline;

// Carga zero-copy real desde mmap. No deserializa.
let pipeline = SearchPipeline::load(".rpgrep".as_ref())?;
let results = pipeline.search("query", 4000, 50)?;
for r in results {
    println!(
        "{}:{}-{}  {:.3}",
        r.chunk.file, r.chunk.start_line, r.chunk.end_line, r.score
    );
}
```

## Arquitectura

| Módulo | Responsabilidad |
|---|---|
| `chunk/` | **AST chunking** vía tree-sitter (Rust/Python/JS/TS/TSX/Dart/C/Go) + fallback line-based con solapamiento; IDs estables `hash(path + start_line)` |
| `index/bloom.rs` | Xor filter por archivo (Graf & Lemire 2020) + `xor_contains_archived` (zero-copy) |
| `index/bm25.rs` | BM25 puro Rust (Robertson 1994) — provee `rᵢ`; `top_n_archived` opera sobre `&ArchivedBm25Index` |
| `index/minhash.rs` | MinHash signatures (Broder 1997) — provee `sᵢⱼ`; `archived_jaccard` sobre slices mmap |
| `index/store.rs` | `IndexStore` owned (build) + **`MmappedStore` zero-copy** (queries). Formato `RPGRP003` (rkyv + mmap) |
| `search/qubo.rs` | Simulated Annealing puro Rust (Metropolis, seed fija `0xC0DEF00D`) |
| `search/pipeline.rs` | Orquestador A→B→C→D operando 100% sobre `&ArchivedPayload`; materializa `Chunk` owned solo al construir los `SearchResult` finales |
| `serve.rs` | `UnixListener` thread-per-conn + `Arc<SearchPipeline>` compartido. Protocolo JSON-line |
| `cli.rs` | Subcomandos `index` / `search` / `stats` / `watch` / `serve` / `version` (+ flag `--version`/`-V`) |

Cero crates de ML, cero ONNX runtime, cero archivos de modelo, cero red
ni en build ni en runtime. Solo matemática clásica: hashing
aleatorizado, modelo probabilístico de relevancia, estimador insesgado
de Jaccard, optimización combinatoria vía relajación térmica simulada.

**Excepción explícita** (`BLUEPRINT.md` §R1): tree-sitter — parsers
determinísticos generados desde DSL, no contienen pesos ni se entrenan,
runtime C ligero (~50 KB). Cero modelos pre-entrenados.

## Comparativa con otras herramientas

| Dimensión | `grep` | `ripgrep` | `ast-grep` | **`rpgrep`** |
|---|---|---|---|---|
| Categoría | Scanner texto | Scanner texto turbo | Scanner sintáctico | **Retrieval probabilístico** |
| P95 query @ 100k chunks | ~250–1000 ms | ~30–100 ms | ~100–300 ms | **53.8 ms (índice precalc.)** |
| Ranking por relevancia | No | No | No | **Sí (BM25 + QUBO)** |
| Diversificación entre resultados | No | No | Parcial | **Sí (MinHash + QUBO; Div@5=0.92)** |
| Chunks sintácticos completos | No | No | Sí | **Sí (AST + fallback)** |
| Top-N controlado + budget tokens | No | No | No | **Sí** |
| Regex completo | Sí | Sí | Sí (en patrones) | No (tokenizado) |
| Patrones estructurales AST | No | No | **Sí** | Parcial (AST chunks, query no) |
| Refactor / reemplazo | `-r` | `--replace` | **`--rewrite` AST-safe** | No (solo retrieval) |
| Filtros `.gitignore` | No | **Sí (nativo)** | Sí | Configurable por extensión |
| Servidor de queries / modo watch | No | No | No | **Sí (`serve` + `watch`)** |
| Modelos pre-entrenados / red | No | No | No | **No (puro probabilístico)** |
| Determinismo bit a bit | Sí | Sí | Sí | Sí (seed fija) |

Las cuatro son complementarias, no sustitutas. En un agente serio
típicamente conviven **ripgrep + rpgrep** (scan rápido + retrieval
rankeado) o **ast-grep + rpgrep** (refactor estructural + contexto LLM).

Ver `docs/LLM_INTEGRATION.md` para la política operativa de decisión
en agentes (LLM directo > rpgrep > ast-grep > rg > grep).

## Estado del proyecto

**v0.2.6 — implementado y testeado:**

- **AST chunking con tree-sitter, 8 lenguajes** (Rust/Python/JS/TS/TSX/Dart/C/Go), fallback line-based
- `index` falla con error claro si la ruta no es un directorio existente (antes: "0 chunks" silencioso)
- QUBO + Simulated Annealing puro Rust con seed fija (R2)
- Xor filter por archivo con tests de zero-false-negative (R3)
- BM25 con tests de IDF, normalización por longitud, top-N filtrado
- MinHash con tests de identidad, disjunción, error estadístico
- IDs estables `hash(path + start_line)` (R4)
- Persistencia rkyv + memmap2, formato `RPGRP003`
- **Zero-copy real** (`MmappedStore` + `&ArchivedPayload`): pipeline
  opera sobre archived sin deserializar; Speedup load 6.9× @ 1k, 13.6× @ 10k
- CLI con `clap`: `index` / `search` / `stats` / `watch` / `serve` / `version` (+ flag `--version`/`-V`)
- Modo `watch` con `notify-debouncer-mini` (re-index ante cambios)
- Modo `serve` con Unix socket JSON-line (thread-per-conn)
- Test de calidad sobre corpus dorado: **Recall@5=0.30 / MRR=0.94 / Diversity@5=0.92**
- Gate P95 latencia: **53.8 ms @ 100k chunks** (margen 2.8× sobre 150 ms)

**Roadmap** (⏳ documentado en `docs/LANGUAGES_ROADMAP.md`):

- ⏳ Refactor a `LangSpec` registry + feature flags por lenguaje
- ⏳ Tier 1 restante: C++, Java (~80% cobertura GitHub)
- ⏳ Tier 2 opcional: Ruby, Kotlin, Swift, PHP, C#

**Deuda heredada** (priorizar bajo petición explícita):

- `watch` hace re-index completo (no incremental)
- `serve` sin hot-reload (requiere reiniciar tras re-index)

## Verificación rápida

```bash
cargo check                                        # compilación
cargo test --lib                                   # tests unitarios
cargo test --test pipeline_invariants              # invariantes (Capa A)
cargo test --test semantic_quality                 # calidad corpus dorado (Capa B)
cargo test --release --test p95_gate -- --ignored  # gate latencia (Capa C)
cargo bench --bench pipeline -- load               # speedup zero-copy vs owned
```

## Documentación

- `BLUEPRINT.md` — Fuente de verdad arquitectónica. Restricciones
  (R1–R9), excepción explícita para tree-sitter, roadmap v0.2.x.
- `docs/VALIDATION.md` — Metodología de calidad: corpus, métricas,
  reproducir, resultados, limitaciones.
- `docs/LLM_INTEGRATION.md` — Política operativa de decisión para
  agentes LLM: árbol de decisión, tabla señal→herramienta, helper
  shell, implementación Python, modo servidor, checklist.
- `docs/LANGUAGES_ROADMAP.md` — Plan v0.2.6: registry `LangSpec` +
  feature flags + tabla `top_level_node_kinds` para 14 lenguajes.
- `SUMMARIES.md` — Árbol de resúmenes por archivo (evita lecturas
  ciegas del repo).

## Bibliografía

- Graf & Lemire 2020 — *Xor filters: Faster and smaller than Bloom filters*
- Robertson & Walker 1994 — *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval* (BM25)
- Broder 1997 — *On the resemblance and containment of documents* (MinHash)
- Kirkpatrick, Gelatt, Vecchi 1983 — *Optimization by Simulated Annealing*

## Licencia

MIT OR Apache-2.0
