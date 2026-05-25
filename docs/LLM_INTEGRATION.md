# Integración con agentes LLM — política de decisión

> **Propósito**: guía operativa para agentes (Claude Code, Cursor, Continue,
> scripts custom) que necesiten elegir entre `rpgrep`, `ripgrep`/`grep`
> nativo, `ast-grep`, o resolver la query **directamente en el LLM** sin
> shell out.
> Referencia cruzada: `BLUEPRINT.md` §R7 (roadmap), `SUMMARIES.md`
> "Servidor de queries" (`rpgrep serve`).

## 1. Principio rector

> **Cada token enviado al LLM cuesta dinero y latencia. Cada query
> innecesaria al disco cuesta tiempo. Elegir la herramienta correcta
> en cada paso minimiza ambas.**

La política se reduce a **una pregunta antes de toda búsqueda**:

> *¿El contenido que necesito ya está en el contexto del LLM?*

Si la respuesta es **sí**, no se invoca ninguna herramienta. Esto es el
primer ahorro y aplica a cualquier integración, no sólo `rpgrep`.

## 2. Árbol de decisión (orden estricto)

```
1. ¿El contenido ya está en contexto del LLM?
   Sí → LLM directo (sin shell out)
   No ↓

2. ¿Existe índice rpgrep (.rpgrep/rpgrep.idx) Y rpgrep está instalado?
   Sí → rpgrep (lookup en índice mmap, P95 ~50 ms @ 100k chunks)
   No ↓

3. ¿La query es estructural (patrón AST con $X, $_, $$$)?
   Sí Y existe ast-grep → sg
   No ↓

4. ¿La query es semántica Y el repo > 5000 chunks estimados
   Y se prevén ≥2 queries en sesión?
   Sí Y existe rpgrep → build on-demand (`rpgrep index .`) + rpgrep
   No ↓

5. ¿ripgrep instalado?
   Sí → rg
   No → grep (siempre presente en Unix)
```

## 3. Tabla de matching: señal → herramienta

| Señal observable | Herramienta óptima | Razón |
|---|---|---|
| Archivo ya en contexto del LLM | **LLM directo** | Shell out gasta tokens y latencia sin ganancia |
| `.rpgrep/rpgrep.idx` existe | **rpgrep** | Índice ya pagado; P95 ~50 ms |
| Query con `\b`, `(?:...)`, `^`, `$`, regex escapes | **rg** (fallback grep) | rpgrep tokeniza, no parsea regex |
| Query estructural (`fn $X() -> Result<_>`) | **ast-grep** | Único con patrones AST de query |
| Query natural ("código que valida JWT") | **rpgrep** | BM25 + QUBO rankea y diversifica |
| Repo < 200 archivos | **rg / grep** | Build de índice no amortiza |
| Repo > 10k chunks Y >2 queries previstas | **rpgrep** (build si falta) | Build ~1.3 s amortiza tras 2 queries |
| One-shot en repo grande | **rg** | Build no compensa para una sola query |
| Agente sin filesystem (sandbox) | **LLM directo** | No hay shell disponible |
| CI con throughput máximo | **rg** | Sin warm-up, escala lineal |

## 4. Detección de disponibilidad (shell)

Helper portable para invocar desde un agente o pre-hook:

```bash
choose_search_tool() {
    local repo_root="${1:-.}"
    local query_type="${2:-lexical}"  # lexical | semantic | regex | ast

    # 0. ¿El LLM tiene el archivo? — decisión del agente, no del shell.
    #    El agente DEBE responder esto antes de invocar este helper.

    # 1. Índice rpgrep precalculado
    if [ -f "$repo_root/.rpgrep/rpgrep.idx" ] && command -v rpgrep >/dev/null 2>&1; then
        echo "rpgrep --index $repo_root/.rpgrep"
        return
    fi

    # 2. AST patterns explícitos
    if [ "$query_type" = "ast" ] && command -v sg >/dev/null 2>&1; then
        echo "sg"
        return
    fi

    # 3. Query semántica + repo grande justifica build on-demand
    local n_files
    n_files=$(find "$repo_root" -type f \( -name '*.rs' -o -name '*.py' -o -name '*.js' \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$query_type" = "semantic" ] && [ "$n_files" -gt 200 ] && command -v rpgrep >/dev/null 2>&1; then
        echo "rpgrep-onthefly"  # el caller decide si construir índice o no
        return
    fi

    # 4. Regex / lexical → throughput crudo
    if command -v rg >/dev/null 2>&1; then
        echo "rg"
        return
    fi

    # 5. Último recurso, siempre presente en Unix
    echo "grep"
}
```

## 5. Caso especial: "LLM directo" — cuándo NO buscar

El agente debe **saltarse toda búsqueda** si se da cualquiera de:

1. **El archivo o snippet ya está en el contexto** (cargado por una
   herramienta previa o adjunto por el usuario).
2. **El usuario pegó el código en la query** ("revisa este fragmento: …").
3. **La query es sobre algo ya discutido en la conversación** (e.g.
   "el helper que mencionaste antes").

**Señal de detección recomendada**: el agente lleva un *set* de
identifiers ya vistos en la sesión (paths, símbolos, función names).
Antes de invocar `grep`/`rpgrep`, hace lookup en ese set. Si hit,
responde de memoria.

Esta optimización vale para cualquier integración LLM + filesystem.
Es el primer ahorro de tokens y muchas veces el más grande.

## 6. Heurística build-on-demand para rpgrep

Cuándo merece ejecutar `rpgrep index .` justo antes de la primera query:

| Condición | Decisión |
|---|---|
| `n_chunks_estimados > 5000` Y `queries_previstas >= 2` | **Build** (~1.3 s amortiza tras 2 queries) |
| `n_chunks_estimados > 50000` Y query es semántica | **Build** aunque sea 1 query (rg dará 100k matches sin rank) |
| `n_chunks_estimados < 1000` | **NO build**, `rg` directo |
| Repo cambia frecuente (iteración rápida del agente) | **NO build estático**; considera `rpgrep watch` si la sesión va a ser larga |

**Estimación rápida de chunks sin índice**:
```bash
find . \( -name '*.rs' -o -name '*.py' -o -name '*.js' \) | wc -l
# multiplicar por ~10 chunks/archivo (regla de oro AST chunking).
```

## 7. Implementación recomendada (Python / pseudocódigo)

Función única que el agente llama antes de **cualquier** búsqueda:

```python
from pathlib import Path

def search(query: str, repo: Path, agent_ctx: AgentContext) -> list[Chunk]:
    # 0. Cache de contexto LLM (primer corte siempre)
    if hit := agent_ctx.lookup_loaded_files(query):
        return hit  # ya en contexto, no shell out

    # 1. Clasificar query
    qtype = classify_query(query)
    # → 'regex' | 'ast' | 'lexical' | 'semantic'

    # 2. Política
    if has_rpgrep_index(repo) and qtype in ('lexical', 'semantic'):
        return run_rpgrep(repo, query)

    if qtype == 'ast' and has_tool('sg'):
        return run_sg(repo, query)

    if qtype == 'semantic' and estimate_chunks(repo) > 5000 and has_tool('rpgrep'):
        build_rpgrep_index(repo)   # ~1.3 s @ 100k chunks
        return run_rpgrep(repo, query)

    if has_tool('rg'):
        return run_rg(repo, query)

    return run_grep(repo, query)
```

### Clasificador de query (heurística simple)

```python
import re

REGEX_METACHARS = re.compile(r'[\\^$.|?*+(){}\[\]]')
AST_PATTERN     = re.compile(r'\$[A-Z_][A-Z0-9_]*|\$\$\$')

def classify_query(q: str) -> str:
    if AST_PATTERN.search(q):
        return 'ast'
    if REGEX_METACHARS.search(q):
        return 'regex'
    # heurística semántica: >3 palabras sin metachars sugiere lenguaje natural
    if len(q.split()) > 3:
        return 'semantic'
    return 'lexical'
```

### Wrappers (esqueleto)

```python
def run_rpgrep(repo: Path, query: str, topk: int = 5) -> list[Chunk]:
    # Vía CLI:
    #   rpgrep search --index <repo>/.rpgrep --query "<q>" --topk <k>
    # O vía socket (rpgrep serve, JSON-line):
    #   {"query": "<q>", "budget": 4000, "topk": <k>}
    ...

def run_rg(repo: Path, query: str, max_per_file: int = 3) -> list[Chunk]:
    # rg --json -m <max_per_file> --max-count <N> -- "<q>" <repo>
    # Parsear JSON, agrupar por archivo.
    ...

def run_grep(repo: Path, query: str) -> list[Chunk]:
    # grep -rn --include='*.rs' --include='*.py' --include='*.js' -- "<q>" <repo>
    ...
```

## 8. Modo servidor para sesiones largas

Para agentes que hacen muchas queries sobre el mismo repo:

```bash
# Una sola vez al inicio de la sesión:
rpgrep serve --index .rpgrep --socket /tmp/rpgrep.sock &

# Cada query del agente envía un JSON por la línea:
echo '{"query":"validate user input","budget":4000,"topk":5}' \
  | nc -U /tmp/rpgrep.sock
```

Ventajas vs CLI repetida:
- Sin coste de re-`MmappedStore::open()` por query (~455 µs @ 10k chunks → ~0 amortizado).
- `Arc<SearchPipeline>` compartido entre conexiones → cero sync overhead.
- Reinicio explícito tras re-index (hot-reload deferred, ver `BLUEPRINT.md` §R7).

## 9. Resumen ejecutivo

| Caso | Herramienta | Por qué |
|---|---|---|
| Contenido en contexto LLM | **LLM directo** | Ahorro de tokens y latencia |
| Índice existe / repo grande con queries recurrentes | **rpgrep** | Top-N rankeado y diversificado |
| Regex / patrón exacto / one-shot | **rg** | Throughput crudo |
| Patrón estructural AST | **ast-grep** | Único con AST queries |
| Sistema mínimo sin herramientas | **grep** | Fallback universal |

El truco no está en "cuál es mejor en abstracto" sino en **detectar la
señal correcta y caer al fallback apropiado** sin que el agente
genere queries vacías o redundantes. La política completa cabe en un
wrapper de ~50 líneas (bash o Python) reutilizable entre agentes.

## 10. Checklist de integración

Para conectar `rpgrep` a un agente nuevo:

1. ☐ Decidir dónde vive el helper de decisión (`scripts/`, módulo
   Python, wrapper bash en `PATH`).
2. ☐ Implementar `lookup_loaded_files` en el agente (cache de paths
   y símbolos vistos en la sesión).
3. ☐ Configurar `rpgrep watch` si la sesión es larga y el repo cambia
   (debounce 500 ms es default razonable).
4. ☐ Si se usa `serve`, abrir el socket al arrancar y cerrarlo en
   `SIGINT/SIGTERM` del agente.
5. ☐ Manejar errores: si `rpgrep` falla o el socket está caído, caer
   a `rg` sin abortar la query del usuario.
6. ☐ Loggear qué herramienta resolvió cada query (telemetría útil
   para ajustar la política).
