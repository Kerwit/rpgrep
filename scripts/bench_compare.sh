#!/usr/bin/env bash
# Harness comparativo: rpgrep vs grep vs ripgrep (rg) vs ast-grep (sg).
# Capa D del Súper Prompt de validación.
#
# - Cada herramienta mide latencia (mediana de N iteraciones) y nº de matches/results
#   sobre las mismas queries del corpus dorado (tests/fixtures/golden_corpus.tsv).
# - Si rg o sg faltan, su columna emite WARN y se omite (no aborta el script).
# - Requiere python3 (incluido por defecto en macOS) para medición sub-ms portable.
#
# Uso:
#   ./scripts/bench_compare.sh                  # corpus=src, índice=.rpgrep-bench
#   CORPUS=mnt ./scripts/bench_compare.sh       # override corpus
#   ITERATIONS=10 ./scripts/bench_compare.sh    # más muestras por query

set -uo pipefail

CORPUS="${CORPUS:-src}"
GOLDEN="${GOLDEN:-tests/fixtures/golden_corpus.tsv}"
INDEX_DIR="${INDEX_DIR:-.rpgrep-bench}"
ITERATIONS="${ITERATIONS:-5}"

# ---- detección de herramientas --------------------------------------------
HAS_RG=0; command -v rg >/dev/null 2>&1 && HAS_RG=1
HAS_SG=0; command -v sg >/dev/null 2>&1 && HAS_SG=1
HAS_PY=0; command -v python3 >/dev/null 2>&1 && HAS_PY=1
HAS_CARGO=0; command -v cargo >/dev/null 2>&1 && HAS_CARGO=1

if [ "$HAS_PY" -eq 0 ]; then
    echo "ERROR: python3 no encontrado; necesario para medir latencia sub-ms portable." >&2
    exit 1
fi
if [ "$HAS_CARGO" -eq 0 ]; then
    echo "ERROR: cargo no encontrado." >&2
    exit 1
fi
if [ ! -f "$GOLDEN" ]; then
    echo "ERROR: corpus dorado no encontrado en $GOLDEN" >&2
    exit 1
fi
if [ ! -d "$CORPUS" ]; then
    echo "ERROR: corpus de búsqueda no encontrado en $CORPUS" >&2
    exit 1
fi

[ "$HAS_RG" -eq 0 ] && echo "WARN: rg (ripgrep) no encontrado — columna rg_* será WARN." >&2
[ "$HAS_SG" -eq 0 ] && echo "WARN: sg (ast-grep) no encontrado — columna sg_* será WARN." >&2

# ---- build rpgrep ---------------------------------------------------------
RPGREP_BIN="target/release/rpgrep"
if [ ! -x "$RPGREP_BIN" ]; then
    echo "[bench_compare] compilando rpgrep en release…" >&2
    cargo build --release --quiet
fi

# ---- build índice demo (workaround R6) ------------------------------------
if [ ! -f "$INDEX_DIR/rpgrep.idx" ]; then
    echo "[bench_compare] construyendo índice demo en $INDEX_DIR (descarga MiniLM la 1ª vez)…" >&2
    cargo run --release --example build_demo_index --quiet -- "$CORPUS" "$INDEX_DIR"
fi

# ---- helpers ---------------------------------------------------------------
# Ejecuta un comando shell N veces y devuelve mediana en ms (1 decimal).
measure_ms() {
    local cmd="$1"
    python3 - <<PY
import subprocess, time, statistics
samples = []
for _ in range($ITERATIONS):
    t0 = time.perf_counter()
    subprocess.run(r'''$cmd''', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    samples.append((time.perf_counter() - t0) * 1000.0)
print(f"{statistics.median(samples):.1f}")
PY
}

# Cuenta matches de grep (líneas con coincidencia).
grep_matches() {
    local needle="$1"
    grep -rnE "$needle" "$CORPUS" 2>/dev/null | wc -l | tr -d ' '
}

rg_matches() {
    local needle="$1"
    rg -e "$needle" "$CORPUS" --count-matches 2>/dev/null \
        | awk -F: '{sum+=$NF} END {print sum+0}'
}

sg_matches() {
    local needle="$1"
    # ast-grep necesita un patrón estructural; usamos el needle como pattern literal.
    sg run -p "$needle" -l rust "$CORPUS" 2>/dev/null | grep -c "^[^[:space:]]" || true
}

# rpgrep: devuelve el score del top-1 (o N/A si vacío).
rpgrep_top1() {
    local query="$1"
    "$RPGREP_BIN" search "$query" --index "$INDEX_DIR" --topk 5 2>/dev/null \
        | head -n 1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^score=/) print substr($i,7)}'
}

# Número de resultados que devuelve rpgrep.
rpgrep_count() {
    local query="$1"
    "$RPGREP_BIN" search "$query" --index "$INDEX_DIR" --topk 5 2>/dev/null \
        | grep -cE "score=" || true
}

# ---- bucle principal -------------------------------------------------------
printf "query\tgrep_matches\trg_matches\tsg_matches\trpgrep_n\trpgrep_top1_score\tgrep_ms\trg_ms\tsg_ms\trpgrep_ms\n"

while IFS=$'\t' read -r query expected files; do
    # Saltar comentarios y vacías
    case "$query" in
        \#*|"") continue ;;
    esac
    [ -z "${query// }" ] && continue

    # Primer token de la query como needle para grep/rg/sg (no son semánticos).
    first_word=$(printf '%s' "$query" | awk '{print $1}')

    g_n=$(grep_matches "$first_word")
    g_ms=$(measure_ms "grep -rnE '$first_word' '$CORPUS'")

    if [ "$HAS_RG" -eq 1 ]; then
        r_n=$(rg_matches "$first_word")
        r_ms=$(measure_ms "rg -e '$first_word' '$CORPUS'")
    else
        r_n="WARN"; r_ms="WARN"
    fi

    if [ "$HAS_SG" -eq 1 ]; then
        s_n=$(sg_matches "$first_word")
        s_ms=$(measure_ms "sg run -p '$first_word' -l rust '$CORPUS'")
    else
        s_n="WARN"; s_ms="WARN"
    fi

    rp_n=$(rpgrep_count "$query")
    rp_top1=$(rpgrep_top1 "$query")
    [ -z "$rp_top1" ] && rp_top1="N/A"
    rp_ms=$(measure_ms "$RPGREP_BIN search '$query' --index '$INDEX_DIR' --topk 5")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$query" "$g_n" "$r_n" "$s_n" "$rp_n" "$rp_top1" "$g_ms" "$r_ms" "$s_ms" "$rp_ms"
done < "$GOLDEN"

echo "" >&2
echo "[bench_compare] hecho. Pipe la salida a 'column -ts $'\t'' para tabular." >&2
