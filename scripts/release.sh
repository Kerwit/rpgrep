#!/usr/bin/env bash
# Release en un solo paso: pregunta la versión y luego commitea, taggea y pushea
# automáticamente (el tag dispara .github/workflows/release.yml).
#
# Uso:
#   scripts/release.sh          # pregunta la versión (default = patch +1)
#   scripts/release.sh 1.2.3    # versión explícita, sin preguntar
#
# Si eliges la misma versión que la actual, reemplaza el tag existente
# (lo recrea en local y remoto).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OLD="$(perl -ne 'if (/^version = "([^"]+)"/) { print $1; exit }' Cargo.toml)"
[ -n "$OLD" ] || { echo "No pude leer la versión de Cargo.toml" >&2; exit 1; }

IFS=. read -r MAJ MIN PAT <<<"$OLD"
DEFAULT="$MAJ.$MIN.$((PAT + 1))"

if [ $# -ge 1 ]; then
  NEW="$1"
else
  echo "Versión actual: $OLD"
  read -rp "Nueva versión (enter=$DEFAULT · escribe $OLD para reemplazar el tag): " NEW
  NEW="${NEW:-$DEFAULT}"
fi

[[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Formato de versión inválido: $NEW" >&2; exit 2; }

REPLACE=0
[ "$NEW" = "$OLD" ] && REPLACE=1

# Sincroniza versión en Cargo.toml y README (no-op si NEW == OLD).
# Cargo.lock está en .gitignore: se actualiza en local pero no se commitea.
perl -i -pe 'BEGIN{$d=0} if (!$d && /^version = "/){ s/"[^"]+"/"'"$NEW"'"/; $d=1 }' Cargo.toml
[ -f Cargo.lock ] && perl -i -pe 'if (/^name = "rpgrep"$/){$h=1} elsif ($h && /^version = "/){ s/"[^"]+"/"'"$NEW"'"/; $h=0 }' Cargo.lock
perl -i -pe 's/\bv\Q'"$OLD"'\E\b/v'"$NEW"'/g' README.md

# Commit solo si los ficheros de versión cambiaron.
git add Cargo.toml README.md
if ! git diff --cached --quiet; then
  git commit -m "chore: release v$NEW"
fi

# Tag (recrea local+remoto si se reemplaza una versión existente).
if [ "$REPLACE" -eq 1 ] && git rev-parse "v$NEW" >/dev/null 2>&1; then
  echo "Reemplazando tag v$NEW…"
  git tag -d "v$NEW"
  git push origin ":refs/tags/v$NEW" || true
fi
git tag -a "v$NEW" -m "v$NEW"

# Push automático.
git push origin HEAD
git push origin "v$NEW"

echo "Release v$NEW publicado — release.yml en marcha."
