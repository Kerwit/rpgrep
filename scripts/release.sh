#!/usr/bin/env bash
# Bump de versión + actualización de README + tag, en un solo paso.
#
# Uso:
#   scripts/release.sh            # bump de patch (0.2.5 -> 0.2.6)
#   scripts/release.sh minor      # 0.2.5 -> 0.3.0
#   scripts/release.sh major      # 0.2.5 -> 1.0.0
#   scripts/release.sh 1.2.3      # versión explícita
#   scripts/release.sh --push     # además hace git push + push del tag
#
# Sincroniza Cargo.toml, Cargo.lock y las referencias `vX.Y.Z` del README,
# crea el commit `chore: release vX.Y.Z` y el tag `vX.Y.Z` (que dispara
# .github/workflows/release.yml).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PUSH=0
BUMP="patch"
for arg in "$@"; do
  case "$arg" in
    --push) PUSH=1 ;;
    patch|minor|major) BUMP="$arg" ;;
    [0-9]*.[0-9]*.[0-9]*) BUMP="$arg" ;;
    *) echo "Argumento no reconocido: $arg" >&2; exit 2 ;;
  esac
done

OLD="$(perl -ne 'if (/^version = "([^"]+)"/) { print $1; exit }' Cargo.toml)"
[ -n "$OLD" ] || { echo "No pude leer la versión de Cargo.toml" >&2; exit 1; }

if [[ "$BUMP" == *.*.* ]]; then
  NEW="$BUMP"
else
  IFS=. read -r MAJ MIN PAT <<<"$OLD"
  case "$BUMP" in
    major) MAJ=$((MAJ+1)); MIN=0; PAT=0 ;;
    minor) MIN=$((MIN+1)); PAT=0 ;;
    patch) PAT=$((PAT+1)) ;;
  esac
  NEW="$MAJ.$MIN.$PAT"
fi

echo "Versión: $OLD -> $NEW"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Aviso: hay cambios sin commitear; solo se incluirán los ficheros de versión." >&2
fi

if git rev-parse "v$NEW" >/dev/null 2>&1; then
  echo "El tag v$NEW ya existe." >&2; exit 1
fi

# Cargo.toml: solo la línea version del paquete.
perl -i -pe 'BEGIN{$d=0} if (!$d && /^version = "/){ s/"[^"]+"/"'"$NEW"'"/; $d=1 }' Cargo.toml

# Cargo.lock: solo el bloque del paquete rpgrep.
perl -i -pe 'if (/^name = "rpgrep"$/){$h=1} elsif ($h && /^version = "/){ s/"[^"]+"/"'"$NEW"'"/; $h=0 }' Cargo.lock

# README: todas las referencias a la versión actual.
perl -i -pe 's/\bv\Q'"$OLD"'\E\b/v'"$NEW"'/g' README.md

git add Cargo.toml Cargo.lock README.md
git commit -m "chore: release v$NEW"
git tag -a "v$NEW" -m "v$NEW"

echo "Creado commit y tag v$NEW."
if [ "$PUSH" -eq 1 ]; then
  git push origin HEAD
  git push origin "v$NEW"
  echo "Push hecho. release.yml se está ejecutando."
else
  echo "Para publicar:  git push origin HEAD && git push origin v$NEW"
fi
