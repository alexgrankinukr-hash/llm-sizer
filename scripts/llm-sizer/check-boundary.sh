#!/usr/bin/env bash
# LLM Sizer boundary check (see src/lib/llm-sizer/README.md). This copy's allowlist names only what this repository ships.
# Fails when a tool file imports something outside the tool that is not on the allowlist,
# when a file outside the tool imports from it, or when the engine touches the environment or Node APIs.
set -euo pipefail
cd "$(dirname "$0")/../.."

TOOL_DIRS=(src/lib/llm-sizer src/components/llm-sizer src/pages/tools/llm-sizer src/pages/api/llm-sizer)
DATA_DIRS=(public/data/llm-sizer)   # the tool's own data files: valid import targets, never scanned for imports
fail=0

resolve() { # resolve a relative import specifier against the importing file's directory
  python3 - "$1" "$2" <<'PY'
import os, sys
base, spec = sys.argv[1], sys.argv[2]
print(os.path.normpath(os.path.join(os.path.dirname(base), spec)))
PY
}

in_tool() { # is a repo-relative path inside one of the tool directories (code or data)?
  for d in "${TOOL_DIRS[@]}" "${DATA_DIRS[@]}"; do [[ "$1" == "$d"/* ]] && return 0; done; return 1
}

while IFS= read -r file; do
  while IFS= read -r line; do
    spec=$(printf '%s' "$line" | sed -nE "s/.*from[[:space:]]+['\"]([^'\"]+)['\"].*/\1/p")
    [[ -z "$spec" ]] && spec=$(printf '%s' "$line" | sed -nE "s/^import[[:space:]]+['\"]([^'\"]+)['\"].*/\1/p")
    [[ -z "$spec" ]] && continue
    if [[ "$spec" == .* ]]; then
      target=$(resolve "$file" "$spec")
      if in_tool "$target"; then continue; fi
      case "$target" in
        src/lib/track) continue ;;
        src/layouts/BaseLayout.astro|src/components/PublicHeader.astro)
          [[ "$file" == src/pages/tools/llm-sizer/* ]] && continue ;;
      esac
      echo "BOUNDARY: $file imports $spec (outside the tool, not on the allowlist)"; fail=1
    else
      case "$spec" in
        react|react/*|react-dom|react-dom/*) [[ "$file" == src/components/llm-sizer/* ]] && continue ;;
        vitest|node:*)
          [[ "$file" == *.test.ts || "$file" == */__fixtures__/* ]] && continue
          [[ "$spec" == node:* && "$file" == src/pages/api/llm-sizer/*.png.ts ]] && continue ;;   # server-rendered PNGs read fonts + hash
        postgres) [[ "$file" == src/lib/llm-sizer/adapters/* ]] && continue ;;   # a fork may point adapters/db.ts back at a database
        @vercel/og) [[ "$file" == src/pages/api/llm-sizer/*.png.ts ]] && continue ;;   # server-rendered PNGs
        astro) [[ "$file" == src/pages/api/llm-sizer/* || "$file" == src/pages/tools/llm-sizer/* ]] && [[ "$line" == *"import type"* ]] && continue ;;
      esac
      echo "BOUNDARY: $file imports package '$spec' (not on the allowlist for that directory)"; fail=1
    fi
  done < <(grep -E "^\s*import\b|^\s*export\s+.*\bfrom\b" "$file" || true)
done < <(find "${TOOL_DIRS[@]}" -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.astro' \) 2>/dev/null | sort)

# nothing outside the tool may import from it
while IFS= read -r file; do
  in_tool "$file" && continue
  if grep -qE "from ['\"][^'\"]*llm-sizer" "$file"; then echo "BOUNDARY: $file (outside the tool) imports from the tool"; fail=1; fi
done < <(find src -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.astro' \) | sort)

# the engine is pure: no environment, no Node APIs, no browser globals
if [[ -d src/lib/llm-sizer/engine ]]; then
  if grep -rnE "import\.meta\.env|from ['\"]node:|process\.|require\(|window\.|document\." src/lib/llm-sizer/engine --include='*.ts' --exclude='*.test.ts' --exclude-dir=__fixtures__ | grep -v "^.*//" ; then
    echo "BOUNDARY: the engine must not touch the environment, Node, or the DOM"; fail=1
  fi
fi

if [[ $fail -eq 0 ]]; then echo "boundary OK"; fi
exit $fail
