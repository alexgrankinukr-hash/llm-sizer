#!/usr/bin/env bash
# Refresh the bundled model records from the open-data repository (mirrored nightly from the live tool).
# Usage: bash scripts/llm-sizer/snapshot-data.sh   (run from the repository root; needs curl and python3)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/public/data/llm-sizer/models"
RAW="https://raw.githubusercontent.com/alexgrankinukr-hash/llm-sizer-data/main"
API="https://api.github.com/repos/alexgrankinukr-hash/llm-sizer-data/contents/models"
mkdir -p "$DEST"
tmp="$(mktemp -d)"
curl -fsSL --retry 4 --retry-all-errors --retry-delay 2 "$RAW/models/index.json" -o "$tmp/index.json"
python3 - "$tmp/index.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"index: {d.get('count')} models, generated {d.get('generated_at')}")
PY
names=$(curl -fsSL --retry 4 --retry-all-errors --retry-delay 2 "$API" | python3 -c "import json,sys; print('\n'.join(x['name'] for x in json.load(sys.stdin) if x['name'].endswith('.json') and x['name'] != 'index.json'))")
n=0
for f in $names; do
  curl -fsSL --retry 4 --retry-all-errors --retry-delay 2 "$RAW/models/$f" -o "$tmp/$f"
  n=$((n + 1))
done
rm -rf "$DEST"
mkdir -p "$DEST"
cp "$tmp"/*.json "$DEST/"
rm -rf "$tmp"
echo "snapshot: index + $n records in public/data/llm-sizer/models"
