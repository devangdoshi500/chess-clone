#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -d "$repo_dir/demo/node_modules" ]]; then
  npm install --prefix "$repo_dir/demo"
fi
exec npm run dev --prefix "$repo_dir/demo"
