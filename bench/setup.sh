#!/usr/bin/env bash
# Use a fresh environment rather than reusing system packages.
set -euo pipefail
cd "$(dirname "$0")"
requirements=requirements.txt
case "${1:-}" in
  "") ;;
  --live) requirements=requirements-live.txt ;;
  *) echo "Usage: bash bench/setup.sh [--live]" >&2; exit 2 ;;
esac
"${PYTHON_BIN:-python3.13}" -m venv .venv
.venv/bin/python -m pip install -r "$requirements"
if [[ "$requirements" == requirements-live.txt && ! -f .env ]]; then
  (umask 077; cp .env.example .env)
fi
