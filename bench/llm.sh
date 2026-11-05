#!/usr/bin/env bash
# 在任何目录下运行实验命令，自动用 bench/.venv 的 Python：
#   bash bench/llm.sh doctor
cd "$(dirname "$0")"
PY="${PYTHON_BIN:-.venv/bin/python}"
if [ "$PY" = ".venv/bin/python" ] && [ ! -x "$PY" ]; then PY=python3; fi
exec "$PY" -m llm_bench "$@"
