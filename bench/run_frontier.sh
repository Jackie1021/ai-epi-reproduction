#!/usr/bin/env bash
# Frontier extension: GPT-6.1 Sol and GPT-6 Astra, including the Astra (iv)-(v) follow-up.
set -eo pipefail
# Stops before a stage if the money spent so far (from the call logs) exceeds BUDGET dollars.
#   nohup bash run_frontier.sh > runs/frontier_loop.log 2>&1 &
cd "$(dirname "$0")"
BUDGET="${BUDGET:-27}"
ORIG="2024-07-06,2024-08-31,2024-10-26,2024-12-21,2025-02-15,2025-04-12,2025-06-07,2025-08-02,2025-09-27,2025-11-22,2026-01-17,2026-03-14"   # the 12 origins of the mechanism test
spent() { .venv/bin/python - <<'PY'
import json, glob
s = 0.0
for f in glob.glob("runs/frontier*/calls.jsonl"):
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if not r.get("cache_hit") and r.get("cost_usd"): s += r["cost_usd"]
print(f"{s:.2f}")
PY
}
stage() {
  local name="$1"; shift
  local s; s=$(spent)
  if .venv/bin/python -c "import sys; sys.exit(0 if float('$s') < float('$BUDGET') else 1)"; then
    echo "===== $name  (spent so far \$$s)  $(date '+%F %T') ====="
    bash llm.sh run "$@" 2>&1 | grep -v '^\s*$' | tail -12
  else
    echo "===== $name skipped: spent \$$s >= budget \$$BUDGET ====="
  fi
}
stage "GPT-6.1 Sol: main study, designs (i)-(v) + external-information experiment" --config config_frontier.yaml --models gpt61sol
stage "GPT-6.1 Sol: mechanism test" --config config_frontier_cf.yaml --models gpt61sol --origins "$ORIG"
stage "GPT-6.1 Sol: 2015-2019 retrospective" --config config_frontier_hist.yaml --models gpt61sol
stage "GPT-6 Astra: main study, designs (i)-(iii)" --config config_frontier.yaml --models gpt6astra --arms direct_blind,direct_named,r0_semi
stage "GPT-6 Astra: mechanism test" --config config_frontier_cf.yaml --models gpt6astra --origins "$ORIG"
stage "GPT-6 Astra: main study, designs (iv)-(v)" --config config_frontier.yaml --models gpt6astra --arms r0_ctx,r0_imp
echo "===== done  (spent \$$(spent))  $(date '+%F %T') ====="
