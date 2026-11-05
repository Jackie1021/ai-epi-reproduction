#!/usr/bin/env bash
# Amendment 15: mechanism test with counterfactual prompts (config_cf.yaml); repeat until every request has an answer
#   nohup bash run_cf.sh 20 > runs/cf_loop.log 2>&1 &
cd "$(dirname "$0")"
WAIT_MIN="${1:-20}"
MODELS="gem38f,gem25f,gemma31,gemma26,qwen,glm52,dsv41f,gptoss20,kimik3,nemo3u"   # Ling excluded (free endpoint withdrawn)
ORIG="2024-07-06,2024-08-31,2024-10-26,2024-12-21,2025-02-15,2025-04-12,2025-06-07,2025-08-02,2025-09-27,2025-11-22,2026-01-17,2026-03-14"
round=0
while :; do
  round=$((round + 1))
  echo "===== mechanism test, round $round  $(date '+%F %T') ====="
  bash llm.sh run --config config_cf.yaml --origins "$ORIG" --models "$MODELS" 2>&1 | grep -v '^\s*$' | tail -15
  left=$(bash llm.sh pending --config config_cf.yaml --origins "$ORIG" --models "$MODELS" --quiet 2>/dev/null | tail -1)
  echo "requests without an answer: ${left:-?}"
  bash llm.sh pending --config config_cf.yaml --origins "$ORIG" --models "$MODELS" 2>/dev/null | tail -14
  if [ "$left" = "0" ]; then echo "mechanism test complete: runs/cf/"; break; fi
  sleep $((WAIT_MIN * 60))
done
