#!/usr/bin/env bash
# Amendment 15 extension: mechanism test with reasoning on (config_cf_reason.yaml); loop until every request is answered
cd "$(dirname "$0")"
WAIT_MIN="${1:-20}"
MODELS="gptoss20r,kimik3r,qwenr,glm52r,dsv41fr,nemo3ur"            # Gemini added once runs/cf has its Gemini answers
ORIG="2024-07-06,2024-08-31,2024-10-26,2024-12-21,2025-02-15,2025-04-12,2025-06-07,2025-08-02,2025-09-27,2025-11-22,2026-01-17,2026-03-14"
round=0
while :; do
  round=$((round + 1))
  echo "===== mechanism test with reasoning, round $round  $(date '+%F %T') ====="
  bash llm.sh run --config config_cf_reason.yaml --origins "$ORIG" --models "$MODELS" 2>&1 | grep -v '^\s*$' | tail -12
  left=$(bash llm.sh pending --config config_cf_reason.yaml --origins "$ORIG" --models "$MODELS" --quiet 2>/dev/null | tail -1)
  echo "requests without an answer: ${left:-?}"
  bash llm.sh pending --config config_cf_reason.yaml --origins "$ORIG" --models "$MODELS" 2>/dev/null | tail -10
  if [ "$left" = "0" ]; then echo "reasoning mechanism test complete: runs/cf_reason/"; break; fi
  sleep $((WAIT_MIN * 60))
done
