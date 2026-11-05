#!/usr/bin/env bash
# 修正案 11：推理测试扩展的调度（主研究优先）
# 每 30 分钟检查一次：某服务商上的主研究模型全部有回答后，才发这个服务商的推理版本请求
# （同一服务商的模型共用免费额度）。全部推理请求有回答、或到 2026-10-31 为止。
#   bash run_reason_ext.sh            日志 runs/reason_ext/run.log
cd "$(dirname "$0")"
mkdir -p runs/reason_ext
LOG=runs/reason_ext/run.log
END=2026-10-31

# 服务商：主研究模型 -> 推理版本
declare -a PROVIDER_GROUPS=(   # 注意：不能叫 GROUPS（bash 保留变量）
  "openrouter|nemo3u,ling3s|nemo3ur,ling3sr"
  "modelscope|qwen,glm52,dsv41f|qwenr,glm52r,dsv41fr"
  "google|gem38f,gem25f,gemma31,gemma26|gem38fr,gem25fr"
)

main_pending() {   # 主研究里给定模型还没有回答的请求数（pending 表里没有的模型 = 0）
  bash llm.sh pending 2>/dev/null | awk -v want=",$1," '
    NR > 2 && $1 != "" && $1 !~ /^arm$|^model$/ { if (index(want, "," $1 ",")) { for (i = 2; i <= NF; i++) s += $i } }
    END { print s + 0 }'
}

ext_pending() {
  bash llm.sh pending --config config_reason_ext.yaml --models "$1" --quiet 2>/dev/null | tail -1
}

while true; do
  [ "$(date +%F)" \> "$END" ] && { echo "[$(date '+%F %T')] 已过截止日 $END，停止" >> "$LOG"; break; }
  total=0; ran=0
  for g in "${PROVIDER_GROUPS[@]}"; do
    IFS='|' read -r prov mains exts <<< "$g"
    [ -z "$prov" ] || [ -z "$mains" ] || [ -z "$exts" ] && { echo "[$(date '+%F %T')] 配置解析失败：$g，停止" >> "$LOG"; exit 1; }
    left=$(ext_pending "$exts"); left=${left:-0}
    total=$((total + left))
    [ "$left" = "0" ] && continue
    mp=$(main_pending "$mains")
    if [ "$mp" = "0" ]; then
      echo "[$(date '+%F %T')] $prov：主研究已完成，发推理请求 $exts（还剩 $left 个）" >> "$LOG"
      [ -n "$DRY" ] && continue
      bash llm.sh run --config config_reason_ext.yaml --models "$exts" >> "$LOG" 2>&1
      ran=1
    else
      echo "[$(date '+%F %T')] $prov：主研究还有 $mp 个请求，推理请求等待" >> "$LOG"
    fi
  done
  if [ "$ran" = "1" ]; then
    bash llm.sh score --config config_reason_ext.yaml > runs/reason_ext/score.log 2>&1 \
      && echo "[$(date '+%F %T')] 已重新评分（runs/reason_ext/）" >> "$LOG"
  fi
  [ "$total" = "0" ] && { echo "[$(date '+%F %T')] 推理请求全部完成" >> "$LOG"; break; }
  [ -n "$DRY" ] && break                                   # DRY=1：只打印一次判断，不发请求
  sleep 1800
done
