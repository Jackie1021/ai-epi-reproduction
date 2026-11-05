#!/usr/bin/env bash
# 反复运行 LLM 实验，直到所有起点、所有模型都有结果：每轮只发还没有答案的请求，其余直接读缓存
#   bash run_until_done.sh        # 每 60 分钟一轮
#   bash run_until_done.sh 30     # 每 30 分钟一轮
# 每轮先跑主研究（config.yaml），再跑历史期（config_hist.yaml，修正案 8：主研究的请求优先用免费额度）。
# 当天免费额度用完的模型本轮自动停下，下一轮再试；Stan 新跑完的起点下一轮自动加入。
# 每轮结束都重新评分、出图并同步到 AI_Epi/results/（中间结果随时可看）；全部完成（且 Stan 也跑完）后停止。
# 中断后重新运行本脚本即可接着跑。
cd "$(dirname "$0")"
WAIT_MIN="${1:-60}"
HIST_CFG=config_hist.yaml
HIST_END="2026-10-31"            # 修正案 8：历史期请求最晚收到这天
# 修正案 11（更新）：推理扩展的请求优先。某服务商的推理请求还没做完时，共用同一额度的主研究 / 历史期模型本轮先等
REASON_CFG=config_reason_ext.yaml
ALL_MODELS="gem38f,gem25f,gemma31,gemma26,qwen,glm52,dsv41f,gptoss20,kimik3,nemo3u,ling3s"
rpend() { bash llm.sh pending --config "$REASON_CFG" --models "$1" --quiet 2>/dev/null | tail -1; }
skip_models() {        # 读不到数时按"还没做完"处理（保守）
  local s=""
  [ "$(rpend nemo3ur,ling3sr)" != "0" ] && s="$s,nemo3u,ling3s"            # OpenRouter 免费额度按账号共用
  [ "$(rpend qwenr,glm52r,dsv41fr)" != "0" ] && s="$s,qwen,glm52,dsv41f"   # ModelScope 余额按账号共用
  [ "$(rpend gem38fr)" != "0" ] && s="$s,gem38f"                          # Gemini 免费额度按模型计
  [ "$(rpend gem25fr)" != "0" ] && s="$s,gem25f"
  echo "${s#,}"
}
keep_models() {
  local skip=",$1," out=""
  for m in ${ALL_MODELS//,/ }; do [[ "$skip" == *",$m,"* ]] || out="$out,$m"; done
  echo "${out#,}"
}
round=0
while :; do
  round=$((round + 1))
  echo "===== 第 $round 轮  $(date '+%F %T') ====="
  if [ -f "$REASON_CFG" ] && [ "$(bash llm.sh pending --config "$REASON_CFG" --quiet 2>/dev/null | tail -1)" != "0" ]; then
    echo "--- 推理扩展（修正案 11，优先）"
    bash llm.sh run --config "$REASON_CFG" 2>&1 | grep -v '^\s*$' | tail -12
    bash llm.sh score --config "$REASON_CFG" > /dev/null 2>&1 && echo "推理扩展已重新评分（runs/reason_ext/）"
  fi
  SKIP=$(skip_models); MODELS=$(keep_models "$SKIP")
  [ -n "$SKIP" ] && echo "本轮暂停（等推理请求做完）：$SKIP"
  bash llm.sh run --models "$MODELS" 2>&1 | grep -v '^\s*$' | tail -25
  left=$(bash llm.sh pending --quiet 2>/dev/null | tail -1)
  hleft=0
  if [ "$(date +%F)" \< "$HIST_END" ] || [ "$(date +%F)" = "$HIST_END" ]; then
    if ls ../output/origins_hist/*/DONE > /dev/null 2>&1; then
      echo "--- 历史期（2015-2019）"
      bash llm.sh run --config "$HIST_CFG" --models "$MODELS" 2>&1 | grep -v '^\s*$' | tail -15
      hleft=$(bash llm.sh pending --config "$HIST_CFG" --quiet 2>/dev/null | tail -1)
    fi
  else
    echo "--- 历史期：已过收集截止日 $HIST_END，不再发请求"
  fi
  stan=$(pgrep -f "[e]xport_origins.R" | wc -l | tr -d ' ')
  echo "还没有答案的请求：主研究 ${left:-?}，历史期 ${hleft:-?}  |  仍在拟合的 Stan 进程：$stan"
  if bash llm.sh score > /dev/null 2>&1 && bash llm.sh plot > /dev/null 2>&1; then
    echo "已重新评分、出图，并同步到 results/"
  else
    echo "评分或出图失败（手动运行 bash llm.sh score / bash llm.sh plot 查看原因）"
  fi
  hdone=$(ls ../output/origins_hist/*/DONE 2>/dev/null | wc -l | tr -d ' ')
  if [ "$hdone" != "0" ] && [ "$hdone" != "${hplotted:-0}" ]; then        # 历史期 Stan 图（有新完成的起点时重画）
    (cd .. && AI_EPI_STUDY=historical Rscript plot_origins.R > output/logs/plot_origins_hist.log 2>&1) \
      && { hplotted=$hdone; echo "历史期 Stan 图已更新（$hdone / 65 个起点）"; } || echo "历史期 Stan 作图失败（output/logs/plot_origins_hist.log）"
  fi
  if ls runs/hist/forecasts.parquet > /dev/null 2>&1; then
    bash llm.sh score --config "$HIST_CFG" > /dev/null 2>&1 && bash llm.sh plot --config "$HIST_CFG" > /dev/null 2>&1 \
      && echo "历史期已重新评分（runs/hist/）" || echo "历史期评分失败（bash llm.sh score --config $HIST_CFG）"
  fi
  if [ "$left" = "0" ] && [ "$stan" = "0" ] && { [ "$hleft" = "0" ] || [ "$(date +%F)" \> "$HIST_END" ]; }; then
    echo "全部完成，结果在 bench/runs/main/、bench/runs/hist/ 和 AI_Epi/results/"
    break
  fi
  echo "下一轮在 $WAIT_MIN 分钟后（Ctrl+C 停止；重新运行本脚本可接着跑）"
  sleep $((WAIT_MIN * 60))
done
