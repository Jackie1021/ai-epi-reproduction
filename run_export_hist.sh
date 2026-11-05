#!/usr/bin/env bash
# 历史期研究（修正案 8）的滚动拟合：65 个起点（2015-01-31 至 2019-12-28），分 N 份并行（每份 4 条链）
#   bash run_export_hist.sh 4          # 16 核跑满；日志 output/logs/hist_shard_*.log；结果 output/origins_hist/
# 中断后重跑即可续上（有 DONE 的起点自动跳过）
cd "$(dirname "$0")"
mkdir -p output/logs
keep_awake() {
  if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi
}
export AI_EPI_STUDY=historical
N="${1:-4}"
mkdir -p output/logs
Rscript -e 'source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))' > output/logs/hist_compile.log 2>&1 || { echo "compile failed"; exit 1; }
for i in $(seq 1 "$N"); do
  keep_awake Rscript export_origins.R --shard "$i/$N" > "output/logs/hist_shard_${i}.log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] historical fits finished: $(ls output/origins_hist/*/DONE 2>/dev/null | wc -l | tr -d ' ') / 65 origins"
