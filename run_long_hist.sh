#!/usr/bin/env bash
# 修正案 12：历史期 5 个重拟合后仍未收敛（R-hat > 1.05）的起点，用更长的链再拟合一次（稳健性检验）
# 每条链 1000 次预热 + 1000 次抽样，种子 2027，8 个优化起点；写到 output/origins_hist_long/，不替换任何结果
# 最多 4 个同时进行（每个 4 条链，16 核）；日志 output/logs/hist_long.log
cd "$(dirname "$0")"
mkdir -p output/logs
keep_awake() {
  if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi
}
export AI_EPI_STUDY=historical
LOG=output/logs/hist_long.log
PAR=${PAR:-4}
ORIGINS="2016-11-05 2016-12-03 2016-01-02 2017-02-25 2016-12-31"   # R-hat 从高到低，最难的先开始
fit_one() {
  o=$1
  echo "[$(date '+%F %T')] start $o" >> $LOG
  keep_awake Rscript export_origins.R "$o" --seed 2027 --starts 8 --iter 2000 --warmup 1000 \
    --outdir origins_hist_long > "output/logs/hist_long_$o.log" 2>&1
  if [ -f "output/origins_hist_long/$o/DONE" ]; then
    echo "[$(date '+%F %T')] done $o: $(python3 -c "import json; d=json.load(open('output/origins_hist_long/$o/meta.json'))['diagnostics']; print('max Rhat', d['max_rhat'], 'divergent', d['n_divergent'])")" >> $LOG
  else
    echo "[$(date '+%F %T')] $o failed (see output/logs/hist_long_$o.log)" >> $LOG
  fi
}
echo "[$(date '+%F %T')] longer-chain fits (amendment 12): $ORIGINS" >> $LOG
for o in $ORIGINS; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$PAR" ]; do sleep 30; done
  fit_one "$o" &
done
wait
echo "[$(date '+%F %T')] longer-chain fits finished" >> $LOG
