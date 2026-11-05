#!/usr/bin/env bash
# 修正案 14：3 个历史期起点 + 主研究 2026-03-14，用更长的链和更小的步长再拟合（稳健性检验，不替换任何结果）
# 每条链 2000 次预热 + 4000 次抽样，adapt_delta 0.99，max_treedepth 13，种子 2028，8 个优化起点；4 个同时进行（16 核）
cd "$(dirname "$0")"
mkdir -p output/logs
keep_awake() {
  if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi
}
LOG=output/logs/longer_fits.log
OPTS="--seed 2028 --starts 8 --iter 6000 --warmup 2000 --adapt-delta 0.99 --max-treedepth 13"
fit_one() {   # $1 = study (historical | main), $2 = origin, $3 = outdir
  echo "[$(date '+%F %T')] start $1 $2" >> $LOG
  AI_EPI_STUDY=$1 keep_awake Rscript export_origins.R "$2" $OPTS --outdir "$3" > "output/logs/longer_$1_$2.log" 2>&1
  if [ -f "output/$3/$2/DONE" ]; then
    echo "[$(date '+%F %T')] done $1 $2: $(python3 -c "import json; d=json.load(open('output/$3/$2/meta.json'))['diagnostics']; print('max Rhat', d['max_rhat'], 'divergent', d['n_divergent'])")" >> $LOG
  else
    echo "[$(date '+%F %T')] $1 $2 failed (see output/logs/longer_$1_$2.log)" >> $LOG
  fi
}
echo "[$(date '+%F %T')] amendment 14 fits: $OPTS" >> $LOG
fit_one historical 2016-11-05 origins_hist_longer &
fit_one historical 2016-12-31 origins_hist_longer &
fit_one historical 2017-02-25 origins_hist_longer &
fit_one main       2026-03-14 origins_longer &
wait
echo "[$(date '+%F %T')] amendment 14 fits finished" >> $LOG
