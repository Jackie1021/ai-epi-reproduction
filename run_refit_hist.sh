#!/usr/bin/env bash
# 历史期（修正案 8）的预先规定重拟合：max R-hat > 1.05 或任一模块发散 > 5% 的起点，用种子 2026、8 个优化起点重拟合一次，
# 无论结果如何都替换原拟合；原拟合移到 output/archive/hist_refit_originals/，
# 该起点 R0 设计（提示词里有拟合估计）的 LLM 回答移到 bench/runs/hist/responses_archive_refit/，下一轮重新请求。
# 拟合进行中每 10 分钟标记新完成且不合格的起点（REFIT_PENDING：LLM 循环跳过它们）；所有分片结束后开始重拟合
# （最多 4 个同时进行，做完一个补一个；每个 4 条链，共 16 核）。
cd "$(dirname "$0")"
mkdir -p output/logs
keep_awake() {
  if command -v caffeinate >/dev/null 2>&1; then caffeinate -i "$@"; else "$@"; fi
}
export AI_EPI_STUDY=historical
LOG=output/logs/hist_refit.log
flagged() {   # 不合格、还没重拟合过的起点
  python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("output/origins_hist/*/meta.json")):
    d = os.path.dirname(f)
    if not os.path.exists(os.path.join(d, "DONE")) or os.path.exists(os.path.join(d, "REFITTED")):
        continue
    m = json.load(open(f)); g = m["diagnostics"]; b = g["blocks"]
    blocks = b if isinstance(b, list) else [dict(zip(b, t)) for t in zip(*b.values())]
    if g["max_rhat"] > 1.05 or max(x["divergent"] for x in blocks) / 2000 > 0.05:
        print(m["origin"])
PY
}
mark() { for o in $(flagged); do [ -f "output/origins_hist/$o/REFIT_PENDING" ] || { touch "output/origins_hist/$o/REFIT_PENDING"; echo "[$(date '+%F %T')] flagged $o" >> $LOG; }; done; }
while pgrep -f "export_origins.R --shard" > /dev/null; do mark; sleep 600; done
mark
refit_one() {
  o=$1
  echo "[$(date '+%F %T')] refit $o" >> $LOG
  keep_awake Rscript export_origins.R "$o" --seed 2026 --starts 8 --outdir origins_hist_refit > "output/logs/hist_refit_$o.log" 2>&1
  if [ -f "output/origins_hist_refit/$o/DONE" ]; then
    mkdir -p output/archive/hist_refit_originals bench/runs/hist/responses_archive_refit/$o
    mv "output/origins_hist/$o" "output/archive/hist_refit_originals/$o"
    mv "output/origins_hist_refit/$o" "output/origins_hist/$o"
    touch "output/origins_hist/$o/REFITTED"
    for a in bench/runs/hist/responses/$o/r0_*; do [ -d "$a" ] && mv "$a" "bench/runs/hist/responses_archive_refit/$o/"; done
    echo "[$(date '+%F %T')] replaced $o: $(python3 -c "import json; d=json.load(open('output/origins_hist/$o/meta.json'))['diagnostics']; print('max Rhat', d['max_rhat'], 'divergent', d['n_divergent'])")" >> $LOG
  else
    echo "[$(date '+%F %T')] refit of $o failed (see output/logs/hist_refit_$o.log); original kept, still pending" >> $LOG
  fi
}
list=$(for o in $(flagged); do echo $o; done)
echo "[$(date '+%F %T')] refitting: $list" >> $LOG
PAR=${PAR:-4}
for o in $list; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$PAR" ]; do sleep 30; done
  refit_one "$o" &
done
wait
echo "[$(date '+%F %T')] refits finished" >> $LOG
