#!/usr/bin/env bash
# 一键滚动拟合：先单进程编译模型（避免多个进程同时写编译缓存），再把起点分成 N 份并行拟合，
# 等待期间每 20 秒刷新一次进度面板
#   bash run_export.sh                              # 3 份并行（每份 4 条链，共 12 核）
#   bash run_export.sh 3 --iter 1000 --warmup 500   # 后面的参数原样传给 export_origins.R
#   bash run_export.sh --watch                      # 只看进度（关掉终端后重新查看用）
# 建议用 caffeinate 防止 Mac 睡眠：caffeinate -i bash run_export.sh
# 日志在 output/logs/；已完成或正在别的进程里跑的起点会自动跳过，中断后重跑即可续上
cd "$(dirname "$0")"
mkdir -p output/logs

status() {
  local total done_n f name cur blk its donec
  total=$(grep -h '\] [0-9]* origins' output/logs/shard_*.log 2>/dev/null |
          sed -E 's/.*\] ([0-9]+) origins.*/\1/' | awk '{s += $1} END {print s + 0}')
  done_n=$(ls output/origins/*/DONE 2>/dev/null | wc -l | tr -d ' ')
  printf '\033[1mStan 滚动拟合进度\033[0m   %s\n' "$(date '+%F %T')"
  printf '已完成起点：%s / %s\n\n' "$done_n" "$total"
  for f in output/logs/shard_*.log output/logs/bench_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    cur=$(grep -E '\] fit [0-9-]+' "$f" | tail -1 | sed -E 's/.*\] fit ([0-9-]+).*/\1/')
    blk=$(grep -E '\] block ' "$f" | tail -1 | sed -E 's/.*\] block (.*)/\1/')
    donec=$(grep -c '\] done ' "$f")
    its=$(grep -E '^Chain [0-9]+: Iteration' "$f" | tail -4 |
          sed -E 's/^Chain ([0-9]+): Iteration: *([0-9]+) \/ ([0-9]+).*\((Warmup|Sampling)\).*/链\1 \2\/\3 \4/' |
          paste -s -d ' ' -)
    printf '%-12s 已完成 %-3s 当前起点 %-11s %s\n' "$name" "$donec" "${cur:--}" "$blk"
    [ -n "$its" ] && printf '%-12s %s\n' "" "$its"
  done
  echo
  echo "最近完成的起点："
  grep -h '\] done [0-9]' output/logs/shard_*.log output/logs/bench_*.log 2>/dev/null | sort | tail -5 |
    sed -E 's/^\[([^]]+)\] done /  \1  /'
  if grep -qh 'ERROR' output/logs/shard_*.log 2>/dev/null; then
    echo
    echo "错误："
    grep -h 'ERROR' output/logs/shard_*.log
  fi
  echo
  echo "（Ctrl+C 只退出这个面板，拟合在后台继续；重新查看：bash run_export.sh --watch）"
}

running() { pgrep -f "export_origins.R" > /dev/null; }

if [ "${1:-}" = "--watch" ]; then
  while :; do clear; status; running || break; sleep 20; done
  echo "没有正在运行的拟合。"
  exit 0
fi

N="${1:-3}"
[ $# -gt 0 ] && shift

echo "[$(date '+%F %T')] compiling Stan model..."
if ! Rscript -e 'source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))' > output/logs/compile.log 2>&1; then
  echo "编译失败，见 output/logs/compile.log"
  exit 1
fi

for i in $(seq 1 "$N"); do
  nohup Rscript export_origins.R --shard "$i/$N" ${@+"$@"} > "output/logs/shard_${i}.log" 2>&1 &
done
sleep 5
while :; do clear; status; running || break; sleep 20; done
echo "[$(date '+%F %T')] 全部完成：$(ls output/origins/*/DONE 2>/dev/null | wc -l | tr -d ' ') 个起点"
