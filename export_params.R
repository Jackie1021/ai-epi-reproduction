# ============================================================================
# 核心流行病学参数的后验抽样（先验 / 后验分布图用）
#
# 用法: Rscript export_params.R [起点，默认 val_end]
# 与 export_origins.R 用同一份数据、同一个种子和初值方案重跑该起点（初值扰动用 R 的随机数，
# 各块并行时与原拟合不同，所以是同一后验的另一次独立抽样），额外保存
# 初始 R0（拟合窗口起点的 R0）、初始易感比例、免疫持续时间、交叉免疫、检出率、过度离散度的全部抽样。
# 输出: output/params/params_<起点>.parquet（长表：draw, parameter, strain, value）
#       并与 output/origins/<起点>/draws.parquet 的共同参数比较（应一致）
# ============================================================================
source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))

args   <- commandArgs(trailingOnly = TRUE)
origin <- if (length(args)) as.Date(args[1]) else val_end
out    <- file.path(out_dir, "params")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

sd  <- make_stan_data(origin, 0)
fit <- run_fit(sd, pars = c("susc0", "immunity_days", "cross", "obs_rate", "phi", "tau_rw",
                            "log_R0_spline_coeff", "R0_end"))
p <- extract_fit(fit, c("susc0", "immunity_days", "cross", "obs_rate", "phi", "tau_rw",
                        "log_R0_spline_coeff", "R0_end"))

# R0 在拟合窗口第一天：夹紧的 B 样条在第一天只有第一个基函数非零，log R0 = 第一个系数；
# 再经过与模型相同的平滑边界 (smooth_bounds = 1) 映射到 [R0_min, R0_max]
bound_R0 <- function(lr, lo = R0_bounds[1], hi = R0_bounds[2]) {
  a <- log(lo); b <- log(hi)
  if (smooth_bounds == 1) exp(a + (b - a) * plogis((lr - 0.5 * (a + b)) * 4 / (b - a)))
  else pmin(hi, pmax(lo, exp(lr)))
}
D <- nrow(p$susc0)
long <- function(m, name, labels = strain_names) {
  data.frame(draw = rep(seq_len(D), times = ncol(m)), parameter = name,
             strain = rep(labels, each = D), value = as.vector(m))
}
res <- rbind(
  long(bound_R0(p$log_R0_spline_coeff[, , 1]), "R0_start"),
  long(p$susc0, "susc0"),
  long(p$immunity_days, "immunity_days"),
  long(p$obs_rate, "obs_rate"),
  long(p$phi, "phi"),
  long(p$tau_rw, "tau_rw"),
  long(p$cross, "cross", cross_names),
  long(p$R0_end, "R0_end"))
res <- res[is.finite(res$value), ]
f <- file.path(out, paste0("params_", format(origin), ".parquet"))
write_parquet(res, f)

# 一致性检查：与滚动拟合导出的同一起点比较（同一种子，应相同或只差抽样误差）
bf <- file.path(origins_dir, format(origin), "draws.parquet")
if (file.exists(bf)) {
  b <- read_parquet(bf)
  for (s in strain_names) {
    cat(sprintf("%-6s R0_end median: this fit %.4f | export %.4f;  omega median: %.6f | %.6f\n", s,
                median(p$R0_end[, strain_names == s]), median(b[[paste0("R0_end_", s)]]),
                median(1 / p$immunity_days[, strain_names == s]), median(b[[paste0("omega_", s)]])))
  }
}
cf <- tryCatch(check_fit(fit), error = function(e) NULL)   # check_fit 需要这里没保存的参数时跳过
if (!is.null(cf)) cat("max Rhat", max(cf$max_rhat), "| divergent", sum(cf$divergent), "\n")
cat("wrote", f, "\n")
