# ============================================================================
# 诊断 HMC 为什么慢（按拟合块分别看）
#   1. 单次梯度耗时
#   2. 从多个初值做优化：众数处的 log 后验是否一致（不一致 = 多峰）
#   3. 众数处 Hessian 的条件数（对角缩放后 ≈ 对角质量矩阵 HMC 看到的）和估计的蛙跳步数
#   4. 最"平"的几个方向由哪些参数组成（= 数据约束不了的方向）
#   5. 众数处 R0 顶到截断的周数比例（那里梯度为 0）
# 用法: Rscript diagnose_sampling.R [起点，默认 val_end]
#   可用环境变量临时改设置对比，例如 KNOT_WEEKS=10 TAU_SD=0.1 RW_ORDER=2 SMOOTH=1 Rscript diagnose_sampling.R
# ============================================================================
source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))
a <- commandArgs(trailingOnly = TRUE)
origin <- if (length(a)) as.Date(a[1]) else val_end
env_num <- function(k, v) if (nzchar(Sys.getenv(k))) as.numeric(Sys.getenv(k)) else v
knot_spacing_weeks <- env_num("KNOT_WEEKS", knot_spacing_weeks)
tau_rw_prior_sd    <- env_num("TAU_SD", tau_rw_prior_sd)
rw_order           <- env_num("RW_ORDER", rw_order)
smooth_bounds      <- env_num("SMOOTH", smooth_bounds)
cat(sprintf("\norigin %s | knots every %g weeks | tau prior sd %g | rw_order %d | smooth_bounds %d\n",
            format(origin), knot_spacing_weeks, tau_rw_prior_sd, rw_order, smooth_bounds))
sd <- make_stan_data(origin, 0)

kappa_of <- function(v) { v <- v[v > 1e-8 * max(v)]; max(v) / min(v) }
fd_hessian <- function(f, u, h = 1e-4) {            # 中心差分，基于含 Jacobian 的梯度
  H <- sapply(seq_along(u), function(j) {
    e <- replace(numeric(length(u)), j, h)
    (grad_log_prob(f, u + e) - grad_log_prob(f, u - e)) / (2 * h)
  })
  -(H + t(H)) / 2
}
par_names <- c("susc0", "log_inf0", "z_spline", "tau_rw", "immunity_days", "cross", "obs_rate", "phi")

summary_rows <- list()
for (bn in names(sd$blocks)) {
  b <- sd$blocks[[bn]]
  n <- length(b$strains)
  nb <- b$num_basis
  cat(sprintf("\n[%s] from %s | %d weeks | %d spline basis\n", bn, min(b$dates), length(b$dates), nb))
  init_fun <- make_init(nb, b)
  full_init <- function() { p <- init_fun(); if (is.null(p$cross)) p$cross <- vec(numeric(0)); p }
  f0 <- sampling(model, data = b$data, algorithm = "Fixed_param", iter = 1, chains = 1,
                 init = list(init_fun()), refresh = 0)
  u0 <- unconstrain_pars(f0, full_init())
  t_grad <- system.time(for (i in 1:30) grad_log_prob(f0, u0))[["elapsed"]] / 30

  # 无约束向量里每个位置的名字（z_spline 的排列顺序用哨兵值解码）
  nc <- length(b$cross_ids)
  nm <- c(paste0("susc0[", b$strains, "]"), paste0("log_inf0[", b$strains, "]"), rep(NA, n * nb),
          paste0("tau_rw[", b$strains, "]"), paste0("immunity_days[", b$strains, "]"),
          if (nc) paste0("cross[", cross_names[b$cross_ids], "]"),
          paste0("obs_rate[", b$strains, "]"), paste0("phi[", b$strains, "]"))
  p <- full_init(); p$z_spline <- outer(1:n, 1:nb, function(i, j) i * 1000 + j)
  zpos <- 2 * n + seq_len(n * nb)
  code <- round(unconstrain_pars(f0, p)[zpos])
  nm[zpos] <- sprintf("z_spline[%s,%d]", b$strains[code %/% 1000], code %% 1000)

  runs <- list()
  for (i in 1:4) {
    set.seed(100 + i)
    o <- NULL
    invisible(capture.output(o <- tryCatch(
      optimizing(model, data = b$data, init = init_fun(), iter = 20000, algorithm = "LBFGS",
                 as_vector = FALSE), error = function(e) NULL)))
    if (is.null(o)) next
    pp <- o$par[par_names]
    if (is.null(pp$cross) || !length(pp$cross)) pp$cross <- vec(numeric(0))
    u <- unconstrain_pars(f0, pp)
    runs[[length(runs) + 1]] <- list(o = o, u = u, lp = log_prob(f0, u))
  }
  lps <- sapply(runs, `[[`, "lp")
  best <- runs[[which.max(lps)]]
  H <- fd_hessian(f0, best$u)
  d <- sqrt(pmax(diag(H), 1e-12))
  es <- eigen(H / outer(d, d), symmetric = TRUE)
  kd <- kappa_of(es$values)
  R0w <- best$o$par$R0_week
  clamp <- mean(R0w <= R0_bounds[1] + 1e-6 | R0w >= R0_bounds[2] - 1e-6)
  summary_rows[[bn]] <- data.frame(
    block = bn, params = length(u0), grad_ms = round(1000 * t_grad, 2),
    optima_spread = round(diff(range(lps)), 1), kappa_diag = signif(kd, 2),
    steps = round(sqrt(kd)), flat_dirs = sum(es$values <= 1e-8 * max(es$values)),
    clamped_weeks = sprintf("%.0f%%", 100 * clamp))
  k <- length(es$values)
  cat(sprintf("\n[%s] flattest directions:\n", bn))
  for (j in k:max(1, k - 2)) {
    v <- es$vectors[, j]; top <- order(abs(v), decreasing = TRUE)[1:min(5, length(v))]
    cat(sprintf("  eig %.1e: %s\n", es$values[j], paste(sprintf("%s(%.2f)", nm[top], v[top]), collapse = ", ")))
  }
}
cat("\nsummary (optima_spread > 5 = several local optima; steps ~ leapfrog steps per iteration with a diagonal metric):\n")
print(do.call(rbind, summary_rows), row.names = FALSE)
