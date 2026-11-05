# ============================================================================
# 新版（分块）模型与原模型（_backup/2026-09-24/ai_epi.stan）是否是同一个模型
#   1. 随机参数点上 log 后验只差一个常数（原模型多一个不进似然的 detection_rate[6]）
#   2. 共享参数的梯度完全相同
#   3. 各块 log 后验之和 = 整体模型（只差常数）
#   4. 预测、隐变量等输出完全相同；分块结果经 extract_fit 合并后也相同
# 用法: Rscript tests/test_model_equivalence.R
# ============================================================================
source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))
set.seed(42)
old_model <- stan_model(file.path(proj_dir, "tests/fixtures/ai_epi_pre_split.stan"))

sd  <- make_stan_data(val_end, 0, common_window = TRUE)              # 分块等价性要在同一时间窗上比
nb  <- sd$blocks[[1]]$num_basis
full <- block_data(sd$fit_df, strain_names, sd$blocks[[1]]$num_knots, 0)  # 一个块包含全部病原体
old_data <- list(
  T_weeks = full$data$T_weeks, N_strains = N, cases = full$data$cases,
  num_knots = full$data$num_knots, spline_degree = full$data$spline_degree,
  population = population, child_ratio = child_ratio,
  logR0_prior_mean = unname(logR0_prior_mean), logR0_prior_sd = unname(logR0_prior_sd),
  tau_rw_prior_sd = tau_rw_prior_sd,
  immunity_days_prior_median = unname(immunity_days_prior_median),
  immunity_days_prior_sdlog = unname(immunity_days_prior_sdlog),
  N_cross = N_cross, cross_map = unname(cross_map),
  cross_prior_a = vec(cross_prior_a), cross_prior_b = vec(cross_prior_b),
  T_fc_weeks = as.integer(forecast_weeks), gen_synthetic = 0L, kappa = kappa)

shell <- function(m, data, init) sampling(m, data = data, algorithm = "Fixed_param", iter = 1,
                                          chains = 1, init = list(init), refresh = 0, seed = 1)
rand_pars <- function() list(
  susc0 = runif(N, 0.2, 0.8), log_inf0 = log(runif(N, 1e-5, 1e-3)),
  z_spline = matrix(rnorm(N * nb, 0, 0.5), N, nb), tau_rw = runif(N, 0.05, 0.4),
  immunity_days = runif(N, 100, 900), cross = vec(runif(N_cross, 0.05, 0.5)),
  obs_rate = runif(N, 0.1, 0.8), phi = runif(N, 2, 20))
to_old <- function(p, d6) {
  q <- p[c("susc0", "log_inf0", "z_spline", "tau_rw", "immunity_days", "cross", "phi")]
  q$detection_rate <- c(p$obs_rate[1:5], d6)
  q$hospitalization_rate <- p$obs_rate[6]
  q
}
to_block <- function(p, b) {
  q <- list(susc0 = vec(p$susc0[b$idx]), log_inf0 = vec(p$log_inf0[b$idx]),
            z_spline = p$z_spline[b$idx, , drop = FALSE], tau_rw = vec(p$tau_rw[b$idx]),
            immunity_days = vec(p$immunity_days[b$idx]),
            obs_rate = vec(p$obs_rate[b$idx]), phi = vec(p$phi[b$idx]))
  q$cross <- vec(p$cross[b$cross_ids])                             # 没有交叉免疫的块为长度 0
  q
}

p0 <- rand_pars()
f_old  <- shell(old_model, old_data, to_old(p0, 0.5))
f_full <- shell(model, full$data, p0)
f_blk  <- lapply(sd$blocks, function(b) shell(model, b$data, to_block(p0, b)))

n_shared <- 2 * N + N * nb + 2 * N + N_cross         # susc0, log_inf0, z_spline, tau_rw, immunity, cross
d_old <- d_blk <- numeric(0)
max_grad_diff <- 0
for (i in 1:6) {
  p  <- rand_pars()
  d6 <- runif(1, 0.1, 0.9)
  u_old  <- unconstrain_pars(f_old, to_old(p, d6))
  u_full <- unconstrain_pars(f_full, p)
  lp_old  <- log_prob(f_old, u_old)
  lp_full <- log_prob(f_full, u_full)
  # 原模型多出的 detection_rate[6] ~ beta(6, 4)（含 logit 变换的 Jacobian）
  d_old <- c(d_old, lp_old - lp_full - (6 * log(d6) + 4 * log(1 - d6)))
  lp_blocks <- sum(sapply(seq_along(sd$blocks), function(j)
    log_prob(f_blk[[j]], unconstrain_pars(f_blk[[j]], to_block(p, sd$blocks[[j]])))))
  d_blk <- c(d_blk, lp_full - lp_blocks)
  g_old  <- grad_log_prob(f_old, u_old)
  g_full <- grad_log_prob(f_full, u_full)
  shared <- c(seq_len(n_shared), n_shared + 1:5)      # + obs_rate[1:5] <-> detection_rate[1:5]
  gd <- max(abs(g_old[shared] - g_full[shared]),
            abs(g_old[n_shared + 7] - g_full[n_shared + 6]),                  # hospitalization_rate
            abs(tail(g_old, N) - tail(g_full, N)))                             # phi
  max_grad_diff <- max(max_grad_diff, gd / max(1, max(abs(g_full))))
}
ok1 <- sd(d_old) < 1e-8
ok2 <- max_grad_diff < 1e-9
ok3 <- sd(d_blk) < 1e-8
cat(sprintf("1. log posterior old - new is constant: sd %.1e  %s\n", sd(d_old), if (ok1) "OK" else "FAIL"))
cat(sprintf("2. gradients of shared parameters match: max rel diff %.1e  %s\n", max_grad_diff,
            if (ok2) "OK" else "FAIL"))
cat(sprintf("3. sum over blocks = joint model: sd %.1e  %s\n", sd(d_blk), if (ok3) "OK" else "FAIL"))

# ---- 4. 输出（确定性部分）----
p  <- rand_pars()
o_old  <- rstan::extract(shell(old_model, old_data, to_old(p, 0.5)))
o_full <- rstan::extract(shell(model, full$data, p))
bf <- structure(list(blocks = lapply(sd$blocks, function(b) list(
  fit = shell(model, b$data, to_block(p, b)), strains = b$strains, idx = b$idx,
  cross_ids = b$cross_ids, dates = b$dates, minutes = 0)), dates = sd$fit_df$date), class = "blockfit")
vars <- c("mu_cases", "inc_week", "S_week", "Rt_week", "R0_week", "R0_end", "mu_fc", "Rt_fc", "S_fc",
          "state_end", "omega", "obs_scale", "cross_matrix")
o_blk <- extract_fit(bf, vars)
# 合并结果在各病原体拟合起点之前是 NA（没有拟合），只比较有值的部分
rel <- function(a, b) { ok <- !is.na(a) & !is.na(b); max(abs(a[ok] - b[ok]) / pmax(abs(b[ok]), 1e-12)) }
res <- sapply(vars, function(v) c(old_vs_new = rel(o_full[[v]], o_old[[v]]),
                                  blocks_vs_joint = rel(o_blk[[v]], o_full[[v]])))
print(signif(t(res), 2))
ok4 <- all(res < 1e-9)
cat(sprintf("4. outputs identical (old / joint / merged blocks): %s\n", if (ok4) "OK" else "FAIL"))
if (!all(ok1, ok2, ok3, ok4)) quit(status = 1)
cat("\nALL EQUIVALENCE TESTS PASSED\n")
