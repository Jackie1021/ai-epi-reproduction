# ============================================================================
# 滚动预测起点：每个起点单独拟合一次 Stan，导出 Python / LLM 实验需要的全部数据
#
# 用法（终端）:
#   Rscript export_origins.R                       # 默认起点序列（见 ai_epi_core.R 的 bench_* 设置）
#   Rscript export_origins.R 2025-06-07 2025-07-05 # 只跑指定起点
#   Rscript export_origins.R --shard 2/3           # 默认序列分 3 份只跑第 2 份（并行用，见 run_export.sh）
#   Rscript export_origins.R --smoke 2025-06-07    # 冒烟测试：极少迭代，只检查导出流程，结果不能用
#   Rscript export_origins.R 2024-10-26 --seed 2026 --starts 8 --outdir origins_refit   # 重拟合到单独目录
#
# 每个起点写到 output/origins/<起点>/（冒烟测试写到 output/origins_smoke/）:
#   draws.parquet              每个后验样本一行：起点时的 S/E/I/R、R0_end、omega、交叉免疫矩阵、
#                              观测尺度 obs_scale、phi —— Python 模拟器从这里接着往前推
#   latent_weekly.parquet      拟合期每周 Rt、易感比例、R0、真实发病、期望病例的分位数（给 LLM 看）
#   stan_fc_quantiles.parquet  Stan 三种 R0 外推方式的预测分位数（基线）
#   stan_mu_fc.parquet         每个样本在三种外推下的期望病例（校验 Python 模拟器用）
#   meta.json                  起点、拟合窗口、固定参数、诊断、运行时间、文件指纹
#   DONE                       完成标记：有它的起点重跑时自动跳过（中断后直接重跑即可续上）
# ============================================================================
source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))
parallel_blocks <- FALSE   # 一个起点内各块依次跑；多个起点由 run_export.sh 分片并行，CPU 更满

# ---- 命令行参数 ----
args  <- commandArgs(trailingOnly = TRUE)
smoke <- "--smoke" %in% args
args  <- setdiff(args, "--smoke")
take_opt <- function(flag) {                  # 取出 "--flag 值" 并从 args 里删掉
  if (!flag %in% args) return(NULL)
  i <- which(args == flag)
  v <- args[i + 1]
  args <<- args[-c(i, i + 1)]
  v
}
shard  <- take_opt("--shard")
shard  <- if (is.null(shard)) c(1L, 1L) else as.integer(strsplit(shard, "/")[[1]])
iter   <- take_opt("--iter")                  # 例如 --iter 1000 --warmup 500，覆盖 core 里的采样设置
warmup <- take_opt("--warmup")
if (!is.null(iter))   sampling_args$iter   <- as.integer(iter)
if (!is.null(warmup)) sampling_args$warmup <- as.integer(warmup)
seed   <- take_opt("--seed")                  # 例如 --seed 2026 --starts 8：换种子、增加初值优化次数（重拟合不收敛的起点用）
starts <- take_opt("--starts")
outdir <- take_opt("--outdir")               # 例如 --outdir origins_refit：写到 output/origins_refit/（重拟合时和原结果对比用）
if (!is.null(seed))   sampling_args$seed <- as.integer(seed)
if (!is.null(starts)) init_n_starts      <- as.integer(starts)
adapt  <- take_opt("--adapt-delta")          # 例如 --adapt-delta 0.99 --max-treedepth 13（修正案 14）；不给就用 core 里的设置
treed  <- take_opt("--max-treedepth")
if (!is.null(adapt)) sampling_args$control$adapt_delta   <- as.numeric(adapt)
if (!is.null(treed)) sampling_args$control$max_treedepth <- as.integer(treed)

origins <- if (length(args) > 0) as.Date(args) else default_origins()
bad <- origins[!origins %in% raw_data$date]
if (length(bad) > 0) stop("这些起点不在数据日期里: ", paste(bad, collapse = ", "))
origins <- origins[(seq_along(origins) - 1) %% shard[2] == shard[1] - 1]

if (smoke) {
  origins_dir   <- file.path(out_dir, "origins_smoke")
  sampling_args <- modifyList(sampling_args, list(
    iter = 40, warmup = 20, chains = 2, cores = 2,
    control = list(adapt_delta = 0.8, max_treedepth = 6)))
  init_n_starts <- 1
}
if (!is.null(outdir)) origins_dir <- file.path(out_dir, outdir)
dir.create(origins_dir, showWarnings = FALSE, recursive = TRUE)

export_pars <- c(core_pars, "state_end", "omega", "cross_matrix", "obs_scale", "R0_end",
                 "mu_fc", "pred_fc", "Rt_week", "S_week", "R0_week", "inc_week", "mu_cases")

state_names <- paste0(rep(c("S", "E", "I", "R"), times = N), "_", rep(strain_names, each = 4))

log_msg <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), ..., "\n", sep = "")

# ---- 单个起点：拟合 + 导出 ----
export_origin <- function(origin) {
  od <- file.path(origins_dir, as.character(origin))
  if (file.exists(file.path(od, "DONE"))) {
    log_msg("skip ", format(origin), " (DONE)")
    return(invisible(TRUE))
  }
  dir.create(od, showWarnings = FALSE, recursive = TRUE)
  # 锁：另一个进程正在拟合这个起点就跳过（进程已不存在的旧锁会被接管）
  lock <- file.path(od, "RUNNING")
  if (file.exists(lock)) {
    pid <- suppressWarnings(as.integer(readLines(lock, n = 1)))
    if (!is.na(pid) && pid != Sys.getpid() && isTRUE(tools::pskill(pid, 0L))) {
      log_msg("skip ", format(origin), " (running in pid ", pid, ")")
      return(invisible(TRUE))
    }
  }
  writeLines(as.character(Sys.getpid()), lock)
  on.exit(unlink(lock), add = TRUE)
  unlink(file.path(od, "ERROR.txt"))
  log_msg("fit ", format(origin), if (smoke) " [smoke]",
          sprintf(" | iter %d, warmup %d", sampling_args$iter, sampling_args$warmup))

  sd <- make_stan_data(origin, 0)
  t0 <- Sys.time()
  fit <- run_fit(sd, pars = export_pars)
  runtime_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  fit_dates <- sd$fit_df$date

  post <- extract_fit(fit, c("state_end", "omega", "cross_matrix", "obs_scale", "R0_end", "phi",
                              "mu_fc"))
  n_total <- dim(post$R0_end)[1]
  idx <- as.integer(unique(round(seq(1, n_total, length.out = min(bench_export_draws, n_total)))))

  # 每个后验样本一行
  draws <- data.frame(draw = idx)
  se <- post$state_end[idx, , drop = FALSE]
  colnames(se) <- state_names
  draws <- cbind(draws, se)
  add_cols <- function(df, m, prefix) {
    m <- m[idx, , drop = FALSE]
    colnames(m) <- paste0(prefix, "_", strain_names)
    cbind(df, m)
  }
  draws <- add_cols(draws, post$R0_end, "R0_end")
  draws <- add_cols(draws, post$omega, "omega")
  draws <- add_cols(draws, post$obs_scale, "obs_scale")
  draws <- add_cols(draws, post$phi, "phi")
  for (j in seq_len(N)) for (k in seq_len(N))
    draws[[paste0("C_", strain_names[j], "_", strain_names[k])]] <- post$cross_matrix[idx, j, k]
  write_parquet(draws, file.path(od, "draws.parquet"))

  # 每个样本在三种外推下的期望病例（draws x 3 x H x N）
  mf <- post$mu_fc[idx, , , , drop = FALSE]
  g  <- expand.grid(di = seq_along(idx), fc_mode = 1:3, horizon = seq_len(forecast_weeks),
                    k = seq_len(N))
  mu_long <- data.frame(draw = idx[g$di], fc_mode = g$fc_mode, horizon = g$horizon,
                        strain = strain_names[g$k], mu = as.vector(mf))
  write_parquet(mu_long[mu_long$strain %in% fit_strains, ], file.path(od, "stan_mu_fc.parquet"))

  # 拟合期每周隐变量的分位数
  lat <- extract_fit(fit, c("Rt_week", "S_week", "R0_week", "inc_week", "mu_cases"))
  latent <- bind_rows(lapply(names(lat), function(v) qs(lat[[v]], fit_dates) %>%
                               mutate(variable = sub("_week$", "", v)))) %>%
    filter(strain %in% fit_strains)                     # 历史期没有 COVID
  write_parquet(latent, file.path(od, "latent_weekly.parquet"))
  rm(lat)

  # Stan 自己的三种外推预测（基线）
  write_parquet(fc_quantiles(fit, origin), file.path(od, "stan_fc_quantiles.parquet"))

  # 诊断
  cf <- check_fit(fit)
  diagnostics <- list(
    max_rhat        = max(cf$max_rhat),
    min_n_eff       = min(cf$min_n_eff),
    n_divergent     = sum(cf$divergent),
    n_max_treedepth = sum(cf$treedepth_hits),
    blocks          = cf)

  meta <- list(
    origin = as.character(origin), fit_start = as.character(min(fit_dates)),
    fit_end = as.character(max(fit_dates)), T_weeks = length(fit_dates),
    strain_start = as.list(setNames(format(strain_start), names(strain_start))),
    blocks = lapply(sd$blocks, function(b) list(strains = b$strains, start = format(min(b$dates)),
                                                weeks = length(b$dates), num_knots = b$num_knots)),
    strain_names = strain_names, fitted_strains = fit_strains, study = study,
    window = if (study == "historical") list(weeks_max = hist_window_weeks, trough_weeks = hist_trough_weeks,
                                             trough_tol = hist_trough_tol),
    state_order = c("S", "E", "I", "R"),
    forecast_weeks = forecast_weeks, fc_mode_labels = fc_mode_labels,
    fc_quantile_levels = fc_quantile_levels,
    population = sd$data$population, child_ratio = sd$data$child_ratio,
    constants = list(sigma_E = unname(sigma_E_fixed[strain_names]),
                     gamma_I = unname(gamma_I_fixed[strain_names]),
                     R0_min = R0_bounds[1], R0_max = R0_bounds[2]),
    model = list(onset_anchor = as.character(start_date), blocks = names(fit_blocks),
                 rw_order = rw_order, smooth_bounds = smooth_bounds,
                 knot_spacing_weeks = knot_spacing_weeks, init_n_starts = init_n_starts),
    n_draws_total = n_total, n_draws_exported = length(idx),
    sampling = list(iter = sampling_args$iter, warmup = sampling_args$warmup,
                    chains = sampling_args$chains,
                    adapt_delta = sampling_args$control$adapt_delta,
                    max_treedepth = sampling_args$control$max_treedepth,
                    metric = sampling_args$control$metric %||% "diag_e"),
    smoke = smoke, runtime_min = runtime_min, diagnostics = diagnostics,
    files = list(stan_md5 = unname(tools::md5sum(model_file)),
                 data_md5 = unname(tools::md5sum(data_path))),
    versions = list(R = R.version.string, rstan = as.character(packageVersion("rstan"))),
    created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
  write_json(meta, file.path(od, "meta.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)

  writeLines(format(Sys.time()), file.path(od, "DONE"))
  log_msg("done ", format(origin), sprintf(" | %.1f min | max Rhat %.3f | divergent %d | ",
                                   runtime_min, diagnostics$max_rhat, diagnostics$n_divergent),
          paste(sprintf("%s %.1f min", cf$block, cf$minutes), collapse = ", "))
  rm(fit, post); invisible(gc())
  invisible(TRUE)
}

log_msg(length(origins), " origins", if (shard[2] > 1) sprintf(" (shard %d/%d)", shard[1], shard[2]),
        ": ", paste(origins, collapse = " "))
for (o in as.list(origins)) {
  ok <- tryCatch(export_origin(o), error = function(e) {
    od <- file.path(origins_dir, as.character(o))
    dir.create(od, showWarnings = FALSE, recursive = TRUE)
    writeLines(conditionMessage(e), file.path(od, "ERROR.txt"))
    log_msg("ERROR ", format(o), ": ", conditionMessage(e))
    FALSE
  })
}
log_msg("all done")
