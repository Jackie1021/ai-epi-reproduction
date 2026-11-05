# ============================================================================
# 共享部分：设置 + 数据 + 工具函数 + 编译模型
#   ai_epi.R（主分析）和 export_origins.R（滚动起点导出）都 source 这个文件，
#   先验、时间切分、采样参数只在这里改一次
#
# 提速（后验不变）:
#   1. 分块拟合：没有交叉免疫联系的病原体分开拟合（流感三型 / COVID / RSV / HFMD）
#   2. 每块先从多个初值优化，取最好的峰作为各条链的起点（后验有多个局部峰）
#   3. 稠密质量矩阵 dense_e、max_treedepth 12、iter 1000 / warmup 500
#   拟合结果是 blockfit 对象，用 extract_fit / summary_fit / check_fit / as_df_fit 取结果，
#   返回格式与 rstan::extract 等相同（病原体维按 strain_names 排）
# ============================================================================
library(rstan)
library(ggplot2)
library(dplyr)
library(tidyr)
library(arrow)      # 导出 parquet；如未安装: install.packages("arrow")
library(jsonlite)   # 导出 meta.json

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# ============================================================================
# 0. 设置
# ============================================================================
proj_dir   <- normalizePath(Sys.getenv("AI_EPI_ROOT", "."), mustWork = TRUE)
data_path  <- file.path(proj_dir, "Data/HK_weekly_6pathogens.csv")   # Data/build_hk_weekly.py 生成（CHP 数据，日期 = 周末周六）
model_file <- file.path(proj_dir, "ai_epi.stan")  # 原来指向 Downloads 里的旧版（没有 seirs_cross_step）
out_dir    <- file.path(proj_dir, "output")
exp_dir    <- file.path(out_dir, "export")        # 给 Python 的 parquet 文件
dir.create(exp_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 拟合起点：每个病原体从自己这一波流行的起点开始 ----
# 在名义起点 start_date 之后 onset_peak_weeks 周内找这一波的峰（3 周滑动平均），再从峰往前
# onset_lookback_weeks 周内找谷底；谷底较平时取离峰最近、且不超过最低值 (1 + onset_tolerance) 倍的那一周
start_date            <- as.Date("2024-01-06")   # 名义起点（原为 2022-12-31）
# 研究期：main = 2024-2026 主研究（冻结设计）；historical = 2015-2019 历史期重复研究（修正案 8）
#   AI_EPI_STUDY=historical Rscript export_origins.R ...
study <- Sys.getenv("AI_EPI_STUDY", "main")
stopifnot(study %in% c("main", "historical"))
# 主研究只用这天以后的观测（与冻结设计一致）；数据文件从 2014-01-04 开始，更早的年份只给历史期研究
data_from <- as.Date(Sys.getenv("AI_EPI_DATA_FROM", if (study == "historical") "2014-01-04" else "2023-01-07"))
onset_peak_weeks      <- 26
onset_lookback_weeks  <- 16
onset_tolerance       <- 0.10
strain_start_override <- c()          # 手动指定，例如 c(HFMD = "2024-03-16")；优先于自动识别
train_end  <- as.Date("2025-03-08")   # 神经网络训练集截止
val_end    <- as.Date("2025-06-07")   # 验证集截止 = Stan 主拟合截止

# ---- LLM 实验用的滚动预测起点（export_origins.R）----
origins_dir             <- file.path(out_dir, if (study == "historical") "origins_hist" else "origins")   # 每个起点一个子目录
bench_origin_first      <- NULL                            # 最早起点：所有病原体都至少有 16 周数据（读数据后计算）
bench_origin_anchor     <- val_end                         # 起点网格锚定在 val_end，向前向后每隔 N 周
bench_origin_step_weeks <- 4
bench_origin_last       <- as.Date("2026-04-11")          # 研究期终点：OAG 航空旅客数据到 2026 年 3 月，起点需要之前的完整月份

# ---- 历史期研究（修正案 8）：同一个模型、先验和采样设置，只有拟合窗口不同 ----
# 起点：同一个 4 周网格上 2015-01-31 至 2019-12-28 的 65 个（最后一个预测到 2020-01-25，学校停课之前）
# 窗口：W0 = max(数据第一周, 起点 - 130 周)；每个病原体从窗口前 26 周里的谷底开始
#       （3 周居中滑动平均第一次落到这段最小值的 1.1 倍以内的那一周），只用起点之前的数据
hist_first_origin <- as.Date("2015-01-31")
hist_last_origin  <- as.Date("2019-12-28")
hist_window_weeks <- 130
hist_trough_weeks <- 26
hist_trough_tol   <- 0.10
bench_export_draws      <- 1000                            # 每个起点导出多少个后验样本给 Python

forecast_weeks <- 4                   # 预测 1-4 周
fc_mode_plot   <- 1                   # 画图用哪种 R0 外推: 1 固定（唯一用于结果的方式，修正案 9）/ 2 去年同期 / 3 缩放后的去年同期
fc_mode_labels <- c("constant R0", "last-year R0", "scaled last-year R0")

gen_synthetic  <- 1                   # 1 = 生成模拟轨迹
kappa          <- 0.5                 # 模拟轨迹的 R0 扰动强度（0 = 不扰动，即后验预测）
n_export_draws <- 1000                # 隐变量 / 后验预测导出的样本数（模拟轨迹导出全部样本）

run_rolling_refits <- FALSE           # TRUE = 在测试期滚动重拟合，生成 Stan 的基线预测
refit_every_weeks  <- 4

fc_quantile_levels <- c(0.01, 0.025, seq(0.05, 0.95, by = 0.05), 0.975, 0.99)   # FluSight 23 个分位数

strain_names  <- c("B", "H3", "H1", "COVID", "RSV", "HFMD")
fit_strains   <- if (study == "historical") setdiff(strain_names, "COVID") else strain_names   # 2020 年前没有 COVID
strain_labels <- c(B = "Influenza B", H3 = "Influenza A(H3N2)", H1 = "Influenza A(H1N1)",
                   COVID = "COVID-19", RSV = "RSV", HFMD = "HFMD")
N <- length(strain_names)

# ---- 人群和文献固定参数（原来写在 Stan 里）----
population    <- 7524000
child_ratio   <- 0.029
pop_k_all     <- population * c(B = 1, H3 = 1, H1 = 1, COVID = 1, RSV = 1, HFMD = child_ratio)
sigma_E_fixed <- c(B = 1 / 3, H3 = 1 / 3, H1 = 1 / 3, COVID = 1 / 3, RSV = 1 / 4, HFMD = 1 / 4.5)
gamma_I_fixed <- c(B = 1 / 5, H3 = 1 / 5, H1 = 1 / 5, COVID = 1 / 3.5, RSV = 1 / 7, HFMD = 1 / 7)

# ---- 样条 ----
knot_spacing_weeks <- 5
spline_degree      <- 3
logR0_prior_mean   <- c(B = 0.30, H3 = 0.40, H1 = 0.35, COVID = 0.75, RSV = 0.65, HFMD = 1.00)
logR0_prior_sd     <- c(B = 0.35, H3 = 0.35, H1 = 0.35, COVID = 0.45, RSV = 0.35, HFMD = 0.40)
tau_rw_prior_sd    <- 0.3
R0_bounds          <- c(0.5, 4.0)    # R0 的下限和上限
rw_order           <- 1               # 1 = 原模型；2 = 二阶随机游走（更平滑，建议同时把 tau_rw_prior_sd 调到 0.05~0.1）
smooth_bounds      <- 1               # 1 = 平滑边界（截断处梯度不再为 0，减少发散）；0 = 原模型的硬截断 [0.5, 4]

# ---- 免疫持续时间先验 (天)：粗略假设，请按文献核实 ----
immunity_days_prior_median <- c(B = 730, H3 = 365, H1 = 545, COVID = 240, RSV = 300, HFMD = 365)
immunity_days_prior_sdlog  <- c(B = 0.5, H3 = 0.5, H1 = 0.5, COVID = 0.5, RSV = 0.5, HFMD = 0.5)

# ---- 观测比例和过度离散的先验（原来写在 Stan 里）----
hosp_strain     <- "HFMD"             # 这个病原体的观测比例是住院比例（结果里仍叫 hospitalization_rate）
obs_prior_a     <- c(B = 2, H3 = 5, H1 = 3, COVID = 5, RSV = 2, HFMD = 6)   # obs_rate ~ beta(a, b)
obs_prior_b     <- c(B = 8, H3 = 5, H1 = 7, COVID = 5, RSV = 8, HFMD = 4)
obs_init        <- c(B = 0.2, H3 = 0.5, H1 = 0.3, COVID = 0.5, RSV = 0.2, HFMD = 0.6)
phi_prior_shape <- c(B = 5, H3 = 5, H1 = 5, COVID = 8, RSV = 5, HFMD = 3)   # phi ~ gamma(shape, rate)
phi_prior_rate  <- c(B = 0.5, H3 = 0.5, H1 = 0.5, COVID = 1, RSV = 0.5, HFMD = 0.3)

# ---- 交叉免疫结构 ----
# 每一行: 感染 from 之后对 to 的保护，使用第 param 个参数；同一 param 的几对共享一个值
# 默认只考虑流感之间：H3 <-> H1 一个参数，甲型 <-> 乙型 一个参数
cross_pairs <- data.frame(
  from  = c("H3", "H1", "H3", "B",  "H1", "B"),
  to    = c("H1", "H3", "B",  "H3", "B",  "H1"),
  param = c(1,    1,    2,    2,    2,    2)
)
cross_names   <- c("H3-H1", "A-B")    # 参数名（用于画图）
cross_prior_a <- c(2, 1)              # cross ~ beta(a, b)
cross_prior_b <- c(5, 9)              # 先验均值: H3-H1 约 0.29，A-B 约 0.10（假设，请核实）

# ---- 拟合方式 ----
split_blocks    <- TRUE               # TRUE = 按交叉免疫联系分块拟合（后验与一起拟合完全相同）
parallel_blocks <- !interactive()     # 各块同时跑（终端 Rscript）；RStudio 里依次跑，避免 fork
init_n_starts   <- 4                  # 每块从几个随机初值做优化，取最好的峰作为各条链的起点
sampling_args <- list(iter = 1000, warmup = 500, chains = 4, cores = 4, seed = 2025,
                      control = list(adapt_delta = 0.98, max_treedepth = 12, metric = "dense_e"))

stopifnot(forecast_weeks >= 1, forecast_weeks <= 52, fc_mode_plot %in% 1:3, rw_order %in% 1:2,
          smooth_bounds %in% 0:1)

# ============================================================================
# 1. 读数据 + 工具函数
# ============================================================================
cat("=== Loading data ===\n")
all_data <- read.csv(data_path)
all_data$date <- as.Date(all_data$date, format = "%Y/%m/%d")
all_data <- all_data[order(all_data$date), c("date", strain_names)]
for (s in strain_names) all_data[[s]] <- suppressWarnings(as.numeric(trimws(all_data[[s]])))
all_data <- all_data[all_data$date >= data_from, ]

# 每个病原体这一波流行的起点（谷底）和峰
# 只用名义起点后 onset_peak_weeks 周以内的数据（即第一个预测起点之前），拟合起点不会用到之后的信息
detect_onset <- function(dates, y) {
  keep  <- dates <= start_date + 7 * onset_peak_weeks
  dates <- dates[keep]
  y     <- y[keep]
  ma <- as.numeric(stats::filter(y, rep(1 / 3, 3), sides = 2))
  ma[is.na(ma)] <- y[is.na(ma)]
  pw <- which(dates >= start_date & dates <= start_date + 7 * onset_peak_weeks & !is.na(ma))
  pk <- pw[which.max(ma[pw])]
  lb <- which(dates >= dates[pk] - 7 * onset_lookback_weeks & dates <= dates[pk] & !is.na(ma))
  near_min <- lb[ma[lb] <= min(ma[lb]) * (1 + onset_tolerance)]
  c(start = dates[max(near_min)], peak = dates[pk])
}
onsets <- do.call(rbind, lapply(strain_names, function(s) {
  o <- detect_onset(all_data$date, all_data[[s]])
  data.frame(strain = s, start = o[["start"]], peak = o[["peak"]])
}))
if (length(strain_start_override))
  onsets$start[match(names(strain_start_override), onsets$strain)] <- as.Date(strain_start_override)
strain_start <- setNames(onsets$start, onsets$strain)       # 每个病原体的拟合起点（Date）
fit_start_global <- min(strain_start)
bench_origin_first <- max(max(strain_start) + 7 * 16, start_date + 7 * onset_peak_weeks)

raw_data <- all_data[all_data$date >= fit_start_global, ]

# 历史期：每个起点有自己的窗口和各病原体起点（make_stan_data 里按起点设置 strain_start）
hist_strain_start <- function(origin) {
  w0 <- max(min(all_data$date), origin - 7 * hist_window_weeks)
  d  <- all_data[all_data$date >= w0 & all_data$date <= origin, ]
  st <- setNames(rep(as.Date(NA), N), strain_names)
  for (s in fit_strains) {
    ma <- as.numeric(stats::filter(d[[s]], rep(1 / 3, 3), sides = 2))
    ma[is.na(ma)] <- d[[s]][is.na(ma)]
    rng <- which(d$date <= w0 + 7 * hist_trough_weeks & !is.na(ma))
    st[[s]] <- d$date[rng[ma[rng] <= min(ma[rng]) * (1 + hist_trough_tol)][1]]
  }
  st
}
if (study == "historical") {
  raw_data <- all_data
  strain_start <- hist_strain_start(hist_first_origin)
  cat("Historical study: origins", format(hist_first_origin), "to", format(hist_last_origin),
      "| window", hist_window_weeks, "weeks | pathogens", paste(fit_strains, collapse = ", "), "\n")
}
cat("Data available:", as.character(range(raw_data$date)), "\n")
cat("Fit start by pathogen:", paste(sprintf("%s %s", names(strain_start), format(strain_start)),
                                    collapse = " | "), "\n")

split_of <- function(d) ifelse(d <= train_end, "train", ifelse(d <= val_end, "val", "test"))

# LLM 实验的默认起点：锚定 bench_origin_anchor，每 bench_origin_step_weeks 周一个，
# 最早 bench_origin_first，最晚要留出 forecast_weeks 周的观测用来评分
default_origins <- function() {
  if (study == "historical") return(seq(hist_first_origin, hist_last_origin, by = 7 * bench_origin_step_weeks))
  last_scorable <- max(raw_data$date) - 7 * forecast_weeks
  step <- 7 * bench_origin_step_weeks
  back <- seq(bench_origin_anchor, bench_origin_first, by = -step)
  fwd  <- if (bench_origin_anchor <= last_scorable)
    seq(bench_origin_anchor, last_scorable, by = step) else as.Date(character(0))
  o <- sort(unique(c(back[back >= bench_origin_first], fwd)))
  o[o <= bench_origin_last]
}

N_cross   <- length(cross_names)
cross_map <- matrix(0L, N, N, dimnames = list(strain_names, strain_names))
for (r in seq_len(nrow(cross_pairs)))
  cross_map[cross_pairs$from[r], cross_pairs$to[r]] <- as.integer(cross_pairs$param[r])

# 分块：交叉免疫图的连通分量（默认 流感三型 / COVID / RSV / HFMD）
fit_blocks <- if (split_blocks) {
  comp <- setNames(seq_len(N), strain_names)
  repeat {
    changed <- FALSE
    for (r in seq_len(nrow(cross_pairs))) {
      a <- comp[[cross_pairs$from[r]]]; b <- comp[[cross_pairs$to[r]]]
      if (a != b) { comp[comp == max(a, b)] <- min(a, b); changed <- TRUE }
    }
    if (!changed) break
  }
  split(strain_names, factor(comp, levels = unique(comp)))
} else list(strain_names)
names(fit_blocks) <- vapply(fit_blocks, paste, character(1), collapse = "+")
fit_blocks <- fit_blocks[vapply(fit_blocks, function(s) all(s %in% fit_strains), logical(1))]

vec <- function(x) array(as.numeric(x), dim = length(x))   # 长度为 1 也按向量传给 Stan

# 一个块的 Stan 数据：病原体自己的起点之前的周记为缺失（-1），不进似然
block_data <- function(fit_df, strains, num_knots, gen_syn) {
  n   <- length(strains)
  cases <- round(as.matrix(fit_df[, strains, drop = FALSE]))
  cases[!is.na(cases) & cases < 0] <- 0
  cases[is.na(cases)] <- -1
  for (s in strains) cases[fit_df$date < strain_start[[s]], s] <- -1
  storage.mode(cases) <- "integer"
  cm   <- cross_map[strains, strains, drop = FALSE]
  used <- sort(unique(cm[cm > 0]))                          # 本块用到的交叉免疫参数（全局编号）
  cm[] <- match(cm, used, nomatch = 0L)
  list(strains = strains, idx = match(strains, strain_names), cross_ids = used,
       dates = fit_df$date, num_knots = num_knots, num_basis = num_knots + spline_degree - 1,
       data = list(
         T_weeks = nrow(fit_df), N_strains = n, cases = cases,
         num_knots = as.integer(num_knots), spline_degree = as.integer(spline_degree),
         sigma_E = vec(sigma_E_fixed[strains]), gamma_I = vec(gamma_I_fixed[strains]),
         pop_k = vec(pop_k_all[strains]),
         logR0_prior_mean = vec(logR0_prior_mean[strains]),
         logR0_prior_sd = vec(logR0_prior_sd[strains]),
         tau_rw_prior_sd = tau_rw_prior_sd, rw_order = as.integer(rw_order),
         smooth_bounds = as.integer(smooth_bounds),
         R0_min = R0_bounds[1], R0_max = R0_bounds[2],
         immunity_days_prior_median = vec(immunity_days_prior_median[strains]),
         immunity_days_prior_sdlog  = vec(immunity_days_prior_sdlog[strains]),
         obs_prior_a = vec(obs_prior_a[strains]), obs_prior_b = vec(obs_prior_b[strains]),
         phi_prior_shape = vec(phi_prior_shape[strains]),
         phi_prior_rate = vec(phi_prior_rate[strains]),
         N_cross = length(used), cross_map = matrix(as.integer(cm), n, n),
         cross_prior_a = vec(cross_prior_a[used]), cross_prior_b = vec(cross_prior_b[used]),
         T_fc_weeks = as.integer(forecast_weeks),
         gen_synthetic = as.integer(gen_syn), kappa = kappa))
}

# 截止到 fit_end 的 Stan 数据（每块一份；每块从块内最早的病原体起点开始）
# common_window = TRUE 时所有块都用全局窗口（只用于测试分块前后是否等价）
make_stan_data <- function(fit_end, gen_syn, common_window = FALSE) {
  if (study == "historical") strain_start <<- hist_strain_start(fit_end)
  fit_df <- raw_data[raw_data$date <= fit_end & raw_data$date >= min(strain_start[fit_strains]), ]
  gaps <- as.numeric(diff(fit_df$date))
  if (any(gaps != 7))
    stop("日期不是连续的周数据，缺口出现在这些日期之后: ",
         paste(fit_df$date[which(gaps != 7)], collapse = ", "))

  blocks <- lapply(fit_blocks, function(s) {
    bdf <- if (common_window) fit_df else fit_df[fit_df$date >= min(strain_start[s]), ]
    block_data(bdf, s, max(4, floor(nrow(bdf) / knot_spacing_weeks) + 1), gen_syn)
  })
  list(fit_df = fit_df, blocks = blocks,
       data = list(T_weeks = nrow(fit_df), population = population, child_ratio = child_ratio))
}

make_init <- function(num_basis, block) {
  strains <- block$strains
  n <- length(strains)
  used <- block$cross_ids
  function() {
    init <- list(
      susc0         = vec(rep(0.5, n)),
      log_inf0      = vec(rep(log(1e-4), n)),
      z_spline      = matrix(rnorm(n * num_basis, 0, 0.1), n, num_basis),
      tau_rw        = vec(rep(0.1, n)),
      immunity_days = vec(immunity_days_prior_median[strains] * exp(rnorm(n, 0, 0.1))),
      obs_rate      = vec(obs_init[strains]),
      phi           = vec(rep(5, n))
    )
    if (length(used) > 0)
      init$cross <- vec(cross_prior_a[used] / (cross_prior_a[used] + cross_prior_b[used]))
    init
  }
}

# 多个随机初值做 L-BFGS 优化，取最好的峰；各条链从这个峰加一点扰动开始
mode_inits <- function(block, num_basis, chains = sampling_args$chains, n_starts = init_n_starts) {
  init_fun <- make_init(num_basis, block)
  best <- NULL
  for (i in seq_len(n_starts)) {
    o <- NULL
    invisible(utils::capture.output(o <- tryCatch(
      optimizing(model, data = block$data, init = init_fun(), iter = 5000, algorithm = "LBFGS",
                 as_vector = FALSE, seed = sampling_args$seed + i),
      error = function(e) NULL)))
    if (!is.null(o) && is.finite(o$value) && (is.null(best) || o$value > best$value)) best <- o
  }
  if (is.null(best)) {
    warning("optimization failed for ", paste(block$strains, collapse = "+"), "; random inits")
    return(lapply(seq_len(chains), function(i) init_fun()))
  }
  p <- best$par
  jit_unit <- function(x) vec(plogis(qlogis(pmin(pmax(x, 1e-6), 1 - 1e-6)) + rnorm(length(x), 0, 0.05)))
  jit_pos  <- function(x) vec(x * exp(rnorm(length(x), 0, 0.05)))
  lapply(seq_len(chains), function(i) {
    init <- list(
      susc0         = jit_unit(p$susc0),
      log_inf0      = vec(pmin(p$log_inf0 + rnorm(length(p$log_inf0), 0, 0.05), -1e-6)),
      z_spline      = matrix(p$z_spline + rnorm(length(p$z_spline), 0, 0.05),
                             nrow = length(block$strains)),
      tau_rw        = jit_pos(p$tau_rw),
      immunity_days = vec(30 + (p$immunity_days - 30) * exp(rnorm(length(p$immunity_days), 0, 0.05))),
      obs_rate      = jit_unit(p$obs_rate),
      phi           = jit_pos(p$phi)
    )
    if (length(block$cross_ids) > 0) init$cross <- jit_unit(p$cross)
    init
  })
}

# 拟合所有块，返回 blockfit
run_fit <- function(sd, pars = NA, parallel = parallel_blocks) {
  one <- function(b) {
    t0 <- Sys.time()
    say <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] block "),
                             paste(b$strains, collapse = "+"), ": ", ..., "\n", sep = "")
    say("optimizing inits")
    inits <- mode_inits(b, b$num_basis)
    say("sampling")
    p <- if (identical(pars, NA)) NA else setdiff(pars, if (b$data$N_cross == 0) "cross")
    fit <- do.call(sampling, c(list(object = model, data = b$data, init = inits, pars = p),
                               sampling_args))
    minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    say(sprintf("done (%.1f min)", minutes))
    list(fit = fit, strains = b$strains, idx = b$idx, cross_ids = b$cross_ids, dates = b$dates,
         num_knots = b$num_knots, minutes = minutes)
  }
  res <- if (parallel && length(sd$blocks) > 1) {
    parallel::mclapply(sd$blocks, one, mc.cores = length(sd$blocks), mc.preschedule = FALSE)
  } else lapply(sd$blocks, one)
  bad <- vapply(res, function(r) !is.list(r) || is.null(r$fit), logical(1))
  if (any(bad)) stop("block fit failed: ", paste(names(res)[bad], collapse = ", "), "\n",
                     paste(vapply(res[bad], function(r) paste(as.character(r), collapse = " "),
                                  character(1)), collapse = "\n"))
  structure(list(blocks = res, dates = sd$fit_df$date), class = "blockfit")
}

# ---- 从 blockfit 取结果（格式同 rstan::extract，病原体维按 strain_names 排）----
bind_strain_dim <- function(pieces, bl, along = NULL) {
  d <- dim(pieces[[1]])
  along <- if (is.null(along)) length(d) else along
  d[along] <- N
  out <- array(NA_real_, dim = d)
  for (i in seq_along(pieces)) {
    if (dim(pieces[[i]])[1] != d[1]) stop("blocks have different numbers of draws")
    ix <- rep(list(TRUE), length(d))
    ix[[along]] <- bl[[i]]$idx
    out <- do.call(`[<-`, c(list(out), ix, list(value = pieces[[i]])))
  }
  out
}

# 按周的量：各块时间轴对齐到全局日期（块开始之前、以及病原体自己起点之前记为 NA）
time_vars <- c("mu_cases", "pred_cases", "log_lik", "inc_week", "S_week", "Rt_week", "R0_week",
               "y_sim", "inc_sim", "S_sim", "Rt_sim")
align_time <- function(a, b, dates) {
  if (dim(a)[2] == 0) return(a)
  out <- array(NA_real_, c(dim(a)[1], length(dates), dim(a)[3]))
  out[, match(b$dates, dates), ] <- a
  for (j in seq_along(b$strains)) out[, dates < strain_start[[b$strains[j]]], j] <- NA
  out
}
pad_dim3 <- function(a, n) {
  if (dim(a)[3] == n) return(a)
  out <- array(NA_real_, c(dim(a)[1:2], n))
  out[, , seq_len(dim(a)[3])] <- a
  out
}

extract_fit <- function(fit, pars) {
  bl  <- fit$blocks
  get <- function(b, p) rstan::extract(b$fit, pars = p)[[1]]
  D   <- length(get(bl[[1]], "lp__"))
  out <- list()
  for (p in pars) {
    out[[p]] <- if (p %in% c("obs_rate", "detection_rate", "hospitalization_rate")) {
      a <- bind_strain_dim(lapply(bl, get, p = "obs_rate"), bl, 2)
      if (p == "detection_rate") { a[, strain_names == hosp_strain] <- NA; a }
      else if (p == "hospitalization_rate") a[, strain_names == hosp_strain]
      else a
    } else if (p == "cross") {
      a <- matrix(NA_real_, D, N_cross)
      for (b in bl) if (length(b$cross_ids)) a[, b$cross_ids] <- get(b, "cross")
      a
    } else if (p == "cross_matrix") {
      a <- array(0, c(D, N, N))
      for (b in bl) a[, b$idx, b$idx] <- get(b, "cross_matrix")
      a
    } else if (p == "state_end") {
      a <- matrix(NA_real_, D, 4 * N)
      for (b in bl) a[, as.vector(sapply(b$idx, function(g) 4 * (g - 1) + 1:4))] <- get(b, "state_end")
      a
    } else if (p %in% time_vars) {
      bind_strain_dim(lapply(bl, function(b) align_time(get(b, p), b, fit$dates)), bl)
    } else if (p %in% c("z_spline", "log_R0_spline_coeff")) {
      pieces <- lapply(bl, get, p = p)
      nb <- max(sapply(pieces, function(a) dim(a)[3]))
      bind_strain_dim(lapply(pieces, pad_dim3, n = nb), bl, 2)
    } else {
      bind_strain_dim(lapply(bl, get, p = p), bl)
    }
  }
  out
}

# 块内参数名 -> 全局名，例如 susc0[2] -> susc0[H3]、cross[1] -> cross[H3-H1]
block_pars <- function(pars, b) {
  pars <- unique(sub("^(detection_rate|hospitalization_rate)$", "obs_rate", pars))
  if (length(b$cross_ids) == 0) pars <- setdiff(pars, "cross")
  pars
}
relabel <- function(rn, b) {
  m <- regmatches(rn, regexec("^([A-Za-z_0-9]+)\\[([0-9]+)(.*)\\]$", rn))
  vapply(seq_along(rn), function(i) {
    x <- m[[i]]
    if (!length(x)) return(rn[i])
    j <- as.integer(x[3])
    lab <- if (x[2] == "cross") cross_names[b$cross_ids[j]] else b$strains[j]
    paste0(x[2], "[", lab, x[4], "]")
  }, character(1))
}

summary_fit <- function(fit, pars, probs = c(0.025, 0.5, 0.975)) {
  do.call(rbind, lapply(fit$blocks, function(b) {
    s <- summary(b$fit, pars = block_pars(pars, b), probs = probs)$summary
    rownames(s) <- relabel(rownames(s), b)
    s
  }))
}

core_pars <- c("susc0", "log_inf0", "z_spline", "tau_rw", "immunity_days", "cross", "obs_rate", "phi")

# 每块的 HMC 诊断
check_fit <- function(fit) {
  do.call(rbind, lapply(names(fit$blocks), function(nm) {
    b  <- fit$blocks[[nm]]
    sp <- rstan::get_sampler_params(b$fit, inc_warmup = FALSE)
    s  <- summary(b$fit, pars = block_pars(core_pars, b))$summary
    data.frame(block = nm, minutes = round(b$minutes, 1),
               divergent = sum(sapply(sp, function(x) sum(x[, "divergent__"]))),
               treedepth_hits = sum(sapply(sp, function(x)
                 sum(x[, "treedepth__"] >= sampling_args$control$max_treedepth))),
               mean_leapfrog = round(mean(sapply(sp, function(x) mean(x[, "n_leapfrog__"])))),
               max_rhat = round(max(s[, "Rhat"], na.rm = TRUE), 3),
               min_n_eff = round(min(s[, "n_eff"], na.rm = TRUE)))
  }))
}

# 等同 as.data.frame(stanfit, pars)：每个样本一行，列名如 susc0[1]（全局编号）
as_df_fit <- function(fit, pars) {
  post <- extract_fit(fit, pars)
  cols <- list()
  for (p in pars) {
    a <- post[[p]]
    if (is.null(dim(a))) cols[[p]] <- a
    else for (j in seq_len(ncol(a))) cols[[sprintf("%s[%d]", p, j)]] <- a[, j]
  }
  as.data.frame(cols, check.names = FALSE)
}

# array[3, H, N] 类的预测结果: 取出第 m 种外推方式 -> draws x H x N
pick_mode <- function(arr, m) {
  a <- arr[, m, , , drop = FALSE]
  dim(a) <- dim(a)[-2]
  a
}

# 预测分位数长表（三种外推方式都输出）
fc_quantiles <- function(fit_obj, origin) {
  pf <- extract_fit(fit_obj, "pred_fc")$pred_fc    # draws x 3 x H x N
  H  <- dim(pf)[3]
  bind_rows(lapply(1:3, function(m) bind_rows(lapply(seq_len(H), function(h)
    bind_rows(lapply(which(strain_names %in% fit_strains), function(k)
      data.frame(origin = origin, fc_mode = m, fc_mode_label = fc_mode_labels[m],
                 horizon = h, target_date = origin + 7 * h, strain = strain_names[k],
                 quantile = fc_quantile_levels,
                 value = quantile(pf[, m, h, k], fc_quantile_levels, names = FALSE))))))))
}

# draws x weeks x strains 数组 -> 长表 (draw, date, strain, value)
arr_long <- function(arr, dates, draw_idx = seq_len(dim(arr)[1])) {
  a <- arr[draw_idx, , , drop = FALSE]
  D <- dim(a)[1]; W <- dim(a)[2]; K <- dim(a)[3]
  data.frame(draw   = rep(draw_idx, times = W * K),
             date   = rep(rep(dates, each = D), times = K),
             strain = rep(strain_names, each = D * W),
             value  = as.vector(a))
}

# draws x weeks x strains 数组 -> 分位数长表
qs <- function(arr, dates) {
  n_draw <- dim(arr)[1]
  bind_rows(lapply(seq_len(N), function(i) {
    m <- matrix(arr[, , i], nrow = n_draw)
    q <- apply(m, 2, quantile, probs = c(0.025, 0.25, 0.5, 0.75, 0.975), na.rm = TRUE)
    data.frame(date = dates, strain = strain_names[i],
               lo95 = q[1, ], lo50 = q[2, ], med = q[3, ], hi50 = q[4, ], hi95 = q[5, ])
  }))
}

lab_fun <- function(df) {
  df %>% mutate(strain = factor(strain, levels = strain_names,
                                labels = strain_labels[strain_names]))
}

# ============================================================================
# 2. 编译
# ============================================================================
cat("=== Compiling Stan model ===\n")
model <- stan_model(model_file)
if (!grepl("obs_prior_a", paste(model@model_code, collapse = "\n")))
  stop("model_file 不是分块版模型（找不到 obs_prior_a），请检查路径和文件内容")
cat("Fit blocks:", paste(names(fit_blocks), collapse = " | "), "\n")
