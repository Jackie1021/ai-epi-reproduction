# ============================================================================
# 多病原 SEIRS（每个病原体独立 S-E-I-R + 交叉免疫）+ B样条 R0(t)
# 拟合 + 预测 + 导出隐变量和模拟轨迹（供神经网络使用）
#
# 时间切分（神经网络实验用）:
#   训练 各病原体拟合起点 ~ 2025-03-08 | 验证 2025-03-15 ~ 2025-06-07 | 测试 2025-06-14 之后
#   每个病原体从自己这一波流行的起点开始拟合（strain_start，见 ai_epi_core.R），之前的数据不进似然
#   Stan 主拟合使用训练 + 验证期数据（截止 val_end），测试期数据不进入拟合
#
# 设置、数据读取、工具函数和模型编译在 ai_epi_core.R 里（与 export_origins.R 共用）
# 拟合按病原体分块进行，fit 是 blockfit 对象：用 extract_fit / summary_fit / check_fit / as_df_fit 取结果
# ============================================================================
source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))

# ============================================================================
# 3. 主拟合（截止 val_end）
# ============================================================================
main   <- make_stan_data(val_end, gen_synthetic)
fit_df <- main$fit_df
T_weeks       <- nrow(fit_df)
fit_dates     <- fit_df$date
last_fit_date <- max(fit_dates)
fc_dates      <- last_fit_date + 7 * seq_len(forecast_weeks)
holdout_df    <- raw_data[raw_data$date %in% fc_dates, ]

cat("Fitting period:", as.character(range(fit_dates)), "(", T_weeks, "weeks )\n")
for (b in main$blocks)
  cat(sprintf("Block %-8s from %s | %d weeks | %d knots\n", paste(b$strains, collapse = "+"),
              min(b$dates), length(b$dates), b$num_knots))
cat("Forecast:", as.character(range(fc_dates)), "| held-out weeks:", nrow(holdout_df), "\n\n")

cat("=== Sampling (main fit) ===\n")
fit <- run_fit(main)
saveRDS(fit, file.path(out_dir, "fit_main.rds"))

# ============================================================================
# 4. 诊断
# ============================================================================
cat("\n=== HMC diagnostics ===\n")
print(check_fit(fit), row.names = FALSE)

key_pars <- c("susc0", "immunity_days", "tau_rw", if (N_cross > 0) "cross",
              "obs_rate", "phi", "R0_end")
print(round(summary_fit(fit, key_pars, probs = c(0.025, 0.5, 0.975)), 3))

core_summary <- summary_fit(fit, core_pars)
cat("\nMax Rhat:", round(max(core_summary[, "Rhat"], na.rm = TRUE), 3), "\n")
cat("Min n_eff:", round(min(core_summary[, "n_eff"], na.rm = TRUE)), "\n")

# ============================================================================
# 5. 提取后验
# ============================================================================
post <- extract_fit(fit, c("mu_cases", "pred_cases", "mu_fc", "pred_fc",
                            "Rt_week", "Rt_fc", "S_week", "S_fc",
                            "inc_week", "R0_week"))
n_draw_total <- dim(post$mu_cases)[1]

mu_df   <- bind_rows(qs(post$mu_cases,   fit_dates),
                     qs(pick_mode(post$mu_fc, fc_mode_plot), fc_dates))
pred_df <- bind_rows(qs(post$pred_cases, fit_dates),
                     qs(pick_mode(post$pred_fc, fc_mode_plot), fc_dates))
summ_df <- mu_df %>%
  left_join(pred_df, by = c("date", "strain"), suffix = c("_mu", "_pred")) %>%
  mutate(phase = ifelse(date > last_fit_date, "Forecast", "Fit"))

reff_df <- bind_rows(qs(post$Rt_week, fit_dates),
                     qs(pick_mode(post$Rt_fc, fc_mode_plot), fc_dates))
imm_df  <- bind_rows(qs(1 - post$S_week, fit_dates),
                     qs(1 - pick_mode(post$S_fc, fc_mode_plot), fc_dates))

obs_df <- bind_rows(
  fit_df     %>% mutate(type = "Observed (fitted)"),
  holdout_df %>% mutate(type = "Observed (held-out)")
) %>%
  pivot_longer(all_of(strain_names), names_to = "strain", values_to = "cases") %>%
  filter(!is.na(cases), cases >= 0) %>%
  mutate(type = ifelse(type == "Observed (fitted)" & date < strain_start[strain],
                       "Observed (before fit start)", type))

summ_df <- lab_fun(summ_df)
obs_df  <- lab_fun(obs_df)
reff_df <- lab_fun(reff_df)
imm_df  <- lab_fun(imm_df)

write.csv(summ_df, file.path(out_dir, "fit_forecast_summary.csv"), row.names = FALSE)

# ============================================================================
# 6. 图：6 个病原体的拟合 + 预测
# ============================================================================
split_x   <- last_fit_date + 3.5
date_step <- if (T_weeks > 80) "3 months" else "1 month"
fc_shade  <- function(xmax) annotate("rect", xmin = split_x, xmax = xmax + 3.5,
                                     ymin = -Inf, ymax = Inf, fill = "grey90")
theme_epi <- theme_bw(base_size = 11) +
  theme(strip.background = element_rect(fill = "grey95"),
        panel.grid.minor = element_blank())

plot_fit_fc <- function(summ, obs, ncol = 2) {
  ggplot() +
    fc_shade(max(summ$date)) +
    geom_vline(xintercept = split_x, linetype = "dashed", colour = "grey40") +
    geom_ribbon(data = summ, aes(x = date, ymin = lo95_pred, ymax = hi95_pred),
                fill = "#4C78A8", alpha = 0.20) +
    geom_ribbon(data = summ, aes(x = date, ymin = lo95_mu, ymax = hi95_mu),
                fill = "#4C78A8", alpha = 0.45) +
    geom_line(data = summ, aes(x = date, y = med_mu), colour = "#1F3B73", linewidth = 0.8) +
    geom_point(data = obs, aes(x = date, y = cases, shape = type), size = 1.4) +
    scale_shape_manual(values = c("Observed (fitted)" = 16, "Observed (held-out)" = 1,
                                  "Observed (before fit start)" = 3),
                       name = NULL) +
    scale_x_date(date_breaks = date_step, date_labels = "%b\n%Y") +
    facet_wrap(~ strain, ncol = ncol, scales = "free_y") +
    labs(x = NULL, y = "Weekly cases",
         title = "Model fit and forecast by pathogen (SEIRS with cross-immunity)",
         subtitle = paste0("Fit: ", min(fit_dates), " to ", last_fit_date,
                           "  |  Forecast: ", forecast_weeks, " weeks (shaded), ",
                           fc_mode_labels[fc_mode_plot]),
         caption = paste("Line: posterior median of expected cases;",
                         "dark band: 95% CrI of expected cases;",
                         "light band: 95% posterior predictive interval")) +
    theme_epi + theme(legend.position = "top")
}

p_all <- plot_fit_fc(summ_df, obs_df)
print(p_all)
ggsave(file.path(out_dir, "fit_forecast_all_strains.png"), p_all, width = 13, height = 10, dpi = 300)

for (s in strain_names) {
  lab <- strain_labels[[s]]
  p_s <- plot_fit_fc(filter(summ_df, strain == lab), filter(obs_df, strain == lab), ncol = 1) +
    labs(title = paste0(lab, ": fit and forecast"))
  ggsave(file.path(out_dir, paste0("fit_forecast_", s, ".png")), p_s,
         width = 10, height = 4.5, dpi = 300)
}

# ============================================================================
# 7. 图：有效再生数 Rt 和人群免疫水平
# ============================================================================
plot_latent <- function(df, ylab, title, col, hline = NULL) {
  ggplot(df, aes(x = date)) +
    fc_shade(max(df$date)) +
    { if (!is.null(hline)) geom_hline(yintercept = hline, linetype = "dotted") } +
    geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = col, alpha = 0.25) +
    geom_ribbon(aes(ymin = lo50, ymax = hi50), fill = col, alpha = 0.40) +
    geom_line(aes(y = med), colour = col, linewidth = 0.8) +
    geom_vline(xintercept = split_x, linetype = "dashed", colour = "grey40") +
    scale_x_date(date_breaks = date_step, date_labels = "%b\n%Y") +
    facet_wrap(~ strain, ncol = 2, scales = "free_y") +
    labs(x = NULL, y = ylab, title = title,
         caption = "Line: posterior median; bands: 50% and 95% CrI; shaded: forecast") +
    theme_epi
}

p_reff <- plot_latent(reff_df, expression(R[t]), "Effective reproduction number by pathogen",
                      "#9E2A2B", hline = 1)
print(p_reff)
ggsave(file.path(out_dir, "Rt_all_strains.png"), p_reff, width = 13, height = 10, dpi = 300)

p_imm <- plot_latent(imm_df, "1 - susceptible fraction",
                     "Population immunity against each pathogen", "#2E7D32")
print(p_imm)
ggsave(file.path(out_dir, "immunity_all_strains.png"), p_imm, width = 13, height = 10, dpi = 300)

# ============================================================================
# 8. 图：关键参数的后验分布（一张图）
# ============================================================================
par_draws <- extract_fit(fit, c("susc0", "log_inf0", "immunity_days", "tau_rw",
                                if (N_cross > 0) "cross",
                                "detection_rate", "hospitalization_rate",
                                "phi", "R0_end", "R0_week"))
n_draw <- length(par_draws$hospitalization_rate)

to_long <- function(mat, param, groups) {
  mat <- matrix(mat, nrow = n_draw)
  data.frame(param = param, strain = rep(groups, each = n_draw), value = as.vector(mat))
}

par_names <- c(
  det   = "Detection rate",
  hosp  = "Hospitalization rate (HFMD)",
  phi   = "Overdispersion phi (NegBin)",
  imm   = "Duration of immunity (days)",
  cross = "Cross-immunity strength",
  tau   = "Spline random-walk SD (tau_rw)",
  r0end = "R0 at end of fitting",
  r0m   = "Time-averaged R0 over fitting period",
  s0    = "Initial susceptible fraction",
  inf0  = "Initial infected (E0 + I0) per 100,000"
)
if (N_cross == 0) par_names <- par_names[names(par_names) != "cross"]

par_long <- bind_rows(
  to_long(par_draws$detection_rate[, 1:5], par_names[["det"]],  strain_names[1:5]),
  to_long(par_draws$hospitalization_rate,  par_names[["hosp"]], "HFMD"),
  to_long(par_draws$phi,                   par_names[["phi"]],  strain_names),
  to_long(par_draws$immunity_days,         par_names[["imm"]],  strain_names),
  if (N_cross > 0) to_long(par_draws$cross, par_names[["cross"]], cross_names),
  to_long(par_draws$tau_rw,                par_names[["tau"]],  strain_names),
  to_long(par_draws$R0_end,                par_names[["r0end"]], strain_names),
  to_long(apply(par_draws$R0_week, c(1, 3), mean), par_names[["r0m"]], strain_names),
  to_long(par_draws$susc0,                 par_names[["s0"]],   strain_names),
  to_long(exp(par_draws$log_inf0) * 1e5,   par_names[["inf0"]], strain_names)
) %>%
  mutate(param  = factor(param,  levels = par_names),
         strain = factor(strain, levels = c(strain_names, cross_names)))
rm(par_draws); invisible(gc())

par_summ <- par_long %>%
  group_by(param, strain) %>%
  summarise(mean = mean(value), sd = sd(value), median = median(value),
            lo95 = quantile(value, 0.025), hi95 = quantile(value, 0.975), .groups = "drop")
write.csv(par_summ, file.path(out_dir, "key_parameters_posterior_summary.csv"), row.names = FALSE)
print(par_summ, n = Inf)

strain_cols <- c(B = "#4C78A8", H3 = "#F58518", H1 = "#54A24B",
                 COVID = "#E45756", RSV = "#72B7B2", HFMD = "#B279A2")
strain_cols <- c(strain_cols, setNames(rep("grey55", N_cross), cross_names))

ref_lines <- data.frame(
  param = factor(c(par_names[["r0end"]], par_names[["r0m"]]), levels = par_names), y = 1)

# 先验中心（红色 x）：后验如果和它几乎重合，说明数据没有提供多少信息
prior_ref <- bind_rows(
  data.frame(param = par_names[["imm"]], strain = strain_names,
             y = unname(immunity_days_prior_median)),
  if (N_cross > 0) data.frame(param = par_names[["cross"]], strain = cross_names,
                              y = cross_prior_a / (cross_prior_a + cross_prior_b))
) %>%
  mutate(param  = factor(param,  levels = par_names),
         strain = factor(strain, levels = c(strain_names, cross_names)))

p_par <- ggplot(par_long, aes(x = strain, y = value, fill = strain)) +
  geom_hline(data = ref_lines, aes(yintercept = y), linetype = "dotted", colour = "grey40") +
  geom_violin(scale = "width", trim = TRUE, alpha = 0.55, colour = NA) +
  geom_pointrange(data = par_summ, aes(x = strain, y = median, ymin = lo95, ymax = hi95),
                  inherit.aes = FALSE, size = 0.25, linewidth = 0.6) +
  geom_point(data = prior_ref, aes(x = strain, y = y), inherit.aes = FALSE,
             shape = 4, colour = "red", size = 2.2, stroke = 1) +
  geom_text(data = par_summ,
            aes(x = strain, y = hi95, label = formatC(median, format = "g", digits = 3)),
            inherit.aes = FALSE, vjust = -0.6, size = 2.8) +
  scale_fill_manual(values = strain_cols, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
  facet_wrap(~ param, ncol = 3, scales = "free") +
  labs(x = NULL, y = "Posterior value",
       title = "Posterior distributions of key epidemiological parameters",
       subtitle = paste0("Fit: ", min(fit_dates), " to ", last_fit_date, "  |  ",
                         main$num_knots, " knots  |  ", n_draw, " posterior draws"),
       caption = paste("Violin: posterior density; point: median; bar: 95% CrI; number: median.",
                       "Red x: prior centre. Dotted line: R0 = 1.",
                       "HFMD uses hospitalization rate instead of detection rate.")) +
  theme_epi + theme(panel.grid.major.x = element_blank())

print(p_par)
ggsave(file.path(out_dir, "key_parameters_posterior.png"), p_par, width = 14, height = 14, dpi = 300)

# ============================================================================
# 9. 模拟轨迹：导出 + 检查图
# ============================================================================
if (gen_synthetic == 1) {
  syn <- extract_fit(fit, c("y_sim", "inc_sim", "S_sim", "Rt_sim"))

  syn_df <- arr_long(syn$y_sim, fit_dates)
  names(syn_df)[names(syn_df) == "value"] <- "y_sim"
  syn_df$inc_sim <- as.vector(syn$inc_sim)
  syn_df$S_sim   <- as.vector(syn$S_sim)
  syn_df$Rt_sim  <- as.vector(syn$Rt_sim)
  write_parquet(syn_df, file.path(exp_dir, "synthetic.parquet"))

  set.seed(1)
  show_draws <- sample(unique(syn_df$draw), 30)
  p_syn <- ggplot() +
    geom_line(data = lab_fun(filter(syn_df, draw %in% show_draws)),
              aes(x = date, y = y_sim, group = draw), colour = "grey50", alpha = 0.35) +
    geom_point(data = filter(obs_df, type == "Observed (fitted)"),
               aes(x = date, y = cases), size = 1.1) +
    scale_x_date(date_breaks = date_step, date_labels = "%b\n%Y") +
    facet_wrap(~ strain, ncol = 2, scales = "free_y") +
    labs(x = NULL, y = "Weekly cases",
         title = paste0("Synthetic trajectories (30 random draws, kappa = ", kappa, ")"),
         caption = "Grey: simulated reported cases; points: observed") +
    theme_epi
  print(p_syn)
  ggsave(file.path(out_dir, "synthetic_trajectories.png"), p_syn, width = 13, height = 10, dpi = 300)
  rm(syn, syn_df); invisible(gc())
}

# ============================================================================
# 10. 导出 parquet（给 Python）
# ============================================================================
cat("\n=== Exporting parquet files to", exp_dir, "===\n")
draw_idx <- unique(round(seq(1, n_draw_total, length.out = min(n_export_draws, n_draw_total))))

# 观测数据（全部日期，含测试期，标注划分）
observed <- raw_data %>%
  pivot_longer(all_of(strain_names), names_to = "strain", values_to = "cases") %>%
  mutate(split = split_of(date))
write_parquet(observed, file.path(exp_dir, "observed.parquet"))

# 隐变量：真实发病人数 inc、易感比例 S、有效再生数 Rt、基本再生数 R0
latent <- bind_rows(
  arr_long(post$inc_week, fit_dates, draw_idx) %>% mutate(variable = "inc"),
  arr_long(post$S_week,   fit_dates, draw_idx) %>% mutate(variable = "S"),
  arr_long(post$Rt_week,  fit_dates, draw_idx) %>% mutate(variable = "Rt"),
  arr_long(post$R0_week,  fit_dates, draw_idx) %>% mutate(variable = "R0")
)
write_parquet(latent, file.path(exp_dir, "latent.parquet"))

# 后验预测病例
postpred <- arr_long(post$pred_cases, fit_dates, draw_idx)
names(postpred)[names(postpred) == "value"] <- "cases_pred"
write_parquet(postpred, file.path(exp_dir, "postpred.parquet"))

# 参数（每个样本一行）
param_vec <- c("susc0", "log_inf0", "immunity_days", "tau_rw", if (N_cross > 0) "cross",
               "detection_rate", "hospitalization_rate", "phi", "R0_end")
params <- as_df_fit(fit, param_vec)
params$draw <- seq_len(nrow(params))
write_parquet(params[draw_idx, ], file.path(exp_dir, "params.parquet"))

# Stan 在主拟合截止日的预测分位数（三种外推方式）
fc_main <- fc_quantiles(fit, last_fit_date)
write_parquet(fc_main, file.path(exp_dir, "stan_forecast_main.parquet"))

# 元信息
meta <- data.frame(
  key   = c("start_date", "train_end", "val_end", "strains", "forecast_weeks",
            "num_knots", "kappa", "cross_names", "n_export_draws"),
  value = c(as.character(start_date), as.character(train_end), as.character(val_end),
            paste(strain_names, collapse = ","), forecast_weeks, main$num_knots, kappa,
            paste(cross_names, collapse = ","), length(draw_idx))
)
write.csv(meta, file.path(exp_dir, "meta.csv"), row.names = FALSE)
rm(post); invisible(gc())

# ============================================================================
# 11. （可选）测试期滚动重拟合：Stan 的基线预测
# ============================================================================
if (run_rolling_refits) {
  test_end <- max(raw_data$date)
  origins  <- seq(val_end, test_end - 7 * forecast_weeks, by = 7 * refit_every_weeks)
  cat("\n=== Rolling refits at", length(origins), "origins ===\n")

  fc_all <- list(fc_main)                       # 第一个起点就是主拟合
  for (i in seq_along(origins)[-1]) {
    origin <- origins[i]
    f_out  <- file.path(exp_dir, paste0("stan_forecast_", origin, ".parquet"))
    if (file.exists(f_out)) {                   # 已完成的起点直接读取（中断后可续跑）
      fc_all[[i]] <- read_parquet(f_out)
      next
    }
    cat("Refit at origin", as.character(origin), "\n")
    sd_o  <- make_stan_data(origin, 0)
    fit_o <- run_fit(sd_o, pars = "pred_fc")
    fc_all[[i]] <- fc_quantiles(fit_o, origin)
    write_parquet(fc_all[[i]], f_out)
    rm(fit_o); invisible(gc())
  }
  write_parquet(bind_rows(fc_all), file.path(exp_dir, "stan_forecast_rolling.parquet"))
}

cat("\nDone. Figures in:", out_dir, "\nParquet files in:", exp_dir, "\n")
