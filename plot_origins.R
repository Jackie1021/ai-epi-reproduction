# ============================================================================
# 滚动起点的拟合 / 预测图（只画已完成的起点，可以在拟合进行中随时运行）
#   rolling_forecasts.png/.pdf       所有起点的 4 周预测拼在一起，叠在完整观测序列上
#   rolling_forecasts_log.png/.pdf   同上，对数坐标，预测区间不截断
#   fits_by_origin.pdf               每个起点一页：拟合期期望病例 + 之后 4 周预测 + 真实观测 + 诊断
#   fit_origin_val_end.png/.pdf      截止 val_end 的那一次（= 主分析 ai_epi.R 的时间窗），单独一张
#   diagnostics.csv                  每个起点、每个块的 Rhat、发散数、耗时
# 图的样式见 plot_style.R（期刊风格）。
# 用法: Rscript plot_origins.R [--smoke]      输出到 output/figures/origins/
#       AI_EPI_STUDY=historical Rscript plot_origins.R    历史期（修正案 8），输出到 output/figures/origins_hist/
# ============================================================================
source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R"))
source(file.path(proj_dir, "plot_style.R"))
smoke <- "--smoke" %in% commandArgs(trailingOnly = TRUE)
smoke_dir <- if (study == "historical") "origins_hist_smoke" else "origins_smoke"
src_dir <- if (smoke) file.path(out_dir, smoke_dir) else origins_dir
fig_dir <- file.path(out_dir, "figures", if (smoke) smoke_dir else basename(origins_dir))
hist_mode <- study == "historical"
strain_names_all <- strain_names
strain_names <- fit_strains                 # 历史期没有 COVID：图里只画拟合了的病原体
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
fc_mode_show <- 1          # 预测用哪种 R0 外推：1 = 固定 R0（也是 LLM 实验的基准）
clip_mult <- 1.5           # 线性坐标图里预测区间上限截到观测最大值的 1.5 倍（只影响显示，数据不变）

dirs <- sort(list.dirs(src_dir, recursive = FALSE))
dirs <- dirs[file.exists(file.path(dirs, "DONE"))]
if (!length(dirs)) stop("还没有完成的起点: ", src_dir)

study_end <- max(as.Date(basename(dirs))) + 7 * forecast_weeks   # 最后一个预测目标：之后的观测不在研究期内，不画
if (hist_mode) raw_data <- raw_data[raw_data$date <= study_end, ]
obs_long <- raw_data %>%
  pivot_longer(all_of(strain_names), names_to = "strain", values_to = "cases") %>%
  filter(!is.na(cases), date <= study_end)

read_one <- function(d) {
  meta   <- fromJSON(file.path(d, "meta.json"))
  origin <- as.Date(meta$origin)
  lat <- read_parquet(file.path(d, "latent_weekly.parquet")) %>%
    filter(variable == "mu_cases", strain %in% strain_names) %>% mutate(date = as.Date(date))
  fc <- read_parquet(file.path(d, "stan_fc_quantiles.parquet")) %>%
    filter(fc_mode == fc_mode_show) %>%
    mutate(target_date = as.Date(target_date), q = round(quantile, 3), origin = origin) %>%
    filter(q %in% c(0.025, 0.25, 0.5, 0.75, 0.975), strain %in% strain_names) %>%
    select(origin, horizon, target_date, strain, q, value) %>%
    pivot_wider(names_from = q, values_from = value, names_prefix = "q")
  list(meta = meta, origin = origin, lat = lat, fc = fc)
}
res <- lapply(dirs, read_one)
cat(length(res), "origins done:", format(sapply(res, function(r) format(r$origin))), "\n")

clip_q <- function(df, obs) {
  cap <- obs %>% group_by(strain) %>% summarise(cap = clip_mult * max(cases), .groups = "drop")
  df %>% left_join(cap, by = "strain") %>%
    mutate(across(starts_with("q0"), ~ pmin(.x, cap))) %>% select(-cap)
}
strip_levels <- unname(strain_labels[strain_names])
facet_pub <- facet_wrap(~ strain, ncol = 2, scales = "free_y", labeller = letter_labeller(strip_levels))
lab_fun <- function(df) df %>% mutate(strain = factor(strain, levels = strain_names,
                                                     labels = strain_labels[strain_names]))
start_df <- function(st) lab_fun(data.frame(strain = names(st), start = as.Date(unname(st)))
                                 %>% filter(strain %in% strain_names))
starts <- if (hist_mode) start_df(strain_start)[0, ] else start_df(strain_start)   # 历史期每个起点的窗口不同，总览图不画
fmt_date <- function(d) sub("^0", "", format(d, "%d %B %Y"))

# ---- 1. 总览：所有起点的预测 ----
fc_raw <- lab_fun(bind_rows(lapply(res, `[[`, "fc")))
fc_cut <- lab_fun(clip_q(bind_rows(lapply(res, `[[`, "fc")), obs_long))
fc_lab <- c(fc = "Forecast: median, 50% and 95% prediction intervals")
overview <- function(log_scale) {
  d <- if (log_scale) fc_raw else fc_cut
  obs <- lab_fun(obs_long)
  p <- ggplot() +
    geom_vline(data = starts, aes(xintercept = start, linetype = "start"), colour = pub$muted,
               linewidth = 0.3) +
    geom_ribbon(data = d, aes(target_date, ymin = q0.025, ymax = q0.975, group = origin, fill = "fc"),
                alpha = 0.13) +
    geom_ribbon(data = d, aes(target_date, ymin = q0.25, ymax = q0.75, group = origin, fill = "fc"),
                alpha = 0.3) +
    geom_line(data = d, aes(target_date, q0.5, group = origin, colour = "fc"), linewidth = 0.45) +
    geom_line(data = obs, aes(date, cases), colour = pub$ink2, linewidth = 0.2) +
    geom_point(data = obs, aes(date, cases, shape = "obs"), colour = pub$ink, size = 0.45) +
    facet_pub +
    scale_colour_manual(NULL, values = c(fc = pub$fc), labels = fc_lab) +
    scale_fill_manual(NULL, values = c(fc = pub$fc), labels = fc_lab) +
    scale_shape_manual(NULL, values = c(obs = 16), labels = c(obs = "Observed weekly cases")) +
    scale_linetype_manual(NULL, values = c(start = "22"), labels = c(start = "Start of fitting window")) +
    scale_x_date(breaks = quarter_breaks, labels = quarter_labels, expand = expansion(mult = 0.01)) +
    guides(shape = guide_legend(order = 1, override.aes = list(size = 1.1)),
           colour = guide_legend(order = 2), fill = guide_legend(order = 2),
           linetype = guide_legend(order = 3)) +
    labs(x = NULL, y = if (log_scale) "Weekly cases (log scale)" else "Weekly cases") +
    theme_pub() + theme(legend.box = "horizontal")
  if (log_scale) {
    p + scale_y_continuous(transform = "log1p", breaks = c(0, 10, 100, 1e3, 1e4, 1e5, 1e6),
                           labels = label_comma(), expand = expansion(mult = c(0.02, 0.03)))
  } else {
    p + scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0.02, 0.03))) +
      expand_limits(y = 0)
  }
}
save_pub(file.path(fig_dir, "rolling_forecasts"), overview(FALSE), height = 6.2)
save_pub(file.path(fig_dir, "rolling_forecasts_log"), overview(TRUE), height = 6.2)

# ---- 2. 每个起点一页 ----
line_lab <- c(fit = "Model fit: posterior median, 50% and 95% credible intervals",
              fc  = "4-week forecast: median, 50% and 95% prediction intervals")
pt_lab <- c(fit = "Observed, used for fitting", held = "Observed, held out",
            pre = "Observed, before fitting window")
plot_one <- function(r, header = TRUE) {
  o  <- r$origin
  dg <- r$meta$diagnostics
  st <- as.Date(unlist(r$meta$strain_start)[strain_names])            # 这个起点各病原体的拟合起点
  names(st) <- strain_names
  obs <- obs_long %>% filter(date <= o + 7 * forecast_weeks, !hist_mode | date >= min(st) - 7 * 26) %>%
    mutate(type = ifelse(date > o, "held", ifelse(date < st[strain], "pre", "fit")))
  fc_o <- lab_fun(clip_q(r$fc, filter(obs, date >= st[strain])))
  lat <- lab_fun(r$lat)
  obs <- lab_fun(obs)
  p <- ggplot() +
    annotate("rect", xmin = o + 3.5, xmax = o + 7 * forecast_weeks + 3.5, ymin = -Inf, ymax = Inf,
             fill = pub$shade) +
    geom_vline(data = start_df(st), aes(xintercept = start), colour = pub$muted, linewidth = 0.3,
               linetype = "22") +
    geom_ribbon(data = lat, aes(date, ymin = lo95, ymax = hi95, fill = "fit"), alpha = 0.16) +
    geom_ribbon(data = lat, aes(date, ymin = lo50, ymax = hi50, fill = "fit"), alpha = 0.34) +
    geom_line(data = lat, aes(date, med, colour = "fit"), linewidth = 0.45) +
    geom_ribbon(data = fc_o, aes(target_date, ymin = q0.025, ymax = q0.975, fill = "fc"), alpha = 0.16) +
    geom_ribbon(data = fc_o, aes(target_date, ymin = q0.25, ymax = q0.75, fill = "fc"), alpha = 0.34) +
    geom_line(data = fc_o, aes(target_date, q0.5, colour = "fc"), linewidth = 0.45) +
    geom_point(data = filter(obs, type == "pre"), aes(date, cases, shape = "pre"), colour = pub$pre,
               size = 0.55) +
    geom_point(data = filter(obs, type == "fit"), aes(date, cases, shape = "fit"), colour = pub$ink,
               size = 0.55) +
    geom_point(data = filter(obs, type == "held"), aes(date, cases, shape = "held"), colour = pub$ink,
               size = 0.95, stroke = 0.35) +
    facet_pub +
    scale_colour_manual(NULL, values = c(fit = pub$fit, fc = pub$fc), breaks = names(line_lab),
                        labels = line_lab) +
    scale_fill_manual(NULL, values = c(fit = pub$fit, fc = pub$fc), breaks = names(line_lab),
                      labels = line_lab) +
    scale_shape_manual(NULL, values = c(fit = 16, held = 1, pre = 16), breaks = names(pt_lab),
                       labels = pt_lab) +
    scale_x_date(breaks = quarter_breaks, labels = quarter_labels, expand = expansion(mult = 0.01)) +
    scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0.02, 0.03))) +
    guides(colour = guide_legend(order = 1), fill = guide_legend(order = 1),
           shape = guide_legend(order = 2, override.aes = list(colour = c(pub$ink, pub$ink, pub$pre),
                                                               size = c(1.1, 1.3, 1.1)))) +
    labs(x = NULL, y = "Weekly cases") + expand_limits(y = 0) +
    theme_pub() + theme(legend.box = "vertical")
  if (!header) return(p)
  p + labs(title = sprintf("Forecast origin %s", fmt_date(o)),
           subtitle = sprintf("Each pathogen fitted from its own start (dashed line) to the origin  |  max R-hat %.3f  |  %d divergent transitions  |  %.0f min%s",
                              dg$max_rhat, dg$n_divergent, r$meta$runtime_min,
                              if (isTRUE(r$meta$smoke)) "  |  SMOKE TEST" else ""))
}
pdf_pub(file.path(fig_dir, "fits_by_origin.pdf"), width = pub$width, height = 6.6)
for (r in res) print(plot_one(r))
invisible(dev.off())
# 截止 val_end 的那一次（= 主分析的时间窗）单独存一张，不带页眉
for (r in res) if (r$origin == val_end)
  save_pub(file.path(fig_dir, "fit_origin_val_end"), plot_one(r, header = FALSE), height = 6.2)

# ---- 3. 诊断表 ----
diag <- bind_rows(lapply(res, function(r) {
  b <- as.data.frame(r$meta$diagnostics$blocks)
  sm <- r$meta$sampling
  data.frame(origin = format(r$origin), weeks = r$meta$T_weeks, block = b$block,
             n_draws = sm$chains * (sm$iter - sm$warmup),           # post-warm-up draws of this fit
             minutes = b$minutes, divergent = b$divergent, treedepth_hits = b$treedepth_hits,
             mean_leapfrog = b$mean_leapfrog, max_rhat = b$max_rhat, min_n_eff = b$min_n_eff)
}))
write.csv(diag, file.path(fig_dir, "diagnostics.csv"), row.names = FALSE)

# 诊断图：行 = 模块，列 = 起点；颜色越红越需要关注（R-hat 高、发散多、有效样本少）
origin_lab <- function(d) paste0(format(d, "%d\n%b\n'"), format(d, "%y"))   # 三行短标签：日 / 月 / 年，24 列也不重叠
if (hist_mode) origin_lab <- function(d) ifelse(format(d, "%m") <= "01" | d == min(d), format(d, "%b\n%Y"),
                                                 format(d, "%b"))                 # 65 列：月份，每年第一个起点加年份
blocks_shown <- c("B+H3+H1", "COVID", "RSV", "HFMD")
blocks_shown <- blocks_shown[blocks_shown %in% names(fit_blocks)]
block_labels <- c("B+H3+H1" = "Influenza (B, H3N2, H1N1)", COVID = "COVID-19", RSV = "RSV", HFMD = "HFMD")
reds <- c("#f2f1ee", "#efb3a4", "#d0584a", "#8a1c24")
diag_long <- diag %>%
  mutate(origin = factor(origin, levels = format(sort(unique(as.Date(origin))))),
         block = factor(block, levels = rev(blocks_shown), labels = rev(unname(block_labels[blocks_shown]))),
         div_pct = 100 * divergent / n_draws)    # n_draws：每个拟合自己的保留抽样数
diag_panel <- function(var, lab, fmt, limits, rev_col = FALSE, letter, title, breaks) {
  cols <- if (rev_col) rev(reds) else reds
  d <- diag_long %>% mutate(v = .data[[var]], txt = fmt(v),
                            pos = pmin(pmax((v - limits[1]) / diff(limits), 0), 1),
                            dark = if (rev_col) pos < 0.35 else pos > 0.65)
  ggplot(d, aes(origin, block, fill = v)) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_text(aes(label = if (hist_mode) "" else txt, colour = dark), size = 1.9, family = pub$font,
              show.legend = FALSE) +
    scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = pub$ink)) +
    scale_fill_gradientn(colours = cols, limits = limits, oob = scales::squish, name = lab, breaks = breaks,
                         guide = guide_colourbar(direction = "horizontal", title.position = "top",
                                                 theme = theme(legend.key.width = unit(55, "pt"),
                                                               legend.key.height = unit(5, "pt")))) +
    scale_x_discrete(position = "top", labels = function(x) origin_lab(as.Date(x))) +
    labs(x = NULL, y = NULL, title = sprintf("%s   %s", letter, title)) +
    theme_pub() +
    theme(panel.grid = element_blank(), axis.line.x = element_blank(), axis.ticks = element_blank(),
          axis.text.x = element_text(size = rel(if (hist_mode) 0.62 else 0.8), lineheight = 0.9),
          axis.text.y = element_text(colour = pub$ink),
          plot.title = element_text(face = "bold", size = rel(1.05)),
          legend.position = "right", legend.title = element_text(size = rel(0.9), colour = pub$ink2),
          legend.text = element_text(size = rel(0.85)))
}
p_diag <- patchwork::wrap_plots(
  diag_panel("max_rhat", "max R-hat", function(v) sprintf("%.3f", v), c(1.00, 1.06), FALSE, "A",
             "Maximum R-hat (target < 1.05; stricter < 1.01)", c(1.00, 1.03, 1.06)),
  diag_panel("div_pct", "% of draws", function(v) sprintf("%.1f", v), c(0, 4), FALSE, "B",
             "Divergent transitions, % of 2,000 post-warm-up draws", c(0, 2, 4)),
  diag_panel("min_n_eff", "draws", function(v) sprintf("%.0f", v), c(100, 600), TRUE, "C",
             "Minimum effective sample size", c(100, 350, 600)),
  ncol = 1)
save_pub(file.path(fig_dir, "sampling_diagnostics"), p_diag, height = if (hist_mode) 4.6 else 5.4)
flag <- diag %>% filter(max_rhat > 1.05 | divergent > 0)
cat("\nper-origin totals:\n")
print(diag %>% group_by(origin, weeks) %>%
        summarise(minutes = sum(minutes), max_rhat = max(max_rhat), divergent = sum(divergent),
                  .groups = "drop"), row.names = FALSE)
cat(sprintf("\n%d block fits with Rhat > 1.05 or divergences%s\n", nrow(flag),
            if (nrow(flag)) ":" else ""))
if (nrow(flag)) print(flag, row.names = FALSE)
cat("\nfigures in", fig_dir, "\n")
if (!smoke) invisible(system2("bash", file.path(proj_dir, "collect_results.sh")))   # 同步到 results/
