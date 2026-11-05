# ============================================================================
# 核心流行病学参数的先验与后验分布（一张图）+ 数值表
#
# 用法: Rscript plot_params.R [起点，默认 val_end]      （先运行 export_params.R）
# 输出: output/figures/origins/parameters_prior_posterior.png/.pdf
#       output/figures/origins/parameters_prior_posterior.csv（先验、后验的中位数和 95% 区间）
# 先验按 ai_epi_core.R 的设置抽样（10 万次），后验来自 export_params.R；两者用相同的核密度估计，
# 各自积分为 1。潜伏期和传染期是文献固定值（不估计），不在图中。
# ============================================================================
suppressMessages(source(file.path(Sys.getenv("AI_EPI_ROOT", "."), "ai_epi_core.R")))
source(file.path(proj_dir, "plot_style.R"))
suppressPackageStartupMessages({ library(patchwork); library(dplyr); library(arrow) })

args   <- commandArgs(trailingOnly = TRUE)
origin <- if (length(args) && !grepl("^--", args[1])) as.Date(args[1]) else val_end
src    <- file.path(out_dir, "params", paste0("params_", format(origin), ".parquet"))
fig    <- file.path(out_dir, "figures", "origins")
if ("--test" %in% args) fig <- Sys.getenv("PARAMS_TEST_DIR", tempdir())
post   <- read_parquet(src)

bound_R0 <- function(lr, lo = R0_bounds[1], hi = R0_bounds[2]) {       # 与模型相同的平滑边界
  a <- log(lo); b <- log(hi)
  if (smooth_bounds == 1) exp(a + (b - a) * plogis((lr - 0.5 * (a + b)) * 4 / (b - a)))
  else pmin(hi, pmax(lo, exp(lr)))
}

# ---- 先验抽样
set.seed(20260926)
n <- 1e5
prior <- list()
for (s in strain_names) {
  imm <- rlnorm(3 * n, log(immunity_days_prior_median[[s]]), immunity_days_prior_sdlog[[s]])
  prior[[s]] <- data.frame(strain = s, rbind(
    data.frame(parameter = "R0_start", value = bound_R0(rnorm(n, logR0_prior_mean[[s]], logR0_prior_sd[[s]]))),
    data.frame(parameter = "susc0", value = rbeta(n, 2, 2)),
    data.frame(parameter = "immunity_days", value = imm[imm >= 30][seq_len(n)]),   # 模型里下限 30 天
    data.frame(parameter = "obs_rate", value = rbeta(n, obs_prior_a[[s]], obs_prior_b[[s]]))))
}
for (j in seq_along(cross_names))
  prior[[cross_names[j]]] <- data.frame(strain = cross_names[j], parameter = "cross",
                                        value = rbeta(n, cross_prior_a[j], cross_prior_b[j]))
prior <- bind_rows(prior) %>% mutate(which = "Prior")
post  <- post %>% filter(parameter %in% c("R0_start", "susc0", "immunity_days", "obs_rate", "cross")) %>%
  select(strain, parameter, value) %>% mutate(which = "Posterior")
both  <- bind_rows(prior, post)
both  <- bind_rows(both, both %>% filter(parameter == "immunity_days") %>%     # 图里用年
                     mutate(parameter = "immunity_years", value = value / 365.25))

strain_lab <- c(B = "Influenza B", H3 = "Influenza A(H3N2)", H1 = "Influenza A(H1N1)",
                COVID = "COVID-19", RSV = "RSV", HFMD = "HFMD",
                "H3-H1" = "A(H3N2) \u2013 A(H1N1)", "A-B" = "Influenza A \u2013 B")
both$strain_f <- factor(strain_lab[both$strain], levels = unname(strain_lab))

# ---- 数值表
tab <- both %>% group_by(parameter, strain, which) %>%
  summarise(median = median(value), lo95 = quantile(value, 0.025), hi95 = quantile(value, 0.975),
            .groups = "drop") %>%
  tidyr::pivot_wider(names_from = which, values_from = c(median, lo95, hi95)) %>%
  select(parameter, strain, median_Prior, lo95_Prior, hi95_Prior, median_Posterior, lo95_Posterior,
         hi95_Posterior)

# ---- 图：每行一个参数，同一行共用横轴
rows <- list(
  list(p = "R0_start", t = "Basic reproduction number R0 at the start of the fitting window",
       log = FALSE, lim = c(0.5, 4), br = c(1, 2, 3, 4)),
  list(p = "susc0", t = "Initial susceptible fraction", log = FALSE, lim = c(0, 1), br = c(0, 0.5, 1)),
  list(p = "immunity_years", t = "Mean duration of immunity (years, log scale)", log = TRUE,
       lim = c(0.08, 16), br = c(0.1, 0.3, 1, 3, 10)),
  list(p = "obs_rate", t = "Ascertainment rate (HFMD: proportion hospitalised)", log = FALSE,
       lim = c(0, 1), br = c(0, 0.5, 1)),
  list(p = "cross", t = "Cross-immunity between influenza viruses", log = FALSE,
       lim = c(0, 1), br = c(0, 0.5, 1)))

cols  <- c(Prior = "#9a9a9a", Posterior = pub$fit)
fills <- c(Prior = "#d9d9d9", Posterior = pub$fit)
row_plot <- function(r, i, first) {
  d <- both %>% filter(parameter == r$p) %>% droplevels()
  med <- d %>% filter(which == "Posterior") %>% group_by(strain_f) %>% summarise(m = median(value), .groups = "drop")
  g <- ggplot(d, aes(value, fill = which, colour = which, linetype = which)) +
    geom_density(aes(alpha = which), linewidth = 0.35, adjust = 1.1, bounds = if (r$log) c(-Inf, Inf) else r$lim,
                 trim = FALSE, key_glyph = draw_key_polygon) +
    geom_vline(data = med, aes(xintercept = m), colour = pub$fit, linewidth = 0.35, inherit.aes = FALSE) +
    facet_wrap(~strain_f, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = fills, breaks = c("Prior", "Posterior"), labels = c("Prior", "Posterior (vertical line: median)")) +
    scale_colour_manual(values = cols, breaks = c("Prior", "Posterior"), labels = c("Prior", "Posterior (vertical line: median)")) +
    scale_linetype_manual(values = c(Prior = "22", Posterior = "solid"), breaks = c("Prior", "Posterior"),
                          labels = c("Prior", "Posterior (vertical line: median)")) +
    scale_alpha_manual(values = c(Prior = 0.4, Posterior = 0.55), breaks = c("Prior", "Posterior"),
                       labels = c("Prior", "Posterior (vertical line: median)")) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.06))) +
    labs(x = NULL, y = NULL, title = paste0(LETTERS[i], "   ", r$t)) +
    theme_pub() +
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          panel.grid.major.y = element_blank(), panel.spacing.x = unit(13, "pt"),
          strip.text = if (first || r$p == "cross") element_text(face = "bold", hjust = 0, size = 6.4,
                                                              colour = pub$ink, margin = margin(t = 1, b = 3))
                       else element_blank(),
          plot.title = element_text(face = "bold", size = 7.2, margin = margin(b = if (first || r$p == "cross") 3 else 4)),
          plot.margin = margin(4, 6, 2, 2))
  g <- if (r$log) g + scale_x_log10(limits = r$lim, breaks = r$br, labels = function(x) format(x, drop0trailing = TRUE),
                                    expand = expansion(mult = 0.01))
       else g + scale_x_continuous(limits = r$lim, breaks = r$br, labels = function(x) format(x, drop0trailing = TRUE),
                                   expand = expansion(mult = 0.01))
  g
}
plots <- lapply(seq_along(rows), function(i) row_plot(rows[[i]], i, i == 1))
# 最后一行只有两个面板（交叉免疫），右边的空白放图例
design <- "AAAAAA
BBBBBB
CCCCCC
DDDDDD
EEFFFF"
p <- wrap_plots(A = plots[[1]], B = plots[[2]], C = plots[[3]], D = plots[[4]], E = plots[[5]],
                F = guide_area(), design = design) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right", legend.direction = "vertical", legend.justification = c(0, 0.5),
        legend.key.width = unit(18, "pt"), legend.key.height = unit(9, "pt"),
        legend.spacing.y = unit(4, "pt"))
save_pub(file.path(fig, "parameters_prior_posterior"), p, height = 6.3)
write.csv(tab, file.path(fig, "parameters_prior_posterior.csv"), row.names = FALSE)
cat("figure and table in", fig, "\n")
print(as.data.frame(tab %>% mutate(across(where(is.numeric), ~ signif(.x, 3)))))
