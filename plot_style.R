# ============================================================================
# 期刊风格的统一绘图设置（plot_origins.R 使用；ai_epi.R 以后也可以 source 这个文件）
#   Arial 7 pt、双栏宽 183 mm；只保留 x 轴线和浅色水平网格；分面标题粗体左对齐，带字母 A–F
#   颜色：拟合 = Okabe–Ito 蓝 #0072B2，预测 = 朱红 #D55E00（色盲安全，已用 dataviz 校验脚本验证）
#   每张图同时存 png（400 dpi，ragg）和 pdf（Quartz，字体嵌入）
# ============================================================================
suppressPackageStartupMessages({ library(ggplot2); library(scales) })
invisible(Sys.setlocale("LC_TIME", "C"))          # 月份用英文缩写

pub <- list(font = "Arial", size = 7, width = 183 / 25.4,
            ink = "#1a1a1a", ink2 = "#4d4d4d", muted = "#8c8c8c", grid = "#e6e6e6",
            shade = "#f2f2f2", pre = "#b3b3b3", fit = "#0072B2", fc = "#D55E00")

theme_pub <- function(base_size = pub$size) {
  theme_minimal(base_size = base_size, base_family = pub$font) +
    theme(text = element_text(colour = pub$ink),
          panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
          panel.grid.major.y = element_line(colour = pub$grid, linewidth = 0.25),
          axis.line.x = element_line(colour = pub$ink2, linewidth = 0.3),
          axis.ticks = element_line(colour = pub$ink2, linewidth = 0.3),
          axis.ticks.length = unit(2, "pt"),
          axis.text = element_text(colour = pub$ink2, size = rel(0.93)),
          axis.title = element_text(colour = pub$ink),
          strip.text = element_text(face = "bold", hjust = 0, colour = pub$ink, size = rel(1.07),
                                    margin = margin(t = 2, b = 4)),
          legend.position = "top", legend.justification = "left", legend.title = element_blank(),
          legend.text = element_text(colour = pub$ink, size = rel(0.97), margin = margin(r = 10, l = 3)),
          legend.key.height = unit(8, "pt"), legend.key.width = unit(16, "pt"),
          legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(0, 0, 0, 0),
          legend.box.spacing = unit(4, "pt"), legend.spacing.y = unit(2, "pt"),
          legend.box.just = "left",
          plot.title = element_text(face = "bold", size = rel(1.15), hjust = 0, margin = margin(b = 2)),
          plot.subtitle = element_text(colour = pub$ink2, hjust = 0, margin = margin(b = 6)),
          plot.title.position = "plot",
          panel.spacing.x = unit(16, "pt"), panel.spacing.y = unit(10, "pt"),
          plot.margin = margin(5, 8, 4, 4))
}

# 分面标题前加粗体字母：A  Influenza B
letter_labeller <- function(levels) {
  as_labeller(function(x) paste0(LETTERS[match(x, levels)], "   ", x))
}

# x 轴：季度刻度，只在一月标出年份
quarter_breaks <- function(lims) {
  lims <- as.Date(lims)
  b <- seq(as.Date(cut(lims[1], "quarter")), lims[2], by = "3 months")
  b[b >= lims[1]]
}
quarter_labels <- function(x) {
  ifelse(is.na(x), "", ifelse(format(x, "%m") == "01", format(x, "%b\n%Y"), format(x, "%b")))
}

# pdf：macOS 上用 Quartz（嵌入 Arial 子集；本机 R 的 cairo 依赖未安装的 XQuartz，不能用），其他系统退回 pdf()
pdf_pub <- function(filename, width, height, ...) {
  if (capabilities("aqua")) grDevices::quartz(type = "pdf", file = filename, width = width,
                                              height = height, bg = "white")
  else grDevices::pdf(filename, width = width, height = height, family = "Helvetica",
                      useDingbats = FALSE, bg = "white")
}
save_pub <- function(path_noext, plot, height, width = pub$width, dpi = 400) {
  ggsave(paste0(path_noext, ".png"), plot, width = width, height = height, dpi = dpi,
         device = ragg::agg_png, bg = "white")
  ggsave(paste0(path_noext, ".pdf"), plot, width = width, height = height, device = pdf_pub)
}
