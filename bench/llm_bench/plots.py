"""出图（期刊风格）：相对 WIS 主图、机制骨架增益、分病原体热图、校准与稳健性、背景信息补充实验

排版参照 Nature / Lancet 的图规范：Arial 7 pt，双栏宽 183 mm，只保留左、下坐标轴和浅色网格，
多面板用粗体字母 A、B、C 标注，图内不放大标题（完整图注写在 results/README.md）。
颜色跟着模型走（按 config 顺序固定分配，不随参与的模型数量变化），每个模型另有固定的标记形状，
色盲或黑白打印时仍能区分。色板取 Okabe–Ito 8 色，已用 dataviz 校验脚本验证（相邻两色色盲 ΔE ≥ 9.6、正常视觉 ΔE ≥ 20）；
同一开发者的模型共用一个颜色（线型和标记区分）；浅色（橙、天蓝、粉、橄榄）对白底对比度 < 3:1，靠图例、
标记形状和数值表补足。
机制模型基线和朴素基线统一用灰黑色 + 不同线型。每张图导出 png（400 dpi）和 pdf（字体嵌入、可编辑）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

SERIES = ["#0072B2", "#D55E00", "#009E73", "#882255", "#E69F00", "#56B4E9", "#CC79A7", "#999933"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "d", "s"]   # 12th, 13th: frontier extension
INK, INK2, MUTED, GRID, SURFACE = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e6e6e6", "#ffffff"
WIDTH = 183 / 25.4                      # 双栏宽度（英寸）
SIDE = 0.735                            # 面板区域右边界；右侧放图例

ARM_LABEL = {
    "direct_blind": "LLM direct forecast, blinded",
    "direct_named": "LLM direct forecast, named",
    "r0_semi": "LLM R0 + mechanistic model",
    "r0_named": "LLM R0 + mechanistic model, named",
    "r0_ctx": "LLM R0 + mechanistic model + context",
    "r0_imp": "LLM R0 + mechanistic model + context + importation",
}
ARM_SHORT = {"direct_blind": "direct/blinded", "direct_named": "direct/named",
             "r0_semi": "R0 + model", "r0_named": "R0 + model/named", "r0_ctx": "R0 + model + context",
             "r0_imp": "R0 + model + context + importation"}
ARM_MARKER = {"direct_blind": ("o", False), "direct_named": ("s", False),
              "r0_semi": ("o", True), "r0_named": ("s", True), "r0_ctx": ("D", True),
              "r0_imp": ("^", True)}   # (形状, 是否实心)
REFERENCE = ("Mechanistic, constant R0 (reference)", INK, "-")
BASELINES = {
    "naive": ("Naive persistence", MUTED, (0, (1, 1.3))),
}
MODEL_LABEL = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini", "grok": "Grok",
               "deepseek": "DeepSeek", "qwen": "Qwen (Groq)", "mock": "Mock (offline test)"}
STRAIN_LABEL = {"B": "Influenza B", "H3": "Influenza A(H3N2)", "H1": "Influenza A(H1N1)",
                "COVID": "COVID-19", "RSV": "RSV", "HFMD": "HFMD"}
# 对数轴的刻度方案，从细到粗；取第一套能把数据夹在两个刻度之间、且刻度不超过 7 个的，坐标轴止于刻度
LOG_GRIDS = [(0.8, 0.9, 1, 1.1, 1.25), (0.6, 0.7, 0.8, 1, 1.25, 1.5), (0.5, 0.75, 1, 1.5, 2, 3),
             (0.25, 0.5, 1, 2, 4), (0.1, 0.2, 0.5, 1, 2, 5, 10), (0.01, 0.1, 1, 10, 100)]

_LABELS: dict[str, str] = {}
_EXTENSION: set[str] = set()     # models shown but not in the pooled estimates (frontier extension)


def _label(m: str) -> str:
    return _LABELS.get(m) or MODEL_LABEL.get(m, m.title())


def _rows(models: list[str], pooled: bool) -> tuple[list[str], int]:
    """Rows of a forest plot: the pooled models, the pooled estimate, then the extension models.
    Returns the rows and the number of models in the pooled estimate."""
    main = [m for m in models if m not in _EXTENSION]
    return main + (["pooled"] if pooled else []) + [m for m in models if m in _EXTENSION], len(main)


def _row_rules(ax, rows: list[str]) -> None:
    """Thin rules above and below the pooled row (or between pooled and extension models)."""
    n_main = len([m for m in rows if m != "pooled" and m not in _EXTENSION])
    cuts = {n_main - 0.5} if "pooled" in rows else set()
    if any(m in _EXTENSION for m in rows):
        cuts.add(n_main + ("pooled" in rows) - 0.5)
    for y in cuts:
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.title_fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.linewidth": 0.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 2.4, "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.5, "legend.borderaxespad": 0, "legend.borderpad": 0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt",
    })


VARIANT_LS = [(0, (4, 1.6)), (0, (1, 1.3)), (0, (5, 1.5, 1, 1.5))]   # 同家族第 2、3、4 个成员：虚线、点线、点划线


def _colors(model_order: list[str], families: dict | None = None) -> dict[str, tuple[str, str, object]]:
    """模型 -> (颜色, 标记, 线型)。颜色按家族分配（家族按首次出现的顺序占用色板槽位），
    标记每个模型各不相同；同家族后出现的成员用虚线。色板前 7 色通过校验，第 8 色找不到合格的，
    所以用家族共色 + 线型/标记的组合编码，而不是再加一个颜色。"""
    families = families or {}
    slot, members, out = {}, {}, {}
    for i, m in enumerate(model_order):
        f = families.get(m) or m
        slot.setdefault(f, len(slot))
        members[f] = members.get(f, 0) + 1
        out[m] = (SERIES[slot[f] % len(SERIES)], MARKERS[i % len(MARKERS)],
                  "-" if members[f] == 1 else VARIANT_LS[(members[f] - 2) % len(VARIANT_LS)])
    return out


def _log_axis(ax, values, axis="y"):
    vals = [v for v in values if np.isfinite(v) and v > 0]
    vmin, vmax = min(min(vals, default=1.0), 1.0), max(max(vals, default=1.0), 1.0)
    for grid in LOG_GRIDS:
        lo = max((t for t in grid if t <= vmin * 1.02), default=None)
        hi = min((t for t in grid if t >= vmax / 1.02), default=None)
        if lo is not None and hi is not None:
            ticks = [t for t in grid if lo <= t <= hi]
            if len(ticks) <= 7:
                break
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo / 1.06, hi * 1.06)
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _panel_title(ax, letter: str, title: str, dy: float = 7):
    """粗体面板字母 + 常规字重的面板标题，左对齐在坐标区上方（dy：离坐标区上沿的距离，pt）"""
    kw = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points", va="baseline")
    ax.annotate(letter, xytext=(0, dy), fontsize=8.5, fontweight="bold", color=INK, **kw)
    ax.annotate(title, xytext=(11, dy), fontsize=7.5, color=INK, **kw)


def _horizon_axis(ax, hs):
    ax.set_xticks(hs)
    ax.set_xlim(hs[0] - 0.3, hs[-1] + 0.3)


def _xlabel(fig, axes, text, y):
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    fig.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=7, color=INK)


def _footnote(fig, note, x=0.075):
    if note:
        fig.text(x, 0.0, note[0].upper() + note[1:] + ".", ha="left", va="bottom", fontsize=6,
                 color=MUTED)


def _llm_handles(llm_models, colors):
    return [Line2D([], [], color=colors[m][0], marker=colors[m][1], ls=colors[m][2], ms=4, lw=1.1,
                   mec=SURFACE, mew=0.6, label=_label(m)) for m in llm_models]


def _baseline_handles():
    h = [Line2D([], [], color=REFERENCE[1], ls=REFERENCE[2], lw=0.8, label=REFERENCE[0])]
    return h + [Line2D([], [], color=c, ls=ls, lw=1.0, label=lab) for lab, c, ls in BASELINES.values()]


def _side_legends(fig, groups, x=SIDE + 0.035, y=0.86, gap=0.05, header=None):
    """右侧竖排的分组图例（组名加粗），逐组往下排"""
    renderer = fig.canvas.get_renderer()
    if header:
        t = fig.text(x, y, header, ha="left", va="top", fontsize=7.5, fontweight="bold", color=INK)
        y = t.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0 - gap
    for title, handles in groups:
        if not handles:
            continue
        leg = fig.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(x, y),
                         alignment="left", title_fontproperties={"weight": "bold", "size": 7})
        y = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0 - gap


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _lines(ax, rel, llm_models, colors, arm, horizons):
    ax.axhline(1.0, color=REFERENCE[1], ls=REFERENCE[2], lw=0.7, zorder=1.5)
    ys = []
    for name, (_, color, ls) in BASELINES.items():
        d = rel[rel["model"] == name].set_index("horizon").reindex(horizons)["rel_wis"]
        if d.notna().any():
            ax.plot(horizons, d.values, color=color, ls=ls, lw=1.0, zorder=2)
            ys += list(d.values)
    for m in llm_models:
        d = rel[(rel["model"] == m) & (rel["arm"] == arm)].set_index("horizon").reindex(horizons)["rel_wis"]
        if d.notna().any():
            c, mk, ls = colors[m]
            ax.plot(horizons, d.values, color=c, ls=ls, lw=1.1, marker=mk, ms=4, mec=SURFACE, mew=0.6,
                    zorder=3)
            ys += list(d.values)
    return ys


def _main_titles(panels, title):
    """面板标题；递进的设计写成"C + calendar & weather"这样的形式，一眼看出是在哪个设计上加的"""
    letters = {a: "ABCDEF"[i] for i, a in enumerate(panels)}
    steps = {"r0_ctx": ("r0_semi", "calendar & weather"), "r0_imp": ("r0_ctx", "importation pressure")}
    out = {}
    for a in panels:
        base, what = steps.get(a, (None, None))
        out[a] = f"{letters[base]} + {what}" if base in letters else title(a)
    return letters, out


def _fig_main(pooled, panels, title, llm_models, colors, hs, out, note):
    n = len(panels)
    letters, titles = _main_titles(panels, title)
    if n <= 3:                                    # one row, legend on the right
        fig = plt.figure(figsize=(WIDTH, 2.5))
        gs = fig.add_gridspec(1, n, left=0.075, right=SIDE, bottom=0.2, top=0.86, wspace=0.12)
        cells = [gs[0, i] for i in range(n)]
        legend_at = (SIDE + 0.035, 0.86)
        xlab_y = 0.07
    else:                                         # 2 x 3; the legend fills the free cell(s)
        ncol, nrow = 3, 2
        fig = plt.figure(figsize=(WIDTH, 4.6))
        gs = fig.add_gridspec(nrow, ncol, left=0.075, right=0.985, bottom=0.11, top=0.93,
                              wspace=0.14, hspace=0.42)
        cells = [gs[i // ncol, i % ncol] for i in range(n)]
        free = gs[n // ncol, n % ncol].get_position(fig) if n < nrow * ncol else None
        legend_at = (free.x0 + 0.02, free.y1) if free is not None else (0.8, 0.93)
        xlab_y = 0.035
    axes, ys = [], []
    for i, arm in enumerate(panels):
        ax = fig.add_subplot(cells[i], sharey=axes[0] if axes else None)
        axes.append(ax)
        ys += _lines(ax, pooled, llm_models, colors, arm, hs)
        _panel_title(ax, letters[arm], titles[arm])
        _horizon_axis(ax, hs)
        first_col = (i == 0) if n <= 3 else (i % 3 == 0)
        if first_col:
            ax.set_ylabel("Relative WIS (lower is better)")
        else:
            ax.tick_params(labelleft=False)
    _log_axis(axes[0], ys)
    bottom = axes[-3:] if n > 3 else axes
    _xlabel(fig, bottom if n > 3 else axes, "Forecast horizon (weeks)", xlab_y)
    _side_legends(fig, [("Language model", _llm_handles(llm_models, colors)),
                        ("Baseline", _baseline_handles())], x=legend_at[0], y=legend_at[1], gap=0.035)
    _footnote(fig, note)
    _save(fig, out, "fig_main_relwis")
    return out / "fig_main_relwis.png"


def _fig_scaffold(pooled, arms, llm_models, colors, out, note, pooled_log=None, sig=None):
    """配对斜线图：同一个 LLM 在两种设计下的合并相对 WIS（对数尺度 WIS；没有时用原尺度），线往下 = 变好。
    A：加上机制模型（直接预测、盲法 → LLM 给 R0 + 机制模型）；B：盲法 → 具名提示。"""
    use = pooled_log if pooled_log is not None and len(pooled_log) else pooled
    scale = "log scale" if use is pooled_log else "natural scale"
    g = (use.assign(log=np.log(use["rel_wis"]))
         .groupby(["model", "arm"])["log"].mean().apply(np.exp))
    pairs = [(("direct_blind", "r0_semi"), "Adding the mechanistic model",
              ("Direct,\nblinded", "R0 +\nmechanistic")),
             (("direct_blind", "direct_named"), "Blinded vs named prompt",
              ("Direct,\nblinded", "Direct,\nnamed"))]
    pairs = [p for p in pairs if all(a in arms for a in p[0])]
    if not pairs:
        return None
    fig = plt.figure(figsize=(WIDTH, 2.6))
    gs = fig.add_gridspec(1, len(pairs), left=0.085, right=0.56, bottom=0.2, top=0.86, wspace=0.28)
    axes, ys = [], []
    for k, ((a0, a1), title, ticklabels) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.axhline(1.0, color=REFERENCE[1], ls=REFERENCE[2], lw=0.7, zorder=1)
        for name, (_, color, ls) in BASELINES.items():
            if (name, "baseline") in g.index:
                ax.axhline(g[(name, "baseline")], color=color, ls=ls, lw=0.9, zorder=1)
                ys.append(g[(name, "baseline")])
        for i, m in enumerate(llm_models):
            dx = (i - (len(llm_models) - 1) / 2) * 0.05          # 轻微错位，避免数值相近的点互相遮挡
            pts = [(x + dx, g[(m, a)]) for x, a in ((0, a0), (1, a1)) if (m, a) in g.index]
            if pts:
                c, mk, ls = colors[m]
                ax.plot(*zip(*pts), color=c, ls=ls, lw=1.2, marker=mk, ms=4.5, mec=SURFACE, mew=0.6,
                        zorder=3)
                if k == 0 and sig is not None and m in sig and not sig[m] and len(pts) == 2:
                    # grounded end point open when the 95% interval of the paired ratio includes 1
                    ax.plot(*pts[1], ls="none", marker=mk, ms=4.5, mfc=SURFACE, mec=c, mew=1.0, zorder=4)
                ys += [v for _, v in pts]
        ax.set_xticks([0, 1], ticklabels)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="x", length=0, pad=4)
        _panel_title(ax, "AB"[k], title)
        if k:
            ax.tick_params(labelleft=False)
    _log_axis(axes[0], ys)
    axes[0].set_ylabel(f"Relative WIS, {scale} (lower is better)")
    groups = [("Language model", _llm_handles(llm_models, colors)), ("Baseline", _baseline_handles())]
    if sig is not None:
        groups.append(("Gain in A (95% interval)", [
            Line2D([], [], ls="none", marker="o", ms=4.5, color=INK, mec=SURFACE, mew=0.6, label="below 1"),
            Line2D([], [], ls="none", marker="o", ms=4.5, mfc=SURFACE, mec=INK, mew=1.0, label="includes 1")]))
    _side_legends(fig, groups, x=0.6, y=0.88)
    _footnote(fig, note)
    _save(fig, out, "fig_scaffold_gain")
    return out / "fig_scaffold_gain.png"



# ---------------------------------------------------------------- heatmap helpers
# 发散色阶：蓝 = 比参照好，中性灰 = 持平，红 = 更差（两臂亮度对称）
DIVERGING = LinearSegmentedColormap.from_list(
    "better_worse", ["#1d4e89", "#4a7fc0", "#a9c6e8", "#f2f1ee", "#efb3a4", "#d0584a", "#8a1c24"])
ARM_TITLE = {"direct_blind": "Direct forecast, blinded", "direct_named": "Direct forecast, named",
             "r0_semi": "LLM R0 + mechanistic model", "r0_named": "LLM R0 + model, named",
             "r0_ctx": "R0 + model + calendar & weather", "r0_imp": "R0 + model + context + importation"}
ARM_COL = {"direct_blind": "Direct,\nblinded", "direct_named": "Direct,\nnamed", "r0_semi": "R0 +\nmodel",
           "r0_named": "R0 + model,\nnamed", "r0_ctx": "+ calendar\n& weather", "r0_imp": "+ import\npressure"}
ARM_ROMAN = {"direct_blind": "(i)", "direct_named": "(ii)", "r0_semi": "(iii)", "r0_ctx": "(iv)", "r0_imp": "(v)",
             "r0_imponly": "(vi)", "r0_placebo": "(vii)"}
ARM_KEY = {"direct_blind": "direct forecast, blinded", "direct_named": "direct forecast, named",
           "r0_semi": "LLM R0 + mechanistic model", "r0_ctx": "(iii) + calendar & weather",
           "r0_imp": "(iv) + importation pressure"}
STRAIN_SHORT = {"B": "B", "H3": "H3N2", "H1": "H1N1", "COVID": "COVID", "RSV": "RSV", "HFMD": "HFMD",
                "all": "All"}
BASE_LABEL = {"stan_const": "Mechanistic, constant R0", "naive": "Naive persistence"}


class _RatioNorm:
    """log scale, 1 in the middle, clipped at lo and hi"""
    def __init__(self, lo=0.5, hi=2.0):
        self.a, self.b = np.log(lo), np.log(hi)

    def __call__(self, v):
        return float(np.clip((np.log(v) - self.a) / (self.b - self.a), 0, 1))


class _CoverageNorm:
    """red = coverage below the nominal level (intervals too narrow), blue = above"""
    def __init__(self, nominal, span=0.4):
        self.n, self.s = nominal, span

    def __call__(self, v):
        return float(np.clip(0.5 + (self.n - v) / (2 * self.s), 0, 1))


def _ink_for(rgba):
    r, g, b = rgba[:3]
    return SURFACE if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def _cell(ax, x, y, w, v, norm, fmt, fontsize):
    c = DIVERGING(norm(v))
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), w, 1, facecolor=c, edgecolor=SURFACE, lw=1.4))
    ax.text(x - 0.5 + w / 2, y, fmt(v), ha="center", va="center", fontsize=fontsize, color=_ink_for(c))


def _heatmap(ax, rows, cols, value, norm, fmt, show_rows=True, fontsize=5.6, merged=(), keys=None):
    """rows: list of (key, label) or None for a gap row; value(row_key, col_key) -> float or nan.
    Rows whose key is in `merged` get one cell spanning all columns (value of the first column)."""
    for i, r in enumerate(rows):
        if r is None:
            continue
        key = r[0]
        if key in merged:
            v = value(key, cols[0][0])
            if np.isfinite(v):
                _cell(ax, 0, i, len(cols), v, norm, fmt, fontsize)
            continue
        for j, (ck, _) in enumerate(cols):
            v = value(key, ck)
            if np.isfinite(v):
                _cell(ax, j, i, 1, v, norm, fmt, fontsize)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)), [c[1] for c in cols], fontsize=6.2, color=INK, linespacing=1.0)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ticks = [i for i, r in enumerate(rows) if r is not None]
    ax.set_yticks(ticks, [rows[i][1] if show_rows else "" for i in ticks], fontsize=6.8, color=INK)
    ax.tick_params(axis="y", length=0, pad=11 if (keys and show_rows) else 3)
    if keys and show_rows:                          # the model's colour and marker, as in the line charts
        tr = offset_copy(ax.transData, fig=ax.figure, x=-6, units="points")
        for i, r in enumerate(rows):
            if r is not None and r[0] in keys:
                c, mk, _ = keys[r[0]]
                ax.plot(-0.5, i, ls="", marker=mk, ms=4, color=c, mec=SURFACE, mew=0.5, clip_on=False,
                        transform=tr, zorder=5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)


def _colorbar(fig, rect, norm, ticks, labels, title, ends=("", "")):
    ax = fig.add_axes(rect)
    x = np.linspace(0, 1, 256)
    ax.imshow(DIVERGING(x)[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
    ax.set_xticks([norm(t) for t in ticks], labels, fontsize=6)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2, width=0.5, color=INK2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=6.5, color=INK, pad=3)
    if ends[0]:
        ax.annotate(ends[0], xy=(0, 0.5), xycoords="axes fraction", xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6, color=INK2)
    if ends[1]:
        ax.annotate(ends[1], xy=(1, 0.5), xycoords="axes fraction", xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6, color=INK2)
    return ax


def _gm(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


# ---------------------------------------------------------------- per-pathogen heatmap
def _fig_pathogen_heatmap(rel, arms, strains, llm_models, colors, out, note):
    """每个面板一个设计：行 = 模型（下方是基线），列 = 病原体 + All；数值 = 各时域的几何平均相对 WIS"""
    cols = [(s, STRAIN_SHORT.get(s, s)) for s in strains] + [("all", "All")]
    base = [m for m in BASELINES if m in set(rel["model"])]
    rows = [(m, _label(m)) for m in llm_models] + ([None] + [(b, BASE_LABEL[b]) for b in base] if base else [])
    tab = rel.groupby(["model", "arm", "strain"])["rel_wis"].apply(_gm)

    def val(arm):
        def f(m, s):
            a = "baseline" if m in BASELINES else arm
            return tab.get((m, a, s), np.nan)
        return f

    ncol = 3
    nrow = int(np.ceil((len(arms) + 1) / ncol))
    cell = 0.155
    ph = len(rows) * cell
    H = nrow * (ph + 0.55) + 0.25
    fig = plt.figure(figsize=(WIDTH, H))
    gs = fig.add_gridspec(nrow, ncol, left=0.18, right=0.99, top=1 - 0.42 / H, bottom=0.12 / H,
                          wspace=0.05, hspace=0.55 / ph)
    norm = _RatioNorm(0.5, 2.0)
    for i, arm in enumerate(arms):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        _heatmap(ax, rows, cols, val(arm), norm, lambda v: f"{v:.2f}", show_rows=(i % ncol == 0),
                 keys=colors)
        _panel_title(ax, "ABCDEFG"[i], ARM_TITLE.get(arm, arm), dy=20)   # above the column labels
    # the free grid cell: colour scale and reading guide
    k = len(arms)
    if k < nrow * ncol:
        pos = gs[k // ncol, k % ncol].get_position(fig)
        _colorbar(fig, [pos.x0 + 0.03, pos.y0 + pos.height * 0.62, pos.width * 0.8, 0.018], norm,
                  [0.5, 0.67, 1, 1.5, 2], ["0.5", "0.67", "1", "1.5", "2"],
                  "Relative WIS (log scale)", ("better", "worse"))
        fig.text(pos.x0 + 0.03, pos.y0 + pos.height * 0.45,
                 "Reference = mechanistic model with\nconstant R0 (relative WIS = 1).\n"
                 "Values: geometric mean over\nforecast horizons 1-4 weeks.",
                 ha="left", va="top", fontsize=6.2, color=INK2, linespacing=1.35)
    _footnote(fig, note, 0.02)
    _save(fig, out, "fig_by_pathogen_heatmap")
    return out / "fig_by_pathogen_heatmap.png"


# ---------------------------------------------------------------- calibration and robustness
def _scatter_pairs(ax, both, llm_models, colors, xlab, ylab, letter, title, note_extra=""):
    """每个 (模型, 设计) 一个点：x、y 是两种算法下的相对 WIS；对角线 = 结论不变。
    颜色和形状 = 哪个 LLM（同 A 面板）；实心 = LLM + 机制模型（设计 iii-v），空心 = 直接预测（i-ii）"""
    for r in both.itertuples():
        if r.model in colors:
            c, mk, _ = colors[r.model]
            if str(r.arm).startswith("direct"):
                ax.plot(r.x, r.y, ls="", marker=mk, ms=4, mfc=SURFACE, mec=c, mew=0.9, zorder=3.5)
            else:
                ax.plot(r.x, r.y, ls="", marker=mk, ms=4, color=c, mec=SURFACE, mew=0.5, zorder=3)
        else:
            mk, ms = BASE_MARK.get(r.model, ("o", 3.8))
            ax.plot(r.x, r.y, ls="", marker=mk, ms=ms, color=MUTED, mec=SURFACE, mew=0.7, zorder=4.5)
    vals = list(both["x"]) + list(both["y"])
    _log_axis(ax, vals, axis="x")
    _log_axis(ax, vals, axis="y")
    lim = ax.get_xlim()
    ax.set_ylim(lim)
    ax.plot(lim, lim, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(1, color=GRID, lw=0.6, zorder=0)
    ax.axvline(1, color=GRID, lw=0.6, zorder=0)
    ax.grid(False)
    rho = both[["x", "y"]].rank().corr().iloc[0, 1]
    ax.annotate(f"Spearman ρ = {rho:.2f}\nn = {len(both)} model-design pairs{note_extra}", xy=(0.97, 0.04),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=5.8, color=INK2)
    ax.set_xlabel(xlab, fontsize=6.5)
    ax.set_ylabel(ylab, fontsize=6.5)
    _panel_title(ax, letter, title)


BASE_MARK = {"naive": ("o", 3.9)}                                  # C、D 里灰点的形状


def _fig_calibration(cov, rel, rel_log, arms, llm_models, colors, out, note, rel_period=None):
    """A、B：50% / 90% 预测区间的实际覆盖率（模型 x 设计；基线跨列合并）；C：原尺度与对数尺度的相对 WIS；
    D（有前瞻期结果时）：开发期与前瞻期的相对 WIS"""
    base = [m for m in BASE_LABEL if m in set(cov["model"])]
    rows = [(m, _label(m)) for m in llm_models] + ([None] + [(b, BASE_LABEL[b]) for b in base] if base else [])
    cols = [(a, ARM_ROMAN.get(a, a)) for a in arms]
    cv = cov.set_index(["model", "arm"])

    def val(col):
        def f(m, a):
            key = (m, "baseline") if m in BASE_LABEL else (m, a)
            return cv[col].get(key, np.nan)
        return f

    have_per = (rel_period is not None and len(rel_period)
                and {"development", "prospective"} <= set(rel_period["period"]))
    n = len(rows)
    cell = 0.16
    extra = 0.05 if have_per else 0.0                 # 有 D 面板时图例多一行，底部略加高
    H = n * cell + 1.45 + (0.9 if have_per else 0) + extra
    fig = plt.figure(figsize=(WIDTH, H))
    top, bottom = 1 - 0.42 / H, (0.92 + extra) / H
    axA = fig.add_axes([0.215, bottom, 0.225, top - bottom])
    axB = fig.add_axes([0.455, bottom, 0.225, top - bottom])
    for ax, col, nominal, letter, title in ((axA, "cov50", 0.5, "A", "50% prediction interval"),
                                            (axB, "cov90", 0.9, "B", "90% prediction interval")):
        _heatmap(ax, rows, cols, val(col), _CoverageNorm(nominal), lambda v: f"{v * 100:.0f}",
                 show_rows=(ax is axA), merged=tuple(base), keys=colors)
        _panel_title(ax, letter, title, dy=16)
    key = [f"{ARM_ROMAN.get(a, a)} {ARM_KEY.get(a, a)}" for a in arms]
    fig.text(0.215, bottom - 0.1 / H, "Designs: " + "; ".join(key[:3]) + (";\n" + "; ".join(key[3:]) if len(key) > 3 else ""),
             ha="left", va="top", fontsize=5.9, color=INK2, linespacing=1.4)
    _colorbar(fig, [0.25, (0.3 + extra) / H, 0.38, 0.1 / H], _CoverageNorm(0.0, 0.4), [-0.4, -0.2, 0, 0.2, 0.4],
              ["+40", "+20", "0", "-20", "-40"],
              "Empirical minus nominal coverage (percentage points); cell values: coverage (%)",
              ("too wide", "too narrow"))
    # C: robustness of the ranking to the scoring scale; D: development vs prospective period
    keep_m = llm_models + list(BASELINES)
    p_nat = rel[rel["strain"] == "all"].groupby(["model", "arm"])["rel_wis"].apply(_gm)
    p_log = rel_log[rel_log["strain"] == "all"].groupby(["model", "arm"])["rel_wis"].apply(_gm)
    both = pd.concat([p_nat.rename("x"), p_log.rename("y")], axis=1).dropna().reset_index()
    both = both[both["model"].isin(keep_m)]
    per = None
    if have_per:
        rp = rel_period[(rel_period["strain"] == "all") & (rel_period["scale"] == "natural")]
        g = rp.groupby(["period", "model", "arm"])["rel_wis"].apply(_gm).unstack("period")
        per = g.dropna().reset_index().rename(columns={"development": "x", "prospective": "y"})
        per = per[per["model"].isin(keep_m) & (per["arm"].isin(arms) | (per["arm"] == "baseline"))]
        n_per = rel_period.groupby("period")["n_origins"].max()
    h = top - bottom
    if per is not None and len(per) >= 3:
        axC = fig.add_axes([0.765, bottom + 0.6 * h, 0.215, 0.38 * h])
        axD = fig.add_axes([0.765, bottom + 0.02 * h, 0.215, 0.38 * h])
        _scatter_pairs(axD, per, llm_models, colors, f"Development period ({int(n_per['development'])} origins)",
                       f"Prospective ({int(n_per['prospective'])} origins)", "D", "Stability over time")
    else:
        axC = fig.add_axes([0.765, bottom + 0.1 * h, 0.215, 0.78 * h])
    _scatter_pairs(axC, both, llm_models, colors, "Relative WIS, natural scale", "Relative WIS, log scale",
                   "C", "Robustness to the scoring scale")
    # C、D 的图例：颜色和形状 = 哪个 LLM；实心 = 放进机制模型；空心 = 直接预测；灰色 = 不用 LLM 的基线
    grey = lambda m: Line2D([], [], ls="", marker=BASE_MARK[m][0], ms=BASE_MARK[m][1], color=MUTED, mec=SURFACE, mew=0.7)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=INK2, mec=SURFACE, mew=0.5),
                        Line2D([], [], ls="", marker="o", ms=4, mfc=SURFACE, mec=INK2, mew=0.9),
                        grey("naive"),
                        Line2D([], [], color="#b3b3b3", lw=0.8)],
               labels=["filled: LLM + mechanistic model (iii-v)", "hollow: direct LLM forecast (i, ii)",
                       "grey: naive persistence (no LLM)", "line at 1: reference, mechanistic model with constant R0"],
               loc="upper left", bbox_to_anchor=(0.725, (0.6 + extra) / H), fontsize=5.8, frameon=False,
               handletextpad=0.3, labelspacing=0.3, borderaxespad=0,
               title="C, D: colour and shape = which LLM (as in A)", title_fontsize=5.8, alignment="left")
    _footnote(fig, note, 0.02)
    _save(fig, out, "fig_calibration")
    return out / "fig_calibration.png"


# ---------------------------------------------------------------- external information composite
def _short_holiday(ev: str) -> str:
    ev = ev.lower()
    for k, v in (("summer", "Summer"), ("christmas", "Christmas"), ("lunar", "Lunar NY"), ("easter", "Easter")):
        if k in ev:
            return v
    return ""


EFFECT_PANELS = [   # (effect, stratum, title, subtitle, x label)
    ("calendar_weather", "all", "Calendar & weather", "all pathogens", "with / without"),
    ("importation", "all", "Importation pressure", "influenza", "with / without"),
    ("real_vs_placebo", "all", "Real vs placebo", "influenza", "real / placebo"),
    ("importation", "strata", "Importation, by signal", "influenza", "with / without"),
]


def _fig_external(eff, context, llm_models, colors, strains, out, note):
    """外部信息的析因实验。A：给大模型的背景信息随时间的变化（输入压力条上的圆点 = 该起点的输入信号分层）；
    B-E：各效应的 WIS 比值（每个模型 + 跨模型合并，95% bootstrap 区间），四个面板共用一条对数横轴"""
    cal, wea, imp, origins = (context.get(k) for k in ("calendar", "weather", "importation", "origins"))
    signal = None
    if imp is not None:
        from .prompts import IMPORT_DELAY_DAYS
        from .scoring import import_signal
        signal = import_signal(imp, origins, IMPORT_DELAY_DAYS)
    lo, hi = min(origins) - pd.Timedelta(days=13), max(origins) + pd.Timedelta(weeks=4)
    strips = [("holiday", 0.45)]
    if wea is not None:
        strips += [("temp", 0.6), ("ah", 0.6)]
    if imp is not None:
        strips += [("imp_H1", 0.42), ("imp_H3", 0.42), ("imp_B", 0.42)]
    ex_models = context.get("experiment_models") or llm_models
    have = set(eff["model"]) if eff is not None and len(eff) else set()
    models = [m for m in llm_models if m in ex_models and m in have]
    rows, n_pooled = _rows(models, "pooled" in have)
    gap = 0.1                                         # between strips (inches)
    h_top = sum(h for _, h in strips) + gap * len(strips)
    h_bot = max(len(ex_models) + 1, 3) * 0.26
    m_top, m_mid, m_bot = 0.4, 0.74, 0.62            # title / axis labels + panel titles / x labels + legend
    H = m_top + h_top + m_mid + h_bot + m_bot
    fig = plt.figure(figsize=(WIDTH, H))
    # ---- A: timeline strips
    y = 1 - m_top / H
    ax0 = None
    axes_a = []
    for key, h in strips:
        hh = h / H
        ax = fig.add_axes([0.165, y - hh, 0.815, hh], sharex=ax0)
        ax0 = ax0 or ax
        axes_a.append((key, ax))
        y -= hh + gap / H
    for key, ax in axes_a:
        ax.set_xlim(lo, hi)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=5.8)
        for o in origins:
            ax.axvline(o, color="#ececec", lw=0.6, zorder=0)
        if key == "holiday":
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            if cal is not None:
                for r in cal.itertuples():
                    if r.end < lo or r.start > hi:
                        continue
                    if r.type == "school_holiday":
                        ax.add_patch(Rectangle((r.start, 0.12), r.end - r.start + pd.Timedelta(days=1), 0.76,
                                               facecolor="#dcdcdc", edgecolor="none", zorder=1))
                        mid = max(r.start, lo) + (min(r.end, hi) - max(r.start, lo)) / 2
                        ax.text(mid, 0.5, _short_holiday(r.event), ha="center", va="center", fontsize=5.6,
                                color=INK)
                    elif r.type == "general_holiday":
                        ax.plot([r.start, r.start], [0.0, 0.12], color=INK2, lw=0.8, zorder=2)
            ax.plot(origins, [1.0] * len(origins), ls="", marker="v", ms=3, color=INK, clip_on=False)
            ax.set_ylabel("School holidays\n(ticks: public holidays)", rotation=0, ha="right", va="center",
                          fontsize=6, labelpad=4)
            ax.annotate("forecast origins", xy=(1, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, 5),
                        textcoords="offset points", ha="right", va="bottom", fontsize=5.8, color=INK2)
            ax.spines["bottom"].set_visible(False)
        elif key in ("temp", "ah"):
            col, unit, lab = (("temp", "C", "Temperature\nvs normal (°C)") if key == "temp"
                              else ("ah", "g/m3", "Absolute humidity\nvs normal (g/m³)"))
            w = wea[(wea["week_end"] >= lo) & (wea["week_end"] <= hi)]
            d = w[col] - w[f"{col}_norm"]
            ax.bar(w["week_end"] - pd.Timedelta(days=3), d, width=5.5,
                   color=np.where(d >= 0, "#b5b5b5", "#6f6f6f"), zorder=2)
            ax.axhline(0, color=INK2, lw=0.5, zorder=3)
            m = float(np.nanmax(np.abs(d))) if len(d) else 1
            ax.set_ylim(-m * 1.1, m * 1.1)
            ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6, labelpad=4)
            ax.spines["bottom"].set_visible(False)
        else:
            k = key.split("_")[1]
            name = {"H1": "A(H1N1)", "H3": "A(H3N2)", "B": "B"}[k]
            d = imp[(imp["subtype"] == k) & (imp["week_end"] >= lo) & (imp["week_end"] <= hi)]
            ax.fill_between(d["week_end"], 0, d["pressure_rel"], color="#d9d9d9", lw=0, zorder=1, step="mid")
            ax.plot(d["week_end"], d["pressure_rel"], color=INK2, lw=0.8, drawstyle="steps-mid", zorder=2)
            ax.axhline(1, color=INK, lw=0.5, ls=(0, (3, 2)), zorder=3)
            top = max(2.0, float(d["pressure_rel"].max()) * 1.1) if len(d) else 2.0
            ax.set_ylim(0, top)
            ax.set_yticks([0, 1] if top < 7 else list(range(0, int(top) + 1, 5)))   # dashed line = 1
            if signal is not None:                   # the importation signal shown at each origin
                sg = signal[signal["strain"] == k]
                for st, face in (("high_or_rising", INK), ("low_and_flat", SURFACE)):
                    x = sg[sg["stratum"] == st]["origin"]
                    ax.plot(x, [0] * len(x), ls="", marker="o", ms=3.2, mfc=face, mec=INK, mew=0.6,
                            clip_on=False, zorder=5)
            ax.set_ylabel(f"Import pressure\ninfluenza {name}", rotation=0, ha="right", va="center",
                          fontsize=6, labelpad=4)
            if key != axes_a[-1][0]:
                ax.spines["bottom"].set_visible(False)
    for key, ax in axes_a[:-1]:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    last = axes_a[-1][1]
    last.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    last.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: mdates.num2date(v).strftime("%b\n%Y" if mdates.num2date(v).month == 1 else "%b")))
    _panel_title(axes_a[0][1], "A", "Information added to the LLM prompt", dy=14)
    if signal is not None and imp is not None:
        first_imp = [ax for key, ax in axes_a if key.startswith("imp")][0]
        first_imp.legend(handles=[
            Line2D([], [], ls="", marker="o", ms=3.2, mfc=INK, mec=INK, mew=0.6,
                   label="signal at origin: high or rising"),
            Line2D([], [], ls="", marker="o", ms=3.2, mfc=SURFACE, mec=INK, mew=0.6, label="low and flat")],
            loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, fontsize=5.8, handletextpad=0.2,
            columnspacing=1.0, handlelength=1.0)
    # ---- B-E: effects, one row per model + pooled
    if not models:
        fig.text(0.165, (m_bot + h_bot / 2) / H, "Results pending: no model has yet answered every design of "
                 "the experiment (3 calls each) at 5 or more forecast origins.", fontsize=6.5, color=INK2)
        _footnote(fig, note, 0.02)
        _save(fig, out, "fig_external_information")
        return out / "fig_external_information.png"
    allh = eff[eff["horizon"].astype(str) == "all"]
    pick = {}
    for key, stratum, *_ in EFFECT_PANELS:
        d = allh[(allh["effect"] == key) & allh["pathogens"].isin(["all", "influenza"])]
        pick[(key, stratum)] = (d[d["stratum"] == "all"] if stratum == "all"
                                else d[d["stratum"].isin(["high_or_rising", "low_and_flat"])])
    rng = [v for d in pick.values() for v in list(d["lo95"]) + list(d["hi95"])]
    yb, hb = m_bot / H, h_bot / H
    x0, w, step = 0.165, 0.168, 0.2
    fax = []
    for j, (key, stratum, title, sub, xlab) in enumerate(EFFECT_PANELS):
        ax = fig.add_axes([x0 + j * step, yb, w, hb], sharex=fax[0] if fax else None)
        fax.append(ax)
        d = pick[(key, stratum)]
        for i, m in enumerate(rows):
            c, mk = (INK, "D") if m == "pooled" else colors[m][:2]
            ms = 4.2 if m == "pooled" else 4.5
            sub_rows = ([("all", 0.0, c)] if stratum == "all"
                        else [("high_or_rising", -0.17, c), ("low_and_flat", 0.17, SURFACE)])
            for st, dy, face in sub_rows:
                r = d[(d["model"] == m) & (d["stratum"] == st)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([r.lo95, r.hi95], [i + dy, i + dy], color=c, lw=1.3 if m != "pooled" else 1.5,
                        solid_capstyle="butt", zorder=2)
                ax.plot(r.ratio, i + dy, ls="", marker=mk, ms=ms, mfc=face, mec=c if face == SURFACE else SURFACE,
                        mew=0.8 if face == SURFACE else 0.6, zorder=3)
        ax.axvline(1.0, color=INK, lw=0.7, zorder=1)
        _row_rules(ax, rows)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"\u2190 better   {xlab}   worse \u2192", fontsize=6, color=INK2, labelpad=2)
        _panel_title(ax, "BCDE"[j], title, dy=13)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(11, 4), textcoords="offset points",
                    va="baseline", fontsize=6, color=MUTED)
        if j:
            ax.tick_params(labelleft=False)
    _log_axis(fax[0], rng, axis="x")
    labels = [f"Pooled ({n_pooled} models)" if m == "pooled" else _label(m) for m in rows]
    fax[0].set_yticks(range(len(rows)), labels, fontsize=6.8, color=INK)
    fax[0].get_yticklabels()[rows.index("pooled")].set_fontweight("bold") if "pooled" in rows else None
    # origins scored (per model, all designs complete) at the right of the last panel
    n = allh[(allh["effect"] == "importation") & (allh["stratum"] == "all") &
             (allh["pathogens"] == "influenza")].set_index("model")["n_origins"]
    for i, m in enumerate(rows):
        if m in n.index and m != "pooled":
            fax[-1].annotate(f"{int(n[m])}", xy=(1, i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                             textcoords="offset points", va="center", fontsize=5.8, color=MUTED)
    fax[-1].annotate("origins", xy=(1, 1), xycoords="axes fraction", xytext=(4, 2), textcoords="offset points",
                     va="bottom", fontsize=5.8, color=MUTED)
    fax[-1].legend(handles=[
        Line2D([], [], ls="", marker="o", ms=4, mfc=INK2, mec=INK2, mew=0.8, label="high or rising"),
        Line2D([], [], ls="", marker="o", ms=4, mfc=SURFACE, mec=INK2, mew=0.8, label="low and flat")],
        loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=5.8, handletextpad=0.2,
        handlelength=1.0, columnspacing=0.9)
    _footnote(fig, note, 0.02)
    _save(fig, out, "fig_external_information")
    return out / "fig_external_information.png"


PHASES = [("growing", "Growing", "Rt \u2265 1.05 at the origin"), ("stable", "Stable", "0.95 < Rt < 1.05"),
          ("declining", "Declining", "Rt \u2264 0.95")]


def _fig_phase(tab, mult, colors, out):
    """LLM + 机制模型 vs 固定 R0 的机制模型，按起点时的流行阶段（protocol 修订 7）。
    A-C：各阶段的配对 WIS 比值（每个模型 + 跨模型合并，95% bootstrap 区间）；
    D：LLM 给出的第 4 周 R0 倍数（中位数）按阶段的分布；E：按病原体（合并）"""
    t = tab[(tab["arm"] == "r0_semi") & (tab["scale"] == "natural")]
    ph = t[t["by"] == "phase"]
    models = [m for m in colors if m in set(ph["model"])]
    rows, n_pooled = _rows(models, "pooled" in set(ph["model"]))
    if not models:
        return None
    H = 3.9
    fig = plt.figure(figsize=(WIDTH, H))
    yb, hb = 0.53, 0.33
    x0, w, step = 0.165, 0.17, 0.195
    rng = list(ph["lo95"]) + list(ph["hi95"])
    fax = []
    for j, (key, title, sub) in enumerate(PHASES):
        ax = fig.add_axes([x0 + j * step, yb, w, hb], sharex=fax[0] if fax else None)
        fax.append(ax)
        d = ph[ph["level"] == key].set_index("model")
        for i, m in enumerate(rows):
            if m not in d.index:
                continue
            r = d.loc[m]
            c, mk = (INK, "D") if m == "pooled" else colors[m][:2]
            ax.plot([r.lo95, r.hi95], [i, i], color=c, lw=1.5 if m == "pooled" else 1.3, solid_capstyle="butt", zorder=2)
            ax.plot(r.ratio, i, ls="", marker=mk, ms=4.3, color=c, mec=SURFACE, mew=0.6, zorder=3)
        ax.axvline(1, color=INK, lw=0.7, zorder=1)
        _row_rules(ax, rows)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("\u2190 LLM better   WIS ratio   worse \u2192", fontsize=6, color=INK2, labelpad=2)
        _panel_title(ax, "ABC"[j], title, dy=13)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(11, 4), textcoords="offset points",
                    va="baseline", fontsize=6, color=MUTED)
        if j:
            ax.tick_params(labelleft=False)
    _log_axis(fax[0], rng, axis="x")
    fax[0].set_yticks(range(len(rows)), [f"Pooled ({n_pooled} models)" if m == "pooled" else _label(m) for m in rows],
                      fontsize=6.8, color=INK)
    if "pooled" in rows:
        fax[0].get_yticklabels()[rows.index("pooled")].set_fontweight("bold")
    # D: the LLM's R0 multipliers by phase
    axD = fig.add_axes([x0 + 3 * step + 0.02, yb, 0.985 - (x0 + 3 * step + 0.02), hb])
    mm = mult[mult["model"].isin([m for m in models if m not in _EXTENSION])]   # the pooled models, as A-C and E
    data = [np.log(mm.loc[mm["phase"] == k, "p50"].clip(lower=0.3)) for k, _, _ in PHASES]
    bp = axD.boxplot(data, positions=range(3), widths=0.5, patch_artist=True, showfliers=False,
                     medianprops=dict(color=INK, lw=1.1), whiskerprops=dict(color=INK2, lw=0.6),
                     capprops=dict(color=INK2, lw=0.6), boxprops=dict(lw=0.6, color=INK2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#dcdcdc")
    axD.axhline(0, color=INK, lw=0.7, zorder=0)
    axD.set_xticks(range(3), [t_ for _, t_, _ in PHASES], fontsize=6.3)
    ticks = [0.7, 0.85, 1, 1.2, 1.4]
    axD.set_yticks(np.log(ticks), [f"{v:g}" for v in ticks])
    axD.set_ylim(np.log(0.6), np.log(1.6))
    axD.grid(axis="x", visible=False)
    axD.set_ylabel("LLM R0 multiplier, week 4", fontsize=6.3)
    _panel_title(axD, "D", "What the LLM predicts", dy=13)
    axD.annotate("median over forecasts", xy=(0, 1), xycoords="axes fraction", xytext=(11, 4),
                 textcoords="offset points", va="baseline", fontsize=6, color=MUTED)
    # E: by pathogen (pooled) and by horizon (pooled)
    pe = t[(t["by"] == "strain") & (t["model"] == "pooled")].set_index("level")
    ho = t[(t["by"] == "horizon") & (t["model"] == "pooled")].set_index("level")
    axE = fig.add_axes([x0, 0.1, 0.37, 0.24])
    order = [s_ for s_ in ["B", "H3", "H1", "COVID", "RSV", "HFMD"] if s_ in pe.index]
    for i, s_ in enumerate(order):
        r = pe.loc[s_]
        axE.plot([i, i], [r.lo95, r.hi95], color=INK2, lw=1.2, solid_capstyle="butt")
        axE.plot(i, r.ratio, ls="", marker="D", ms=4, color=INK, mec=SURFACE, mew=0.6)
    axE.axhline(1, color=INK, lw=0.7)
    axE.set_xticks(range(len(order)), [STRAIN_SHORT.get(s_, s_) for s_ in order], fontsize=6.3)
    _log_axis(axE, list(pe["lo95"]) + list(pe["hi95"]), axis="y")
    axE.set_ylabel("WIS ratio (pooled)", fontsize=6.3)
    _panel_title(axE, "E", "By pathogen", dy=6)
    axF = fig.add_axes([x0 + 0.47, 0.1, 0.33, 0.24])
    hs = [h for h in ["1", "2", "3", "4"] if h in ho.index]
    axF.fill_between(range(len(hs)), ho.loc[hs, "lo95"], ho.loc[hs, "hi95"], color="#e3e3e3", lw=0)
    axF.plot(range(len(hs)), ho.loc[hs, "ratio"], color=INK, lw=1.1, marker="D", ms=3.5, mec=SURFACE, mew=0.5)
    axF.axhline(1, color=INK, lw=0.7, ls=(0, (3, 2)))
    axF.set_xticks(range(len(hs)), [f"{h} wk" for h in hs], fontsize=6.3)
    _log_axis(axF, list(ho["lo95"]) + list(ho["hi95"]), axis="y")
    axF.set_ylabel("WIS ratio (pooled)", fontsize=6.3)
    _panel_title(axF, "F", "By forecast horizon", dy=6)
    _save(fig, out, "fig_llm_phase")
    return out / "fig_llm_phase.png"


def plot_all(rel: pd.DataFrame, cov: pd.DataFrame, model_order: list[str], strains: list[str],
             horizons: int, out: Path, note: str = "", labels: dict | None = None,
             min_origins: int = 5, families: dict | None = None,
             gain: pd.DataFrame | None = None, rel_log: pd.DataFrame | None = None,
             context: dict | None = None) -> list[Path]:
    _style()
    _LABELS.clear()
    _LABELS.update(labels or {})
    _EXTENSION.clear()
    _EXTENSION.update((context or {}).get("extension_models") or ())
    hs = list(range(1, horizons + 1))
    colors = _colors(model_order, families)
    # 已评分起点太少的 (模型, 组) 先不画，免得一两个起点的结果误导（n 是每个病原体 x 时域的目标数）
    n_orig = rel[rel["strain"] != "all"].groupby(["model", "arm"])["n"].max()
    thin = n_orig[n_orig < min_origins].index
    if len(thin):
        by_model = {}
        for m, a in thin:
            by_model.setdefault(m, []).append(ARM_SHORT.get(a, a))
        n_arms = len({a for a in rel["arm"].unique() if a in ARM_LABEL})
        note = (note + "; " if note else "") + f"not shown (fewer than {min_origins} scored origins so far): " + \
            "; ".join(f"{_label(m)} ({'all designs' if len(v) >= n_arms else ', '.join(v)})"
                      for m, v in by_model.items())
    keep = lambda df: df[~df.set_index(["model", "arm"]).index.isin(thin)]  # noqa: E731
    rel, cov = keep(rel), keep(cov)
    rel_log = keep(rel_log) if rel_log is not None else None
    arms = [a for a in ARM_LABEL if a in set(rel["arm"])]
    llm_models = [m for m in model_order if m in set(rel["model"])]
    panels = arms or ["baseline"]            # 还没有 LLM 结果时只画基线
    title = lambda a: ARM_TITLE.get(a, "Baselines only (no LLM results yet)")  # noqa: E731
    pooled = rel[rel["strain"] == "all"]

    made = [_fig_main(pooled, panels, title, llm_models, colors, hs, out, note)]
    if llm_models and len(arms) > 1:
        made.append(_fig_scaffold(pooled, arms, llm_models, colors, out, note,
                                  pooled_log=rel_log[rel_log["strain"] == "all"] if rel_log is not None else None,
                                  sig=(context or {}).get("scaffold_sig")))
    if llm_models:
        made.append(_fig_pathogen_heatmap(rel, arms, strains, llm_models, colors, out, note))
    cov = cov[cov["model"].isin(llm_models + list(BASELINES) + ["stan_const"])]
    if llm_models and len(cov) and rel_log is not None:
        made.append(_fig_calibration(cov, rel, rel_log, arms, llm_models, colors, out, note,
                                     rel_period=(context or {}).get("rel_period")))
    if context and context.get("phase") is not None and context.get("mult") is not None:
        made.append(_fig_phase(context["phase"], context["mult"], colors, out))
    if context and context.get("origins"):
        g = gain[gain["n_origins"] >= min_origins] if gain is not None and len(gain) else None
        made.append(_fig_external(g, context, model_order, colors, strains, out, ""))
    return [p for p in made if p is not None]
