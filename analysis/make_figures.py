#!/usr/bin/env python3
"""
make_figures.py — the eight figures of the manuscript, at 300 dpi.

Inputs, all produced by the scripts in this repository:
    reference_analysis.csv   analyse_reference_results.py
    results.csv              run_experiments.py
    drift_curve*.csv         the observed log, from the earlier study
    logs/n168251/...         a generated log of the same size, for the curve

    python make_figures.py --data . --out figures
"""

import argparse, csv, math, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

INK = "#1a1a1a"
ACCENT = "#b2182b"
COOL = "#2166ac"
GREY = "#999999"


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    print("  %-34s %s" % (name, "ok"))


# ---------------------------------------------------------------- figure 1
def fig1(data, out):
    rs = [r for r in rows(os.path.join(data, "reference_analysis.csv")) if num(r["k"], 0) > 0]
    bands = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.00), (1.00, 4.0)]
    xs, f1s, recs, ns = [], [], [], []
    for lo, hi in bands:
        sel = [r for r in rs if lo <= num(r["ratio_T_over_k"], 0) < hi]
        if len(sel) < 20:
            continue
        xs.append(math.sqrt(lo * min(hi, 2.0)))
        f1s.append(sum(num(r["f1"], 0) for r in sel) / len(sel))
        recs.append(sum(num(r["recall"], 0) for r in sel) / len(sel))
        ns.append(len(sel))

    # our own logs, on the same axis: T is one, so the ratio is one over k
    by = e1(data)
    ours = []
    for alg in SHOW:
        for n in SIZES:
            r = by.get(alg, {}).get(n)
            if r and num(r["n"], 0) > 0:
                ours.append((1.0 / num(r["n"]), num(r["f1"], 0), num(r["recall"], 0)))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    ax.axvspan(0.9, 3, color="#ededed", zorder=0)
    ax.text(1.55, 0.04, "where every published\ncollection sits", ha="center",
            fontsize=7.5, color="#666666", zorder=1)
    ax.axvspan(0.0012, 0.02, color="#fbecec", zorder=0)
    ax.text(0.0048, 0.04, "where an industrial\nlog sits", ha="center",
            fontsize=7.5, color=ACCENT, zorder=1)

    grid = [10 ** (x / 40.0) for x in range(-120, 21)]
    ax.plot(grid, [min(1.0, g) for g in grid], "--", color=GREY, lw=1, zorder=2,
            label="bound on precision, $T/k$")

    ax.scatter([o[0] for o in ours], [o[2] for o in ours], s=14, facecolors="none",
               edgecolors=INK, lw=0.8, zorder=3, label="recall, generated logs")
    ax.scatter([o[0] for o in ours], [o[1] for o in ours], s=16, color=ACCENT,
               marker="x", lw=1.1, zorder=4, label="F1, generated logs")

    ax.plot(xs, recs, "^-", color=INK, ms=4.5, lw=1.2, alpha=0.8, zorder=5,
            label="mean recall, published results")
    ax.plot(xs, f1s, "o-", color=ACCENT, ms=5, lw=1.8, zorder=6,
            label="mean F1, published results")

    ax.set_xscale("log")
    ax.set_xlim(0.0025, 2.6)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("$T/k$ — true change points per detection returned")
    ax.set_ylabel("score")
    ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.6,
              borderpad=0.2, labelspacing=0.3)
    ax.set_title("F1 tracks the ratio, not the detection.  Recall does not.", loc="left")
    fig.tight_layout()
    save(fig, out, "fig1_f1_vs_ratio.png")


# ---------------------------------------------------------------- figure 2
def fig2(data, out):
    rs = [r for r in rows(os.path.join(data, "reference_analysis.csv"))
          if num(r["k"], 0) > 0 and num(r["T"], 0) == 2]
    bands = [(1, 2, "1"), (2, 4, "2–3"), (4, 8, "4–7"), (8, 16, "8–15"), (16, 10 ** 9, "≥16")]
    algs = sorted({r["algorithm"] for r in rs})
    keep = []
    for a in algs:
        cells = [[r for r in rs if r["algorithm"] == a and lo <= num(r["k"], 0) < hi]
                 for lo, hi, _ in bands]
        if sum(1 for c in cells if len(c) >= 10) >= 2:
            keep.append((a, cells))

    ncol = 4
    nrow = int(math.ceil(len(keep) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.0 * nrow), sharey=True)
    axes = axes.ravel()
    for ax, (a, cells) in zip(axes, keep):
        x = range(len(bands))
        f1 = [sum(num(r["f1"], 0) for r in c) / len(c) if len(c) >= 10 else None for c in cells]
        rc = [sum(num(r["recall"], 0) for r in c) / len(c) if len(c) >= 10 else None for c in cells]
        xs = [i for i, v in zip(x, f1) if v is not None]
        ax.plot(xs, [v for v in f1 if v is not None], "o-", color=ACCENT, ms=4, lw=1.4, label="F1")
        ax.plot([i for i, v in zip(x, rc) if v is not None],
                [v for v in rc if v is not None], "^--", color=INK, ms=3.5, lw=1, alpha=0.7,
                label="recall")
        ax.axvline(1, color=GREY, lw=0.8, ls=":")
        label = {"Earth Mover's Distance": "EMD", "Process Graph Metrics": "Process graph",
                 "Martjushev ADWIN J": "Martjushev", "Zheng DBSCAN": "RINV",
                 "Bose J": "J-measure", "Bose WC": "Window count"}.get(a, a)
        ax.set_title(label, loc="left")
        ax.set_xticks(list(x))
        ax.set_xticklabels([b[2] for b in bands])
        ax.set_ylim(-0.05, 1.08)
    for ax in axes[len(keep):]:
        ax.axis("off")
    axes[0].legend(loc="lower left", frameon=False)
    fig.supxlabel("detections returned, $k$   (the true count is 2, marked)", fontsize=9)
    fig.supylabel("score", fontsize=9)
    fig.suptitle("Holding the algorithm and the true count fixed, F1 still peaks where $k=T$",
                 fontsize=9.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    save(fig, out, "fig2_inverted_u.png")


# ---------------------------------------------------------------- figures 3, 4, 8
def e1(data):
    rs = [r for r in rows(os.path.join(data, "results.csv")) if r["experiment"] == "E1"]
    by = defaultdict(dict)
    for r in rs:
        by[r["algorithm"]][int(float(r["cases"]))] = r
    return by


SHORT = {"EMD (Brockhoff 2020)": "EMD", "Bose J (2011/2014)": "J-measure",
         "Bose WC (2011/2014)": "Window count", "LCDD (Lin 2020)": "LCDD",
         "PGM (Seeliger 2017)": "Process graph", "ProDrift (Maaradji 2015/17)": "ProDrift",
         "RINV (Zheng 2017)": "RINV"}
SHOW = ["EMD (Brockhoff 2020)", "Bose J (2011/2014)", "Bose WC (2011/2014)",
        "LCDD (Lin 2020)", "PGM (Seeliger 2017)"]
STYLE = {"EMD (Brockhoff 2020)": (ACCENT, "o"), "Bose J (2011/2014)": (COOL, "s"),
         "Bose WC (2011/2014)": (INK, "D"), "LCDD (Lin 2020)": ("#7b3294", "^"),
         "PGM (Seeliger 2017)": ("#1b7837", "v")}
SIZES = [1000, 6000, 20000, 60000]


def fig3(data, out):
    by = e1(data)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for alg in SHOW:
        c, m = STYLE[alg]
        xs = [n for n in SIZES if n in by[alg]]
        a1.plot(xs, [num(by[alg][n]["f1"], 0) for n in xs], marker=m, color=c, ms=4.5,
                lw=1.4, label=SHORT[alg])
        d = [(n, num(by[alg][n]["best"])) for n in xs if num(by[alg][n]["best"]) is not None]
        d = [(n, v) for n, v in d if v < 1000]
        if d:
            a2.plot([n for n, _ in d], [v for _, v in d], marker=m, color=c, ms=4.5, lw=1.4)
    a1.set_xscale("log"); a1.set_xticks(SIZES)
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, p: "%g" % (v / 1000) + "k"))
    a1.set_ylim(-0.03, 1.06)
    a1.set_xlabel("cases in the log"); a1.set_ylabel("F1")
    a1.set_title("(a) the metric collapses", loc="left")
    a1.legend(frameon=False, loc="upper right")
    a2.set_xscale("log"); a2.set_xticks(SIZES)
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: "%g" % (v / 1000) + "k"))
    a2.axhline(200, color=GREY, ls="--", lw=1)
    a2.annotate("lag window", (1100, 208), fontsize=7, color=GREY)
    a2.set_ylim(0, 260)
    a2.set_xlabel("cases in the log")
    a2.set_ylabel("distance to the true boundary, cases")
    a2.set_title("(b) the localisation does not", loc="left")
    fig.tight_layout()
    save(fig, out, "fig3_f1_vs_length.png")


def fig4(data, out):
    by = e1(data)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for alg in SHOW:
        c, m = STYLE[alg]
        xs = [n for n in SIZES if n in by[alg]]
        ax.plot(xs, [num(by[alg][n]["n"], 0) for n in xs], marker=m, color=c, ms=4.5,
                lw=1.4, label=SHORT[alg])
    ax.plot(SIZES, [1] * 4, ":", color=GREY, lw=1.2)
    ax.annotate("true change points in the log", (1300, 1.15), fontsize=7.5, color=GREY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: "%g" % (v / 1000) + "k"))
    ax.set_xlabel("cases in the log")
    ax.set_ylabel("change points returned, $k$")
    ax.set_title("Detections grow with the log; true change points do not", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig4_detection_rate.png")


def fig8(data, out):
    rs = [r for r in rows(os.path.join(data, "results.csv")) if r["experiment"] == "E4"]
    widths = [0, 250, 1000, 2500]
    by = defaultdict(dict)
    for r in rs:
        by[r["algorithm"]][int(float(r["gradual_cases"]))] = num(r["best"])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for alg in SHOW:
        c, m = STYLE[alg]
        pts = [(w, by[alg].get(w)) for w in widths]
        pts = [(w, v) for w, v in pts if v is not None and v < 5000]
        if pts:
            ax.plot([w for w, _ in pts], [v for _, v in pts], marker=m, color=c, ms=4.5,
                    lw=1.4, label=SHORT[alg])
    ax.axhline(200, color=GREY, ls="--", lw=1)
    ax.annotate("lag window: beyond this a detection scores zero", (60, 215),
                fontsize=7, color=GREY)
    ax.set_xlabel("cases over which the drift is phased in")
    ax.set_ylabel("distance to the true change point, cases")
    ax.set_title("No method is invariant to the shape of the drift", loc="left")
    ax.set_xticks(widths)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig8_drift_shape.png")


# ---------------------------------------------------------------- figure 5
def fig5(data, out, gen_dir):
    settings = [(500, 100), (1000, 125), (2000, 250), (4000, 500)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True)
    for ax, (w, s) in zip(axes.ravel(), settings):
        obs = os.path.join(data, "drift_curve_W%d.csv" % w if w != 2000 else "drift_curve.csv")
        if not os.path.exists(obs):
            obs = os.path.join(data, "drift_curve_W%d.csv" % w)
        o = rows(obs)
        oy = [num(r["tvd"], 0) for r in o]
        ox = [i / (len(oy) - 1) for i in range(len(oy))]
        ax.plot(ox, oy, color=INK, lw=0.9, label="observed")

        g = os.path.join(gen_dir, "log_tvd_W%d_S%d.csv" % (w, s))
        if os.path.exists(g):
            gr = rows(g)
            gy = [num(r["tvd"], 0) for r in gr]
            gx = [i / (len(gy) - 1) for i in range(len(gy))]
            ax.plot(gx, gy, color=ACCENT, lw=0.9, alpha=0.85, label="generated")
        ax.set_title("W = %s, step %d" % ("{:,}".format(w), s), loc="left")
        ax.set_ylim(0, 1.05)
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.supxlabel("position through the log", fontsize=9)
    fig.supylabel("variant distance", fontsize=9)
    fig.suptitle("The generated log reproduces the peak, the baseline and the ratio",
                 fontsize=9.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    save(fig, out, "fig5_variant_distance.png")


# ---------------------------------------------------------------- figure 6
PRE = ["Fault report", "Permit approval", "Supervisory review", "Receipt (Division)",
       "Receipt (Department)", "Receipt (Technician)", "Isolation", "Execution",
       "Completion approval"]
POST = [("Fault report", 0), ("Permit approval", 0), ("Supervisory review", 0),
        ("Receipt (Division)", 0), ("Receipt (Department)", 0),
        ("Ack. (Division)", 1), ("Receipt (Technician)", 0), ("Ack. (Department)", 1),
        ("Isolation", 0), ("Ack. (Technician)", 1), ("Execution", 0),
        ("Report (Dept head)", 1), ("Completion approval", 0), ("Report approval", 1)]


def fig6(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    def draw(y, items, label):
        ax.text(-0.6, y, label, ha="right", va="center", fontsize=8.5, color=INK)
        for i, it in enumerate(items):
            name, new = (it, 0) if isinstance(it, str) else it
            face = "#f4c6c6" if new else "#e8e8e8"
            edge = ACCENT if new else "#777777"
            ax.add_patch(plt.Rectangle((i, y - 0.32), 0.86, 0.64, facecolor=face,
                                       edgecolor=edge, lw=1.0, zorder=2))
            ax.text(i + 0.43, y, name, ha="center", va="center", fontsize=5.6,
                    zorder=3, wrap=True)
            if i:
                ax.annotate("", (i, y), (i - 0.14, y), zorder=1,
                            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.7))
    draw(1.0, PRE, "before\n16 July 2020")
    draw(0.0, POST, "after")
    ax.set_xlim(-3.4, 14.2); ax.set_ylim(-0.8, 1.8)
    ax.axis("off")
    ax.set_title("Five control steps were added; the operational core was not touched",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "fig6_process_model.png")


# ---------------------------------------------------------------- figure 7
AUTOM = [("Permit approval", 0.04), ("Receipt (Technician)", 0.18),
         ("Receipt (Department)", 0.03), ("Supervisory review", 0.28),
         ("Fault report", 0.09), ("Receipt (Division)", 0.14),
         ("Execution", 0.12), ("Completion approval", 0.09),
         ("Isolation", 0.03), ("Report (Dept head)", 0.0),
         ("Report approval", 0.003), ("Ack. (Technician)", 71.6),
         ("Ack. (Department)", 80.4), ("Ack. (Division)", 86.4)]


def fig7(out):
    items = sorted(AUTOM, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ys = range(len(items))
    vals = [max(v, 0.002) for _, v in items]
    cols = [ACCENT if v > 50 else "#8c8c8c" for _, v in items]
    ax.barh(list(ys), vals, color=cols, height=0.68)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([k for k, _ in items], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.0015, 200)
    ax.axvspan(0.4, 71, color="#f0f0f0", zorder=0)
    ax.annotate("no activity falls in this band", (1.7, 1.2), fontsize=7.5, color=GREY,
                ha="center")
    ax.set_xlabel("events closed without ever being opened, per cent")
    ax.set_title("Automation is bimodal, and it is a property of events", loc="left")
    fig.tight_layout()
    save(fig, out, "fig7_system_generated.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--generated", default="gen_curves")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("\n  writing to %s\n" % a.out)
    fig1(a.data, a.out)
    fig2(a.data, a.out)
    fig3(a.data, a.out)
    fig4(a.data, a.out)
    fig5(a.data, a.out, a.generated)
    fig6(a.out)
    fig7(a.out)
    fig8(a.data, a.out)
    print()


if __name__ == "__main__":
    main()
