# -*- coding: utf-8 -*-
"""
analyse_reference_results.py — experiment E0.

Reads `algorithm_results.csv`, the output of the reference evaluation of Adams
and colleagues, and asks one question of it: what does F1 measure on these logs?

Every row of that file records, for one algorithm at one parameter setting on
one log, the change points it detected and the change points that are actually
there. Precision is the number of detections that land near a true change point
divided by the number of detections. Since at most one detection can be credited
to each true change point,

    precision <= T / k

where T is the number of true change points in the log and k the number of
detections returned. The bound is arithmetic, not empirical. What is empirical
is how close to it the algorithms sit, and therefore how much of F1 is decided
by the ratio of detections to true drifts rather than by the quality of the
detections.

The published collections hold that ratio near one by construction: two or nine
drifts in a log of a few thousand cases, against algorithms that return a
detection every few hundred cases. An industrial log has the same detection rate
and one or two orders of magnitude fewer real drifts.

Scoring uses the reference implementation's own assignment function, so the
numbers here are theirs and not a reimplementation.

    python analyse_reference_results.py --cdrift C:/.../cdrift-clean
    python analyse_reference_results.py --cdrift ... --algorithm "Earth Mover"

Writes reference_analysis.csv and prints the summary tables.
"""

import argparse, ast, csv, os, sys, math
from collections import defaultdict

VERSION = "2026-09-19c"   # adds the per-algorithm control and log sizing
LAG = 200


def parse_list(s):
    if s is None:
        return []
    s = s.strip()
    if not s or s in ("[]", "nan", "None"):
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for x in v:
        try:
            out.append(int(round(float(x))))
        except (TypeError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdrift", required=True)
    ap.add_argument("--csv", default="algorithm_results.csv")
    ap.add_argument("--lag", type=int, default=LAG)
    ap.add_argument("--algorithm", default="")
    ap.add_argument("--out", default="experiments/reference_analysis.csv")
    ap.add_argument("--logs", default="",
                    help="EvaluationLogs folder; measures the case count of each collection")
    args = ap.parse_args()
    print("\n  analyse_reference_results.py  version %s" % VERSION)

    if not os.path.isdir(args.cdrift):
        sys.exit("cdrift repository not found: %s" % args.cdrift)
    path = args.csv if os.path.isabs(args.csv) else os.path.join(args.cdrift, args.csv)
    if not os.path.exists(path):
        sys.exit("results file not found: %s" % path)

    sys.path.insert(0, args.cdrift)
    import numpy as np
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    from cdrift import evaluation as cdrift_eval

    print("\n  reading %s" % path)
    rows, skipped = [], 0
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if args.algorithm and args.algorithm.lower() not in r["Algorithm"].lower():
                continue
            detected = parse_list(r.get("Detected Changepoints"))
            actual = parse_list(r.get("Actual Changepoints for Log"))
            if not actual:
                skipped += 1
                continue
            k, T = len(detected), len(actual)
            if k == 0:
                precision = recall = f1 = 0.0
                tp = 0
            else:
                tp, fp = cdrift_eval.getTP_FP(detected, actual, args.lag)
                precision = tp / k
                recall = tp / T
                f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            rows.append(dict(
                algorithm=r["Algorithm"], source=r.get("Log Source", ""), log=r.get("Log", ""),
                k=k, T=T, tp=tp, ratio_T_over_k=round(T / k, 4) if k else None,
                bound=round(min(1.0, T / k), 4) if k else 0.0,
                precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
            ))
    print("  %d rows scored, %d skipped for want of a ground truth\n" % (len(rows), skipped))
    if not rows:
        sys.exit("nothing to analyse")

    # --- 1. how tight is the bound ------------------------------------------
    withk = [r for r in rows if r["k"]]
    tight = [r for r in withk if r["precision"] >= 0.95 * r["bound"]]
    print("  " + "=" * 72)
    print("  1. IS PRECISION DECIDED BY THE RATIO OF DETECTIONS TO TRUE DRIFTS?")
    print("  " + "=" * 72)
    print("  rows with at least one detection      : %d" % len(withk))
    print("  rows where precision reaches 95%% of T/k: %d  (%.1f%%)"
          % (len(tight), 100 * len(tight) / len(withk)))
    mean_gap = sum(r["bound"] - r["precision"] for r in withk) / len(withk)
    print("  mean distance below the bound          : %.4f" % mean_gap)

    # --- 2. F1 against the ratio, bucketed ----------------------------------
    print("\n  " + "=" * 72)
    print("  2. F1 AGAINST T/k")
    print("  " + "=" * 72)
    buckets = [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.25),
               (0.25, 0.5), (0.5, 1.0), (1.0, 1e9)]
    print("  %-14s %8s %10s %10s %10s" % ("T/k", "rows", "mean F1", "mean prec", "mean rec"))
    for lo, hi in buckets:
        sel = [r for r in withk if lo <= r["ratio_T_over_k"] < hi]
        if not sel:
            continue
        label = "%.2f - %.2f" % (lo, hi) if hi < 1e8 else "1.00 and above"
        print("  %-14s %8d %10.3f %10.3f %10.3f"
              % (label, len(sel),
                 sum(r["f1"] for r in sel) / len(sel),
                 sum(r["precision"] for r in sel) / len(sel),
                 sum(r["recall"] for r in sel) / len(sel)))

    # --- 3. where the published collections sit on that axis ----------------
    print("\n  " + "=" * 72)
    print("  3. WHERE THE PUBLISHED COLLECTIONS SIT")
    print("  " + "=" * 72)
    by_source = defaultdict(list)
    for r in withk:
        by_source[r["source"]].append(r)
    print("  %-12s %8s %10s %10s %10s %10s"
          % ("collection", "rows", "med T", "med k", "med T/k", "mean F1"))
    for src, sel in sorted(by_source.items()):
        med = lambda key: sorted(x[key] for x in sel)[len(sel) // 2]
        print("  %-12s %8d %10d %10d %10.3f %10.3f"
              % (src or "(none)", len(sel), med("T"), med("k"),
                 sorted(x["ratio_T_over_k"] for x in sel)[len(sel) // 2],
                 sum(x["f1"] for x in sel) / len(sel)))

    # --- 4. per algorithm, at its best F1 -----------------------------------
    print("\n  " + "=" * 72)
    print("  4. EACH ALGORITHM AT ITS BEST PARAMETER SETTING")
    print("  " + "=" * 72)
    best = {}
    for r in withk:
        cur = best.get(r["algorithm"])
        if cur is None or r["f1"] > cur["f1"]:
            best[r["algorithm"]] = r
    print("  %-22s %6s %6s %9s %8s" % ("algorithm", "k", "T", "T/k", "F1"))
    for a, r in sorted(best.items(), key=lambda kv: -kv[1]["f1"]):
        print("  %-22s %6d %6d %9.3f %8.3f" % (a[:22], r["k"], r["T"], r["ratio_T_over_k"], r["f1"]))

    # --- 5. the control: hold the algorithm and the log fixed ---------------
    # Within one algorithm on one log, T is fixed and the parameter sweep varies
    # k. If F1 still falls as k rises while recall does not, nothing about the
    # log or the method can account for it.
    print("\n  " + "=" * 72)
    print("  5. CONTROL: SAME ALGORITHM, SAME LOG, PARAMETERS VARYING k")
    print("  " + "=" * 72)
    groups = defaultdict(list)
    for r in withk:
        groups[(r["algorithm"], r["log"])].append(r)

    usable = {g: v for g, v in groups.items()
              if len({x["k"] for x in v}) >= 3 and len({x["T"] for x in v}) == 1}
    print("  algorithm-log pairs with three or more distinct k: %d" % len(usable))

    def spearman(xs, ys):
        def rank(v):
            srt = sorted(v)
            return [(srt.index(a) + 1 + len(srt) - 1 - srt[::-1].index(a)) / 2 for a in v]
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else None

    rhos, neg, recall_flat = [], 0, 0
    for g, v in usable.items():
        rho = spearman([x["k"] for x in v], [x["f1"] for x in v])
        if rho is None:
            continue
        rhos.append(rho)
        if rho < 0:
            neg += 1
        if min(x["recall"] for x in v) >= 0.999:
            recall_flat += 1
    if rhos:
        rhos.sort()
        print("  median Spearman rho between k and F1 within a pair : %+.3f" % rhos[len(rhos) // 2])
        print("  pairs where more detections means lower F1         : %d of %d  (%.1f%%)"
              % (neg, len(rhos), 100 * neg / len(rhos)))
        print("  pairs where recall is 1.000 at every setting       : %d  (%.1f%%)"
              % (recall_flat, 100 * recall_flat / len(rhos)))

    # the same, pooled: bucket by k with T fixed at the commonest value
    from collections import Counter
    common_T = Counter(x["T"] for x in withk).most_common(1)[0][0]
    sel = [x for x in withk if x["T"] == common_T]
    print("\n  pooled over every log with T = %d  (%d results)" % (common_T, len(sel)))
    print("  %-14s %8s %10s %10s %10s" % ("k", "results", "mean F1", "mean prec", "mean rec"))
    for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 10 ** 9)]:
        b = [x for x in sel if lo <= x["k"] < hi]
        if not b:
            continue
        label = "%d - %d" % (lo, hi - 1) if hi < 10 ** 8 else "32 and above"
        print("  %-14s %8d %10.3f %10.3f %10.3f"
              % (label, len(b), sum(x["f1"] for x in b) / len(b),
                 sum(x["precision"] for x in b) / len(b),
                 sum(x["recall"] for x in b) / len(b)))

    # --- 6. the same, one algorithm at a time -------------------------------
    # If the shape above were an artefact of good algorithms happening to return
    # few change points, it would disappear here. Each row below holds both the
    # algorithm and T fixed, and varies only the parameter setting.
    print("\n  " + "=" * 72)
    print("  6. CONTROL BY ALGORITHM: T = %d THROUGHOUT, ONLY k VARIES" % common_T)
    print("  " + "=" * 72)
    bands = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 10 ** 9)]
    labels = ["k=1", "k=2-3", "k=4-7", "k=8-15", "k>=16"]
    print("  %-24s %s" % ("algorithm", " ".join("%12s" % l for l in labels)))
    print("  %-24s %s" % ("", " ".join("%12s" % "F1 / recall" for _ in labels)))
    for alg in sorted({x["algorithm"] for x in sel}):
        cells = []
        for lo, hi in bands:
            b = [x for x in sel if x["algorithm"] == alg and lo <= x["k"] < hi]
            cells.append("%.2f/%.2f" % (sum(x["f1"] for x in b) / len(b),
                                        sum(x["recall"] for x in b) / len(b))
                         if len(b) >= 10 else "-")
        if any(c != "-" for c in cells):
            print("  %-24s %s" % (alg[:24], " ".join("%12s" % c for c in cells)))
    print("\n  cells with fewer than ten results are left blank.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # --- 7. how large are the logs of each collection ------------------------
    if args.logs:
        import glob as _glob
        from cdrift.utils import helpers
        print("\n  " + "=" * 72)
        print("  7. LOG SIZE BY COLLECTION")
        print("  " + "=" * 72)
        print("  %-12s %7s %10s %10s %10s" % ("collection", "logs", "min cases", "median", "max"))
        for d in sorted(_glob.glob(os.path.join(args.logs, "*"))):
            if not os.path.isdir(d):
                continue
            paths = []
            for ext in ("*.xes", "*.xes.gz", "*.mxml", "*.mxml.gz"):
                paths += _glob.glob(os.path.join(d, ext))
            if not paths:
                continue
            paths.sort()
            probe = paths if len(paths) <= 6 else paths[:3] + paths[len(paths) // 2:len(paths) // 2 + 1] + paths[-2:]
            sizes = []
            for p in probe:
                try:
                    sizes.append(len(helpers.importLog(p, verbose=False)))
                except Exception:
                    pass
            if sizes:
                sizes.sort()
                print("  %-12s %7d %10d %10d %10d"
                      % (os.path.basename(d), len(paths), sizes[0],
                         sizes[len(sizes) // 2], sizes[-1]))
        print("\n  sizes are measured on a sample of up to six logs per collection.")

    print("\n  written: %s  (%d rows)\n" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
