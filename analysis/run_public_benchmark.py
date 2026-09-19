# -*- coding: utf-8 -*-
"""
run_public_benchmark.py — experiment E0.

Runs the same harness used in run_experiments.py against the published
Business Process Drift collection of Maaradji and colleagues, downloadable from
4TU at https://doi.org/10.4121/uuid:aa01a720-4616-43e9-af67-370942019f48

The collection contains 75 logs. Each holds nine sudden control-flow drifts
placed at regular intervals, and the same eighteen change patterns are issued
at four log sizes: 2,500, 5,000, 7,500 and 10,000 traces. Drift i therefore sits
at case i * N / 10.

E0 does two things at once.

  VALIDATION. If our configuration of the seven algorithms reproduces the
  behaviour reported for them on these logs, the harness is sound, and the
  results on our generated logs can be read as findings rather than as
  misconfiguration. If it does not, we need to know that before submitting.

  MECHANISM. Drift density in this collection is held constant by construction:
  nine drifts however long the log. Density therefore ranges only from one drift
  per 278 cases to one per 1,111. The detection rate of a window-based algorithm
  is roughly one per 240 to 500 cases, so across this collection the number of
  detections and the number of true drifts are of the same order, and precision
  can be high. Our generated logs extend the range to one drift per 60,000
  cases. E0 supplies the lower end of that axis from data we did not produce.

    python run_public_benchmark.py --logs D:/bpdrift
    python run_public_benchmark.py --logs D:/bpdrift --pattern cb --limit 8

Expects the logs as .xes, .xes.gz or .mxml in one folder, at their original
file names.
"""

import argparse, csv, glob, os, re, sys, time, types, warnings
warnings.filterwarnings("ignore")

LAG = 200
W, S = 200, 50
N_DRIFTS = 9          # by construction of this collection

import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan


def install_emd_shim():
    try:
        import wasserstein  # noqa: F401
        return "wasserstein (original)"
    except ImportError:
        import ot

        class _EMD:
            def __call__(self, w1, w2, cost):
                a = np.asarray(w1, dtype=float)
                b = np.asarray(w2, dtype=float)
                a = a / a.sum() if a.sum() else a
                b = b / b.sum() if b.sum() else b
                M = np.ascontiguousarray(np.asarray(cost, dtype=float))
                return float(ot.emd2(a, b, M))

        shim = types.ModuleType("wasserstein")
        shim.EMD = _EMD
        sys.modules["wasserstein"] = shim
        return "POT (shim)"


def detectors(log):
    from cdrift.approaches import earthmover, bose, lcdd
    from cdrift.approaches import maaradji as runs
    from cdrift.approaches.zheng import applyMultipleEps
    from cdrift.approaches import process_graph_metrics as pgm
    return [
        ("EMD", lambda: earthmover.detect_change(log, W, S, show_progress_bar=False)),
        ("Bose J", lambda: bose.visualInspection_Step(
            bose.detectChange_JMeasure_KS_Step(log, W, step_size=S, show_progress_bar=False), W, S)),
        ("Bose WC", lambda: bose.visualInspection_Step(
            bose.detectChange_WC_KS_Step(log, W, step_size=S, show_progress_bar=False), W, S)),
        ("ProDrift", lambda: runs.detectChangepoints_Stride(
            log, W, S, pvalue=0.05, return_pvalues=False, show_progress_bar=False)),
        ("RINV", lambda: applyMultipleEps(log, mrid=W, epsList=[0.1, 0.2, 0.3], show_progress_bar=False)),
        ("LCDD", lambda: lcdd.calculate(log, complete_window_size=W,
                                        detection_window_size=W, stable_period=10)),
        ("PGM", lambda: pgm.detectChange(log, W, 2 * W, pvalue=0.05, show_progress_bar=False)),
    ]


def score_multi(detected, truths, lag, n_cases):
    """Precision, recall and F1 against several true change points.

    A detection is a true positive if it falls within lag of any true point; a
    true point is recalled if some detection falls within lag of it. This is the
    convention of the reference evaluation.
    """
    d = sorted(int(c) for c in detected)
    if not d:
        return dict(n=0, precision=0.0, recall=0.0, f1=0.0, mean_distance=None,
                    detected_per_1000=0.0)
    tp = sum(1 for c in d if any(abs(c - t) <= lag for t in truths))
    hit = sum(1 for t in truths if any(abs(c - t) <= lag for c in d))
    precision = tp / len(d)
    recall = hit / len(truths) if truths else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    mean_dist = sum(min(abs(c - t) for c in d) for t in truths) / len(truths)
    return dict(n=len(d), precision=round(precision, 4), recall=round(recall, 4),
                f1=round(f1, 4), mean_distance=round(mean_dist, 1),
                detected_per_1000=round(1000 * len(d) / n_cases, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="folder holding the published logs")
    ap.add_argument("--cdrift", default=r"C:\Users\Amin\cdrift-benchmark\cdrift-evaluation")
    ap.add_argument("--pattern", default="", help="only logs whose name contains this")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many logs")
    ap.add_argument("--out", default="experiments/public_benchmark.csv")
    args = ap.parse_args()

    files = []
    for ext in ("*.xes", "*.xes.gz", "*.mxml", "*.mxml.gz"):
        files += glob.glob(os.path.join(args.logs, "**", ext), recursive=True)
    if args.pattern:
        files = [f for f in files if args.pattern.lower() in os.path.basename(f).lower()]
    files.sort()
    if args.limit:
        files = files[:args.limit]
    if not files:
        if not os.path.isdir(args.logs):
            sys.exit("\n  that folder does not exist: %s\n"
                     "  download the collection first, extract it, and pass the\n"
                     "  folder holding the log files.\n" % args.logs)
        seen = {}
        for root, _, names in os.walk(args.logs):
            for nm in names:
                seen[os.path.splitext(nm)[1].lower() or "(no extension)"] = \
                    seen.get(os.path.splitext(nm)[1].lower() or "(no extension)", 0) + 1
        if not seen:
            sys.exit("\n  %s is empty.\n" % args.logs)
        sys.exit("\n  no .xes, .xes.gz, .mxml or .mxml.gz under %s\n"
                 "  what is there:\n%s\n"
                 "  if you see .zip, extract it first.\n"
                 % (args.logs,
                    "".join("    %-16s %d file(s)\n" % (e, c)
                            for e, c in sorted(seen.items(), key=lambda kv: -kv[1])[:10])))
    print("%d logs to process" % len(files))

    if not os.path.isdir(args.cdrift):
        sys.exit("cdrift repository not found: %s" % args.cdrift)
    print("EMD backend: %s\n" % install_emd_shim())
    sys.path.insert(0, args.cdrift)
    from cdrift.utils import helpers

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, skipped = [], []
    print("%-26s %7s %-9s %5s %8s %9s %7s" %
          ("log", "cases", "algorithm", "#cp", "dist", "cp/1000", "F1"))
    print("-" * 82)

    for path in files:
        name = os.path.basename(path)
        try:
            log = helpers.importLog(path, verbose=False)
        except Exception as e:
            print("%-26s  import failed: %s" % (name[:26], type(e).__name__))
            continue
        n = len(log)
        # Nine drifts at regular intervals is the construction of the Ostovar
        # collection only. The other collections shipped with the reference
        # evaluation are built differently and would be scored wrongly here.
        if n not in (2500, 5000, 7500, 10000):
            print("%-26s %7d  skipped: %d cases is not one of the four sizes of this\n"
                  "%-26s          collection, so the nine-drift construction cannot be assumed."
                  % (name[:26], n, n, ""))
            skipped.append((name, n))
            continue
        truths = [round(i * n / (N_DRIFTS + 1)) for i in range(1, N_DRIFTS + 1)]

        for alg, fn in detectors(log):
            t0 = time.time()
            try:
                cps = [int(c) for c in fn()]
                sc = score_multi(cps, truths, LAG, n)
                print("%-26s %7d %-9s %5d %8s %9s %7.3f" %
                      (name[:26], n, alg, sc["n"], sc["mean_distance"],
                       sc["detected_per_1000"], sc["f1"]))
                rows.append(dict(log=name, cases=n, n_true_drifts=len(truths),
                                 drift_density_per_1000=round(1000 * len(truths) / n, 2),
                                 algorithm=alg, seconds=round(time.time() - t0, 1), **sc))
            except Exception as e:
                print("%-26s %7d %-9s   FAIL %s" % (name[:26], n, alg, type(e).__name__))
                rows.append(dict(log=name, cases=n, algorithm=alg,
                                 error="%s: %s" % (type(e).__name__, e)))
        print("-" * 82)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if skipped:
        print("\n  %d log(s) skipped as not matching the nine-drift construction:" % len(skipped))
        for nm, n in skipped[:10]:
            print("    %s  (%d cases)" % (nm, n))
    print("\nwritten: %s  (%d rows)" % (args.out, len(rows)))
    print("\nwhat to read: cp/1000 is the detection rate and should be roughly")
    print("constant across log sizes; drift_density_per_1000 is fixed at 3.6 by")
    print("the construction of this collection. Where the two are comparable,")
    print("F1 is informative. Our generated logs move the second far below the")
    print("first, and that is where F1 stops being informative.\n")


if __name__ == "__main__":
    main()
