# -*- coding: utf-8 -*-
"""
run_experiments.py — the four experiments of the synthetic-log paper.

Builds logs with gen_erp_drift.py and runs the seven published detectors from
Adams et al. (PVLDB 2023) against them. Unlike the real log, the change point
here is known exactly rather than inferred, so every experiment below is an
exact measurement rather than an estimate.

  E1  LOG-LENGTH DOMINANCE
      The same drift, at the same relative position, in logs of 1,000 to
      60,000 cases. F1 is expected to collapse while the localisation error
      stays flat, because precision is roughly 1/k and k grows with the log.
      This could not be shown on a single real log.

  E2  THE METADATA TRAP
      Every detection scored twice: against the true change point, and against
      the date the configuration table declares. The gap between the two F1
      columns is the error a researcher makes by trusting metadata.

  E3  ACTIVITY-LEVEL VERSUS EVENT-LEVEL FILTERING
      The measured drift magnitude under three preprocessing choices, against
      the magnitude the generator actually produced. No detector needed.

  E4  DRIFT SHAPE
      Sudden against gradual, at four transition widths.

Prerequisites, once:
    git clone https://github.com/cpitsch/cdrift-evaluation.git
    pip install POT strsimpy ruptures hotelling pulp scikit-learn deprecation pyarrow pm4py

    python run_experiments.py                 # all four
    python run_experiments.py --only 1,3      # a subset
    python run_experiments.py --only 3        # E3 needs no cdrift install
"""

import argparse, csv, json, os, subprocess, sys, time, types, warnings
warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
CDRIFT_DIR = r"C:\Users\Amin\cdrift-benchmark\cdrift-evaluation"
GENERATOR  = "gen_erp_drift.py"
OUT_DIR    = "experiments"

REQUIRED_GENERATOR = "2026-09-18a"   # older generators void experiment 2

LAG  = 200          # permitted lag, as in the VLDB paper
W, S = 200, 50      # window / step
SEED = 7

E1_SIZES    = [1000, 6000, 20000, 60000]
E4_GRADUAL  = [0, 250, 1000, 2500]
E4_SIZE     = 20000
# ================================================

import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan


def install_emd_shim():
    """wasserstein needs a compiler on Windows; POT gives an identical result."""
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


# ---------------------------------------------------------------- generation
def generate(tag, cases, gradual=0, seed=SEED):
    out = os.path.join(OUT_DIR, tag)
    cmd = [sys.executable, GENERATOR, "--cases", str(cases), "--seed", str(seed),
           "--out", out, "--quiet"]
    if gradual:
        cmd += ["--gradual", str(gradual)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit("generator failed for %s:\n%s" % (tag, r.stderr))
    with open(os.path.join(out, "ground_truth.json"), encoding="utf-8") as fh:
        gt = json.load(fh)
    got = gt.get("generator_version")
    if got != REQUIRED_GENERATOR:
        sys.exit("\n  %s is version %s, this runner needs %s.\n"
                 "  Replace the generator and run again.\n"
                 % (GENERATOR, got or "(unversioned, pre-2026-09-17b)", REQUIRED_GENERATOR))
    return out, gt


def read_log(folder):
    import pandas as pd
    df = pd.read_csv(os.path.join(folder, "log.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def to_xes(df, path):
    import pm4py
    log = pm4py.format_dataframe(df, case_id="case_id",
                                 activity_key="activity", timestamp_key="timestamp")
    pm4py.write_xes(log, path)
    return path


# ---------------------------------------------------------------- scoring
def score(detected, gt_index, lag, n_cases):
    """F1 plus a null model: is the hit distinguishable from chance?"""
    d = sorted(int(c) for c in detected)
    if not d:
        return dict(n=0, best=None, precision=0.0, recall=0.0, f1=0.0,
                    p_null=None, expected_if_random=None)
    tp = sum(1 for c in d if abs(c - gt_index) <= lag)
    precision = tp / len(d)
    recall = 1.0 if tp else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    best = min(abs(c - gt_index) for c in d)
    k = len(d)
    p_null = 1 - (1 - min((2 * best + 1) / n_cases, 1.0)) ** k
    return dict(n=k, best=best, precision=round(precision, 4), recall=recall,
                f1=round(f1, 4), p_null=round(p_null, 4),
                expected_if_random=round(n_cases / (2 * (k + 1))))


def detectors(log):
    from cdrift.approaches import earthmover, bose, lcdd
    from cdrift.approaches import maaradji as runs
    from cdrift.approaches.zheng import applyMultipleEps
    from cdrift.approaches import process_graph_metrics as pgm
    return [
        ("EMD (Brockhoff 2020)",
         lambda: earthmover.detect_change(log, W, S, show_progress_bar=False)),
        ("Bose J (2011/2014)",
         lambda: bose.visualInspection_Step(
             bose.detectChange_JMeasure_KS_Step(log, W, step_size=S, show_progress_bar=False), W, S)),
        ("Bose WC (2011/2014)",
         lambda: bose.visualInspection_Step(
             bose.detectChange_WC_KS_Step(log, W, step_size=S, show_progress_bar=False), W, S)),
        ("ProDrift (Maaradji 2015/17)",
         lambda: runs.detectChangepoints_Stride(
             log, W, S, pvalue=0.05, return_pvalues=False, show_progress_bar=False)),
        ("RINV (Zheng 2017)",
         lambda: applyMultipleEps(log, mrid=W, epsList=[0.1, 0.2, 0.3], show_progress_bar=False)),
        ("LCDD (Lin 2020)",
         lambda: lcdd.calculate(log, complete_window_size=W,
                                detection_window_size=W, stable_period=10)),
        ("PGM (Seeliger 2017)",
         lambda: pgm.detectChange(log, W, 2 * W, pvalue=0.05, show_progress_bar=False)),
    ]


def run_detectors(xes_path):
    """Returns [(name, detected_list_or_None, seconds, error)]."""
    from cdrift.utils import helpers
    log = helpers.importLog(xes_path, verbose=False)
    out = []
    for name, fn in detectors(log):
        t0 = time.time()
        try:
            cps = [int(c) for c in fn()]
            out.append((name, cps, time.time() - t0, None))
        except Exception as e:
            out.append((name, None, time.time() - t0, "%s: %s" % (type(e).__name__, e)))
    return out, len(log)


def declared_index(folder, df):
    """Case index of the date the configuration table declares for the added steps."""
    import pandas as pd
    meta = {}
    with open(os.path.join(folder, "metadata.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["lag_days"] and int(row["lag_days"]) > 0:
                meta[row["activity"]] = row["declared_date"]
    if not meta:
        return None, None
    declared = min(meta.values())
    starts = df.groupby("case_id")["timestamp"].min().sort_values()
    idx = int((starts < pd.Timestamp(declared)).sum())
    return idx, declared


# ---------------------------------------------------------------- experiments
def experiment_1(rows):
    print("\n" + "=" * 78)
    print("E1  LOG-LENGTH DOMINANCE")
    print("=" * 78)
    print("%-28s %7s %5s %7s %9s %7s %7s" %
          ("algorithm", "cases", "#cp", "dist", "E[rand]", "p_null", "F1"))
    print("-" * 78)
    for n in E1_SIZES:
        folder, gt = generate("e1_%d" % n, n)
        df = read_log(folder)
        xes = to_xes(df, os.path.join(folder, "log.xes.gz"))
        cp = [d for d in gt["drifts"] if d["id"] == "C"][0]["case_index"]
        res, n_cases = run_detectors(xes)
        for name, cps, sec, err in res:
            if err:
                print("%-28s %7d   FAIL  %s" % (name, n, err[:34]))
                rows.append(dict(experiment="E1", cases=n, algorithm=name, error=err))
                continue
            sc = score(cps, cp, LAG, n_cases)
            print("%-28s %7d %5d %7s %9s %7s %7.3f" %
                  (name, n, sc["n"], sc["best"], sc["expected_if_random"],
                   sc["p_null"], sc["f1"]))
            rows.append(dict(experiment="E1", cases=n, algorithm=name,
                             seconds=round(sec, 1), ground_truth=cp, **sc))
        print("-" * 78)
    print("read down each algorithm's rows: F1 falls with log length while dist does not.")


def experiment_2(rows):
    print("\n" + "=" * 78)
    print("E2  THE METADATA TRAP")
    print("=" * 78)
    folder, gt = generate("e2", 20000)
    df = read_log(folder)
    xes = to_xes(df, os.path.join(folder, "log.xes.gz"))
    cp = [d for d in gt["drifts"] if d["id"] == "C"][0]["case_index"]
    decl_idx, decl_date = declared_index(folder, df)
    if not decl_idx or abs(cp - decl_idx) < 5 * LAG:
        sys.exit("\n  the declared index came out as %s against a true change point of %d.\n"
                 "  The metadata lag did not take effect, so E2 would be meaningless.\n"
                 "  Check experiments/e2/metadata.csv: rows A11-A14 should show a\n"
                 "  declared_date well before their first_event_date.\n" % (decl_idx, cp))
    print("true change point   : case %d" % cp)
    print("declared in metadata: case %d  (%s)   gap %d cases\n" %
          (decl_idx, decl_date, cp - decl_idx))
    print("%-28s %10s %12s %10s" % ("algorithm", "F1 (true)", "F1 (declared)", "dist"))
    print("-" * 78)
    res, n_cases = run_detectors(xes)
    for name, cps, sec, err in res:
        if err:
            print("%-28s   FAIL  %s" % (name, err[:40]))
            continue
        a = score(cps, cp, LAG, n_cases)
        b = score(cps, decl_idx, LAG, n_cases)
        print("%-28s %10.3f %12.3f %10s" % (name, a["f1"], b["f1"], a["best"]))
        rows.append(dict(experiment="E2", algorithm=name, ground_truth=cp,
                         declared_index=decl_idx, f1_true=a["f1"], f1_declared=b["f1"],
                         best_distance=a["best"]))
    print("-" * 78)
    print("every algorithm that finds the real boundary scores zero against metadata.")


def experiment_3(rows):
    print("\n" + "=" * 78)
    print("E3  ACTIVITY-LEVEL VERSUS EVENT-LEVEL FILTERING")
    print("=" * 78)
    folder, gt = generate("e3", 20000)
    df = read_log(folder)
    cp = [d for d in gt["drifts"] if d["id"] == "C"][0]["case_index"]
    df["idx"] = df["case_id"].str[1:].astype(int)
    df["period"] = np.where(df["idx"] < cp, "PRE", "POST")

    # activities a naive filter would drop: predominantly system-generated
    share = df.groupby("activity")["is_system_generated"].mean()
    drop = set(share[share > 0.5].index)

    def magnitude(d):
        g = d.groupby(["period", "case_id"]).size().groupby("period").mean()
        if "PRE" not in g or "POST" not in g:
            return None, None
        return round(g["POST"] - g["PRE"], 3), round(100 * (g["POST"] / g["PRE"] - 1), 1)

    variants = [
        ("unfiltered", df),
        ("activity-level filter (drops %s)" % ",".join(sorted(drop)),
         df[~df["activity"].isin(drop)]),
        ("event-level filter", df[df["is_system_generated"] == 0]),
    ]
    print("%-46s %10s %10s" % ("preprocessing", "delta", "percent"))
    print("-" * 78)
    truth = None
    for label, d in variants:
        delta, pct = magnitude(d)
        if label == "event-level filter":
            truth = delta
        print("%-46s %10s %9s%%" % (label[:46], delta, pct))
        rows.append(dict(experiment="E3", preprocessing=label,
                         delta_events_per_case=delta, percent_change=pct))
    print("-" * 78)
    print("the event-level figure (%s) is the true human-work increase; the" % truth)
    print("activity-level figure discards real human events along with the automated ones.")


def experiment_4(rows):
    print("\n" + "=" * 78)
    print("E4  DRIFT SHAPE")
    print("=" * 78)
    print("%-28s %9s %5s %7s %7s" % ("algorithm", "gradual", "#cp", "dist", "F1"))
    print("-" * 78)
    for g in E4_GRADUAL:
        folder, gt = generate("e4_%d" % g, E4_SIZE, gradual=g)
        df = read_log(folder)
        xes = to_xes(df, os.path.join(folder, "log.xes.gz"))
        cp = [d for d in gt["drifts"] if d["id"] == "C"][0]["case_index"]
        res, n_cases = run_detectors(xes)
        for name, cps, sec, err in res:
            if err:
                print("%-28s %9d   FAIL" % (name, g))
                continue
            sc = score(cps, cp, LAG, n_cases)
            print("%-28s %9d %5d %7s %7.3f" % (name, g, sc["n"], sc["best"], sc["f1"]))
            rows.append(dict(experiment="E4", gradual_cases=g, algorithm=name,
                             ground_truth=cp, **sc))
        print("-" * 78)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2,3,4")
    ap.add_argument("--cdrift", default=CDRIFT_DIR)
    args = ap.parse_args()
    want = {x.strip() for x in args.only.split(",")}

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(GENERATOR):
        sys.exit("generator not found next to this script: %s" % GENERATOR)

    if want - {"3"}:
        if not os.path.isdir(args.cdrift):
            sys.exit("cdrift repository not found: %s\n"
                     "clone it, or run with --only 3" % args.cdrift)
        backend = install_emd_shim()
        sys.path.insert(0, args.cdrift)
        print("EMD backend: %s" % backend)

    rows = []
    if "1" in want: experiment_1(rows)
    if "2" in want: experiment_2(rows)
    if "3" in want: experiment_3(rows)
    if "4" in want: experiment_4(rows)

    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        out = os.path.join(OUT_DIR, "results.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("\nwritten: %s  (%d rows)\n" % (out, len(rows)))


if __name__ == "__main__":
    main()
