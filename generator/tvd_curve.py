#!/usr/bin/env python3
"""
tvd_curve.py — total variation distance over the variant distribution.

Two adjacent windows of W cases slide across the log in steps of STEP. For each
position the variant distributions of the two windows are compared:

    TVD(P, Q) = 0.5 * sum_v |P(v) - Q(v)|

Cases are ordered by start time, never by identifier. The curve is reported
together with its peak, its mean, and the ratio between them, which is the
figure that says whether a peak is a signal or the top of the noise.

    python tvd_curve.py --input synthetic_log/log.csv
    python tvd_curve.py --input log.csv --w 2000 --step 250
    python tvd_curve.py --input log.csv --sweep      # the four settings

Writes <input>_tvd_W<w>_S<step>.csv next to the input.
Standard library only.
"""

import argparse, csv, os, sys
from collections import Counter

SWEEP = [(500, 100), (1000, 125), (2000, 250), (4000, 500)]


def load(path, case_col, act_col, ts_col):
    if not os.path.exists(path):
        sys.exit("not found: %s" % path)
    seqs, starts = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        for c in (case_col, act_col, ts_col):
            if c not in rdr.fieldnames:
                sys.exit("column %r not in %s\ncolumns: %s"
                         % (c, path, ", ".join(rdr.fieldnames)))
        for row in rdr:
            cid, ts = row[case_col], row[ts_col]
            seqs.setdefault(cid, []).append(row[act_col])
            if cid not in starts or ts < starts[cid]:
                starts[cid] = ts
    order = sorted(starts, key=lambda c: starts[c])
    return [(c, starts[c], ">".join(seqs[c])) for c in order]


def tvd(a, b):
    ca, cb = Counter(a), Counter(b)
    na, nb = sum(ca.values()), sum(cb.values())
    if not na or not nb:
        return 0.0
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys)


def curve(cases, w, step):
    variants = [c[2] for c in cases]
    out = []
    i = w
    while i + w <= len(cases):
        out.append((i, cases[i][1], tvd(variants[i - w:i], variants[i:i + w])))
        i += step
    return out


def report(cases, w, step, truths, write_to=None):
    pts = curve(cases, w, step)
    if not pts:
        print("  W=%-5d STEP=%-4d  log too short (%d cases)" % (w, step, len(cases)))
        return None
    peak = max(pts, key=lambda p: p[2])
    mean = sum(p[2] for p in pts) / len(pts)
    ratio = peak[2] / mean if mean else float("inf")
    near = min((abs(peak[0] - t), t) for t in truths) if truths else (None, None)

    print("  W=%-5d STEP=%-4d  peak %.4f at case %-7d  mean %.4f  ratio %.2f  "
          "distance to nearest true point: %s"
          % (w, step, peak[2], peak[0], mean, ratio,
             near[0] if near[0] is not None else "-"))

    if write_to:
        with open(write_to, "w", newline="", encoding="utf-8") as fh:
            ww = csv.writer(fh)
            ww.writerow(["case_index", "timestamp", "tvd"])
            ww.writerows(pts)
    return dict(w=w, step=step, peak=round(peak[2], 4), peak_index=peak[0],
                mean=round(mean, 4), ratio=round(ratio, 2), distance=near[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--case", default="case_id")
    ap.add_argument("--activity", default="activity")
    ap.add_argument("--timestamp", default="timestamp")
    ap.add_argument("--w", type=int, default=2000)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--truth", default="",
                    help="comma-separated true change-point case indices; "
                         "read from ground_truth.json beside the input if absent")
    args = ap.parse_args()

    cases = load(args.input, args.case, args.activity, args.timestamp)
    print("\n  %s\n  %d cases" % (args.input, len(cases)))

    truths = []
    if args.truth:
        truths = [int(x) for x in args.truth.split(",") if x.strip()]
    else:
        gt = os.path.join(os.path.dirname(args.input) or ".", "ground_truth.json")
        if os.path.exists(gt):
            import json
            truths = [d["case_index"] for d in json.load(open(gt, encoding="utf-8"))["drifts"]]
    if truths:
        print("  true change points: %s" % ", ".join(str(t) for t in truths))
    print()

    base = os.path.splitext(args.input)[0]
    settings = SWEEP if args.sweep else [(args.w, args.step)]
    for w, s in settings:
        report(cases, w, s, truths, "%s_tvd_W%d_S%d.csv" % (base, w, s))
    print()


if __name__ == "__main__":
    main()
