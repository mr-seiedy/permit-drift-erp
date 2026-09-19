#!/usr/bin/env python3
"""
calibrate.py — extracts the parameters needed to calibrate a synthetic
event-log generator, from the exported permit log.

Reads   : permit_log.csv   (pipe-delimited, as produced by queries.sql #15)
Writes  : calibration.json (aggregate statistics only)

Nothing identifying is written. No case id, no user id, no organisation id,
no timestamp, no free text. Only counts, ratios and distributions.

    python calibrate.py
    python calibrate.py --input D:/path/permit_log.csv

Standard library only.
"""

import csv, json, sys, os
from collections import defaultdict, Counter

CUT = "1399/04/26"          # drift C, 16 July 2020 — see queries.sql
COLS = ["ID", "OBJ_ID", "TAS_ID", "ORG_ID", "USE_ID_RECEIVER",
        "REFER_DATE_TIME", "VIEW_DATE_TIME", "DONE_DATE_TIME",
        "DONE_TYPE", "STATUS"]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def jdn(s):
    """Jalali 'YYYY/MM/DD_hh:mm:ss' -> absolute minute. None if unparseable."""
    if not s or len(s) < 10:
        return None
    try:
        jy, jm, jd = int(s[0:4]), int(s[5:7]), int(s[8:10])
        hh = int(s[11:13]) if len(s) >= 13 else 0
        mi = int(s[14:16]) if len(s) >= 16 else 0
    except ValueError:
        return None
    y = jy + 1595
    d = -355668 + (365 * y) + ((y // 33) * 8) + (((y % 33) + 3) // 4) + jd
    d += (jm - 1) * 31 if jm < 7 else ((jm - 7) * 30) + 186
    return d * 1440 + hh * 60 + mi


def detect_encoding(path):
    """bcp writes -c as the local code page and -w as UTF-16LE. Work out which."""
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if head.startswith(b"\xff\xfe"):
        return "utf-16"
    if head.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    # no BOM: UTF-16LE ASCII text puts a zero byte after every character
    if len(head) > 20 and head[1::2].count(0) > len(head) // 4:
        return "utf-16-le"
    return "utf-8"


def clean(fh):
    """Strip stray NULs and blank lines so csv.reader never chokes."""
    for line in fh:
        if "\x00" in line:
            line = line.replace("\x00", "")
        if line.strip():
            yield line


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo), 1)


def main():
    path = arg("--input", "permit_log.csv")
    if not os.path.exists(path):
        sys.exit("\n  not found: %s\n  run queries.sql #15 (bcp) first, or pass --input <path>\n" % path)

    enc = detect_encoding(path)
    print("reading %s  (encoding: %s, %.0f MB)" % (path, enc, os.path.getsize(path) / 1048576))

    cases = {}                       # obj_id -> [start, [(tas, refer, view, done), ...]]
    dur_rv, dur_vd, dur_rd = defaultdict(list), defaultdict(list), defaultdict(list)
    four_way = defaultdict(lambda: [0, 0, 0, 0])   # total, auto, human, open
    user_events = Counter()
    n = 0

    with open(path, "r", encoding=enc, errors="replace", newline="") as fh:
        sniff = fh.readline()
        fh.seek(0)
        rdr = csv.reader(clean(fh), delimiter="|")
        if "OBJ_ID" in sniff.upper():
            next(rdr)
        for row in rdr:
            if len(row) < 8:
                continue
            n += 1
            obj, tas = row[1], row[2]
            usr = row[4].strip()
            refer, view, done = row[5].strip(), row[6].strip(), row[7].strip()
            if usr and usr.upper() != "NULL":
                user_events[usr] += 1

            f = four_way[tas]
            f[0] += 1
            if not view and done:
                f[1] += 1
            elif view and done:
                f[2] += 1
            if not done:
                f[3] += 1

            c = cases.get(obj)
            if c is None:
                cases[obj] = c = [refer, []]
            elif refer < c[0]:
                c[0] = refer
            c[1].append(tas)

            r, v, d = jdn(refer), jdn(view), jdn(done)
            if r is not None and v is not None and 0 <= v - r <= 525600:
                dur_rv[tas].append(v - r)
            if v is not None and d is not None and 0 <= d - v <= 525600:
                dur_vd[tas].append(d - v)
            if r is not None and d is not None and 0 <= d - r <= 525600:
                dur_rd[tas].append(d - r)

            if n % 250000 == 0:
                print("  %d rows" % n)

    print("  %d rows, %d cases" % (n, len(cases)))

    out = {"source_rows": n, "cases": len(cases), "cut": CUT}

    # --- four-way split, every activity -------------------------------------
    out["four_way"] = {t: {"total": v[0], "auto": v[1], "human": v[2], "open": v[3]}
                       for t, v in sorted(four_way.items(), key=lambda kv: -kv[1][0])}

    # --- per period ----------------------------------------------------------
    dfg = {"PRE": Counter(), "POST": Counter()}
    pos = {"PRE": defaultdict(lambda: [0, 0]), "POST": defaultdict(lambda: [0, 0])}
    variants = {"PRE": Counter(), "POST": Counter()}
    lens = {"PRE": Counter(), "POST": Counter()}
    arrivals = Counter()

    for start, seq in cases.values():
        p = "PRE" if start < CUT else "POST"
        arrivals[start[:7]] += 1
        lens[p][len(seq)] += 1
        variants[p][">".join(seq)] += 1
        for i, t in enumerate(seq):
            a = pos[p][t]
            a[0] += i + 1
            a[1] += 1
            if i + 1 < len(seq):
                dfg[p][(t, seq[i + 1])] += 1

    out["dfg"] = {p: [{"from": a, "to": b, "n": c}
                      for (a, b), c in sorted(d.items(), key=lambda kv: -kv[1]) if c >= 200]
                  for p, d in dfg.items()}
    out["mean_position"] = {p: {t: {"mean_pos": round(a[0] / a[1], 2), "n": a[1]}
                                for t, a in sorted(d.items(), key=lambda kv: kv[1][0] / kv[1][1])}
                            for p, d in pos.items()}
    out["monthly_case_arrivals"] = dict(sorted(arrivals.items()))

    out["variants"] = {}
    for p, c in variants.items():
        tot = sum(c.values())
        top = c.most_common(20)
        out["variants"][p] = {
            "cases": tot,
            "distinct_variants": len(c),
            "top20_counts": [k for _, k in top],
            "top20_share": round(sum(k for _, k in top) / tot, 4),
            "singleton_variants": sum(1 for v in c.values() if v == 1),
        }

    out["trace_length_hist"] = {p: dict(sorted(c.items())) for p, c in lens.items()}

    # --- durations, minutes --------------------------------------------------
    dur = {}
    for tas in sorted(four_way, key=lambda t: -four_way[t][0]):
        rv, vd, rd = sorted(dur_rv[tas]), sorted(dur_vd[tas]), sorted(dur_rd[tas])
        dur[tas] = {
            "n_refer_to_view": len(rv),
            "refer_to_view": [pct(rv, .25), pct(rv, .5), pct(rv, .75), pct(rv, .9)],
            "view_to_done":   [pct(vd, .25), pct(vd, .5), pct(vd, .75), pct(vd, .9)],
            "refer_to_done":  [pct(rd, .25), pct(rd, .5), pct(rd, .75), pct(rd, .9)],
        }
    out["durations_minutes_q25_q50_q75_q90"] = dur

    # --- workload distribution, ranks only, no identifiers -------------------
    ranked = sorted(user_events.values(), reverse=True)
    out["user_workload"] = {
        "distinct_users": len(ranked),
        "top40_event_counts": ranked[:40],
        "total_attributed_events": sum(ranked),
        "deciles": [pct(sorted(ranked), q / 10) for q in range(1, 10)],
    }

    with open("calibration.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print("\n  written: calibration.json  (%.1f KB)" % (os.path.getsize("calibration.json") / 1024))
    print("  PRE  %d cases, %d distinct variants" % (out["variants"]["PRE"]["cases"],
                                                     out["variants"]["PRE"]["distinct_variants"]))
    print("  POST %d cases, %d distinct variants" % (out["variants"]["POST"]["cases"],
                                                     out["variants"]["POST"]["distinct_variants"]))
    print("  DFG edges kept: PRE %d, POST %d" % (len(out["dfg"]["PRE"]), len(out["dfg"]["POST"])))
    print("\n  upload calibration.json to the chat.\n")


if __name__ == "__main__":
    main()
