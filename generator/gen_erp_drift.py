#!/usr/bin/env python3
"""
gen_erp_drift.py — a synthetic event-log generator for concept drift research.

Every parameter is calibrated against an observed industrial ERP workflow, but
no observed record is reproduced: cases, timestamps and resources are all
generated. The generator exists to make available what the source log cannot
be: a drift benchmark anyone can download, with a change point that is known
exactly rather than inferred.

Three properties are reproduced that published generators do not model:

  1. METADATA LAG. The configuration table declares when a capability was
     defined, not when it began producing events. Four activities here are
     declared three years before their first event. A detector trusting the
     declaration date is wrong by that margin, and the emitted metadata.csv
     lets that error be measured.

  2. EVENT-LEVEL AUTOMATION. An activity is not wholly automated or wholly
     human. Each activity carries its own probability that a given event is
     closed by the system without ever being opened. Filtering at activity
     level, the common practice, therefore distorts the measured drift.

  3. ACKNOWLEDGEMENT LAG. Notification steps fire one position after the
     handover they acknowledge, and the further a role sits from the work,
     the less often its notification is opened.

    python gen_erp_drift.py                      # default 20,000 cases
    python gen_erp_drift.py --cases 60000 --seed 7
    python gen_erp_drift.py --gradual 2000       # drift spread over 2,000 cases
    python gen_erp_drift.py --out logs/run1

Outputs, all under --out:
    log.csv            case_id, activity, timestamp, resource, lifecycle, is_system
    log.xes            the same log in XES
    metadata.csv       the misleading declaration dates
    ground_truth.json  the true change points, by case index and timestamp

Standard library only.
"""

GENERATOR_VERSION = "2026-09-18a"   # + background churn; fraction-based metadata lag

import argparse, csv, json, math, os, random, xml.sax.saxutils as esc
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Calibration. Every figure below is an aggregate statistic of the observed
# system; none of it identifies a case, a person or an organisation.
# ---------------------------------------------------------------------------

ACTIVITIES = {
    "A01": "Fault report issued",
    "A02": "Permit approval",
    "A03": "Supervisory unit review",
    "A04": "Work request receipt (Division)",
    "A05": "Work request receipt (Department)",
    "A06": "Work request receipt (Technician)",
    "A07": "Isolation and work authorization",
    "A08": "Maintenance execution",
    "A09": "Work completion approval",
    "A10": "Work report approval",
    "A11": "Work report (Department head)",
    "A12": "Receipt acknowledgement (Division)",
    "A13": "Receipt acknowledgement (Department)",
    "A14": "Receipt acknowledgement (Technician)",
    "A15": "Supervisory unit attendance",
    "A16": "Safety attendance",
}

# Backbone before and after the addition drift. The acknowledgements sit one
# position after the handover they acknowledge, which is what the observed
# directly-follows counts show.
PATH_PRE = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]
PATH_POST = ["A01", "A02", "A03", "A04", "A05", "A12", "A06", "A13",
             "A07", "A14", "A08", "A11", "A09", "A10"]

# The replacement drift: A15 and A10-as-optional give way to A04 and A07.
PATH_EARLY = ["A01", "A02", "A03", "A05", "A06", "A15", "A08", "A09", "A10"]

# Probability that an event of this activity is closed without being opened.
P_SYSTEM = {
    "A12": 0.864, "A13": 0.804, "A14": 0.716,
    "A03": 0.0028, "A06": 0.0018, "A04": 0.0014, "A08": 0.0012,
    "A01": 0.0009, "A09": 0.0009, "A02": 0.0004, "A07": 0.0003,
    "A05": 0.0003, "A10": 0.00003, "A11": 0.0, "A15": 0.0, "A16": 0.0036,
}

# Probability that an event is referred and never completed at all.
P_OPEN = {
    "A12": 0.0131, "A13": 0.0126, "A14": 0.0046, "A08": 0.0022,
    "A06": 0.0026, "A03": 0.0013, "A04": 0.0006, "A09": 0.0003,
    "A05": 0.0003, "A02": 0.0001, "A07": 0.0001, "A10": 0.0002,
    "A11": 0.0008, "A15": 0.0, "A16": 0.0004,
}

# Waiting and handling time in minutes, as (q25, q50, q75) of the observed
# distribution. A lognormal is fitted to each. Zero means same-minute stamping.
WAIT = {   # referral -> first opened
    "A01": (0, 0, 0),        "A02": (5, 46, 211),
    "A03": (111, 666, 1182), "A04": (8, 24, 122),
    "A05": (3, 18, 133),     "A06": (64, 1119, 4298),
    "A07": (76, 269, 634),   "A08": (151, 1020, 2880),
    "A09": (82, 374, 1229),  "A10": (143, 875, 1861),
    "A11": (33, 189, 1387),  "A12": (3, 309, 7165),
    "A13": (1, 133, 5582),   "A14": (2, 164, 3001),
    "A15": (52, 393, 1362),  "A16": (57, 196, 1334),
}
HANDLE = {  # opened -> completed
    "A01": (0, 0, 0),     "A02": (0, 1, 2),      "A03": (0, 0, 1),
    "A04": (0, 0, 1),     "A05": (0, 0, 1),      "A06": (0, 1, 1575),
    "A07": (0, 0, 1),     "A08": (0, 1, 2),      "A09": (0, 0, 1),
    "A10": (0, 0, 0),     "A11": (0, 0, 1),      "A12": (0, 0, 11903),
    "A13": (0, 4324, 22674), "A14": (1409, 4278, 8752),
    "A15": (0, 0, 0),     "A16": (0, 0, 0),
}

# Deviation rates, tuned so that the dominant variant covers the observed share.
REWORK = {"A06": 0.118, "A07": 0.078, "A08": 0.030, "A09": 0.029,
          "A02": 0.024, "A05": 0.014, "A03": 0.017}
P_SKIP = 0.085          # a backbone step is occasionally not recorded
P_SWAP = 0.070          # two adjacent steps are occasionally recorded out of order
P_OPTIONAL_A16 = 0.052  # safety attendance, inserted before execution
P_TRUNCATE = 0.024      # case abandoned after one or two steps

# Behavioural variation is not constant over time. Before the addition drift
# the observed process ran to 6,506 distinct variants over 91,546 cases; after
# it, 2,563 over 79,998 — a 61 per cent fall with the same workforce. The
# generator treats that as a property of the period, not a constant.
DEVIATION = {"EARLY": 1.30, "PRE": 1.30, "POST": 0.78}

# Between structural drifts a real organisation is not stationary. Working
# practice moves slowly, so the variant distribution of one window differs a
# little from the next even where nothing was reconfigured. Without this the
# baseline of the variant-distance curve sits at half its observed level and
# every peak looks more significant than it is. Modelled as an AR(1) walk on
# the deviation multiplier, with a correlation length in cases.
CHURN_AMPLITUDE = 0.70
CHURN_LENGTH = 800

# The trap. In the observed system these four capabilities were entered in the
# configuration table roughly three years before they produced a single event:
# 1,129, 1,068, 1,068 and 1,018 days respectively, against a horizon of 5,322
# days. The lag is therefore stored as a fraction of the log's own time span,
# so that the trap keeps its proportions at any --cases setting. Pinning it to
# absolute days collapses it to the start of a short log.
DECLARED_LAG_FRAC = {"A11": 0.2121, "A12": 0.2007, "A13": 0.2007, "A14": 0.1913}

N_RESOURCES = 603
RESOURCE_ALPHA = 1.35   # Zipf exponent giving the observed workload skew
UNATTRIBUTED = 0.108    # share of events routed to a role, not a named user
ORIGIN = datetime(2011, 12, 21, 8, 0)
CASES_PER_DAY = 40.0


# ---------------------------------------------------------------------------
def lognormal_from_quartiles(q25, q50, q75):
    """Return a sampler. Degenerate inputs collapse to a constant."""
    if q50 <= 0 and q75 <= 0:
        return lambda rng: 0
    med = max(q50, 0.5)
    spread = max(q75, med + 1) / max(q25, 0.5)
    sigma = max(math.log(spread) / 1.349, 0.1)
    mu = math.log(med)
    return lambda rng: int(min(math.exp(rng.gauss(mu, sigma)), 525600))


WAIT_S = {a: lognormal_from_quartiles(*v) for a, v in WAIT.items()}
HANDLE_S = {a: lognormal_from_quartiles(*v) for a, v in HANDLE.items()}


def resource_pool(rng):
    weights = [1.0 / (i ** RESOURCE_ALPHA) for i in range(1, N_RESOURCES + 1)]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    return cum


def pick_resource(rng, cum):
    if rng.random() < UNATTRIBUTED:
        return ""
    x = rng.random()
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return "R%03d" % (lo + 1)


def build_trace(rng, backbone, scale=1.0):
    """Apply deviation operators to the backbone to produce one variant."""
    seq = list(backbone)

    if rng.random() < P_OPTIONAL_A16 * scale and "A08" in seq:
        seq.insert(seq.index("A08"), "A16")

    # Repeatable, so that a long tail of rare variants forms: three quarters
    # of the distinct variants in the observed log occur exactly once.
    for _ in range(3):
        if rng.random() < P_SKIP * scale and len(seq) > 3:
            seq.pop(rng.randrange(1, len(seq) - 1))
        else:
            break
    for _ in range(3):
        if rng.random() < P_SWAP * scale and len(seq) > 3:
            i = rng.randrange(1, len(seq) - 2)
            seq[i], seq[i + 1] = seq[i + 1], seq[i]
        else:
            break

    out = []
    for a in seq:
        out.append(a)
        p = REWORK.get(a, 0.0) * scale
        while p and rng.random() < p:
            out.append(a)
            p *= 0.35          # repeated rework decays sharply

    if rng.random() < P_TRUNCATE:
        out = out[:rng.choice([1, 2, 2, 3])]
    return out


def emit_case(rng, cum, case_id, backbone, start, scale=1.0):
    """Return the event rows of one case."""
    rows, t = [], start
    for act in build_trace(rng, backbone, scale):
        refer = t
        is_sys = rng.random() < P_SYSTEM.get(act, 0.0)
        res = "" if is_sys else pick_resource(rng, cum)

        if rng.random() < P_OPEN.get(act, 0.0):
            rows.append((case_id, act, refer, res, "schedule", is_sys, None, None))
            t = refer + timedelta(minutes=WAIT_S[act](rng) or 1)
            continue

        if is_sys:
            view = None
            done = refer + timedelta(minutes=max(HANDLE_S[act](rng), 0))
        else:
            view = refer + timedelta(minutes=WAIT_S[act](rng))
            done = view + timedelta(minutes=HANDLE_S[act](rng))

        rows.append((case_id, act, refer, res, "complete", is_sys, view, done))
        t = done + timedelta(minutes=rng.randint(0, 30))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradual", type=int, default=0,
                    help="cases over which the addition drift is phased in; 0 = sudden")
    ap.add_argument("--out", default="synthetic_log")
    ap.add_argument("--cp-replace-frac", type=float, default=0.18)
    ap.add_argument("--cp-add-frac", type=float, default=0.534)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cum = resource_pool(rng)
    n = args.cases

    # Two change points, placed where the observed log places them in
    # proportion: an early replacement, then a later addition.
    cp_replace = int(n * args.cp_replace_frac)
    cp_add = int(n * args.cp_add_frac)

    events, first_seen, t = [], {}, ORIGIN
    cp_times, last_case_start = {}, ORIGIN
    churn_rho = math.exp(-1.0 / CHURN_LENGTH)
    churn_sd = CHURN_AMPLITUDE * math.sqrt(1 - churn_rho ** 2)
    churn = 0.0
    for i in range(n):
        if i < cp_replace:
            backbone, scale = PATH_EARLY, DEVIATION["EARLY"]
        elif i < cp_add:
            backbone, scale = PATH_PRE, DEVIATION["PRE"]
        elif args.gradual and i < cp_add + args.gradual:
            share = (i - cp_add) / args.gradual
            switched = rng.random() < share
            backbone = PATH_POST if switched else PATH_PRE
            scale = DEVIATION["POST"] if switched else DEVIATION["PRE"]
        else:
            backbone, scale = PATH_POST, DEVIATION["POST"]

        if i in (cp_replace, cp_add):
            cp_times[i] = t
        churn = churn_rho * churn + rng.gauss(0, churn_sd)
        last_case_start = t
        rows = emit_case(rng, cum, "C%06d" % i, backbone, t,
                         scale * max(0.15, 1.0 + churn))
        for r in rows:
            first_seen.setdefault(r[1], r[2])
        events.extend(rows)
        t += timedelta(minutes=rng.expovariate(CASES_PER_DAY / 1440.0))

    os.makedirs(args.out, exist_ok=True)

    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "activity", "activity_name", "timestamp",
                    "resource", "lifecycle", "is_system_generated",
                    "view_time", "complete_time"])
        for c, a, refer, res, lc, sysgen, view, done in events:
            w.writerow([c, a, ACTIVITIES[a], refer.isoformat(timespec="seconds"),
                        res, lc, int(sysgen),
                        view.isoformat(timespec="seconds") if view else "",
                        done.isoformat(timespec="seconds") if done else ""])

    xes_path = os.path.join(args.out, "log.xes")
    with open(xes_path, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<log xes.version="1.0">\n')
        fh.write(' <extension name="Concept" prefix="concept" '
                 'uri="http://www.xes-standard.org/concept.xesext"/>\n')
        fh.write(' <extension name="Time" prefix="time" '
                 'uri="http://www.xes-standard.org/time.xesext"/>\n')
        cur = None
        for c, a, refer, res, lc, sysgen, view, done in events:
            if c != cur:
                if cur is not None:
                    fh.write(" </trace>\n")
                fh.write(' <trace>\n  <string key="concept:name" value="%s"/>\n' % c)
                cur = c
            fh.write("  <event>\n")
            fh.write('   <string key="concept:name" value="%s"/>\n' % esc.quoteattr(ACTIVITIES[a])[1:-1])
            fh.write('   <date key="time:timestamp" value="%s"/>\n' % refer.isoformat(timespec="seconds"))
            fh.write('   <string key="org:resource" value="%s"/>\n' % (res or "role"))
            fh.write('   <boolean key="systemGenerated" value="%s"/>\n' % str(bool(sysgen)).lower())
            fh.write("  </event>\n")
        if cur is not None:
            fh.write(" </trace>\n")
        fh.write("</log>\n")

    # The horizon is measured over case arrivals. A handful of acknowledgements
    # sit unopened for months, so the last event is a poor measure of it.
    span_days = max((last_case_start - ORIGIN).days, 1)
    meta_path = os.path.join(args.out, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["activity", "activity_name", "declared_date",
                    "first_event_date", "lag_days"])
        for a in sorted(ACTIVITIES):
            fs = first_seen.get(a)
            if fs is None:
                w.writerow([a, ACTIVITIES[a], "", "", ""])
                continue
            frac = DECLARED_LAG_FRAC.get(a, 0.0)
            decl = max(fs - timedelta(days=frac * span_days), ORIGIN)
            w.writerow([a, ACTIVITIES[a], decl.date().isoformat(),
                        fs.date().isoformat(), (fs.date() - decl.date()).days])
    if all(DECLARED_LAG_FRAC.get(a, 0) == 0 or first_seen.get(a) is None
           for a in ACTIVITIES):
        print("  warning: no metadata lag was written; experiment 2 will be void.")

    gt = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "cases": n,
        "events": len(events),
        "drifts": [
            {"id": "A", "type": "replacement", "case_index": cp_replace,
             "timestamp": cp_times[cp_replace].isoformat(timespec="seconds"),
             "shape": "sudden",
             "description": "two activities cease to be recorded, two begin"},
            {"id": "C", "type": "addition", "case_index": cp_add,
             "timestamp": cp_times[cp_add].isoformat(timespec="seconds"),
             "shape": "gradual" if args.gradual else "sudden",
             "transition_cases": args.gradual,
             "description": "five control and documentation steps inserted; "
                            "the operational backbone is unchanged"},
        ],
        "warning": "metadata.csv declares four of the added activities three "
                   "years before their first event. Evaluating a detector "
                   "against the declared dates rather than these case indices "
                   "reproduces the error this log exists to demonstrate.",
    }
    with open(os.path.join(args.out, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=1)

    # --- validation summary, against the observed aggregates -------------
    seqs = {}
    for c, a, *_ in events:
        seqs.setdefault(c, []).append(a)

    def stats(lo, hi):
        sel = [">".join(s) for c, s in seqs.items() if lo <= int(c[1:]) < hi]
        if not sel:
            return 0, 0, 0.0, 0.0, 0.0
        cnt = {}
        for k in sel:
            cnt[k] = cnt.get(k, 0) + 1
        top = sorted(cnt.values(), reverse=True)
        mean_len = sum(len(s) for c, s in seqs.items() if lo <= int(c[1:]) < hi) / len(sel)
        return (len(sel), len(cnt), 100 * top[0] / len(sel),
                100 * sum(top[:20]) / len(sel), mean_len)

    pre = stats(cp_replace, cp_add)
    post = stats(cp_add + args.gradual, n)
    n_sys = sum(1 for e in events if e[5])

    if args.quiet:
        return
    print("\n  %s" % args.out)
    print("  %d cases, %d events, %.2f events/case, %.1f%% system-generated"
          % (n, len(events), len(events) / n, 100 * n_sys / len(events)))
    print("  change points: case %d (replacement), case %d (addition%s)"
          % (cp_replace, cp_add, ", gradual over %d" % args.gradual if args.gradual else ""))
    print()
    print("  %-22s %>18s %18s" .replace(">", "") % ("", "generated", "observed"))
    rows = [
        ("PRE  cases",            "%d" % pre[0],            "91,546"),
        ("PRE  variants",         "%d" % pre[1],            "6,506"),
        ("PRE  dominant share",   "%.1f%%" % pre[2],        "51.1%"),
        ("PRE  top-20 share",     "%.1f%%" % pre[3],        "74.5%"),
        ("PRE  mean length",      "%.2f" % pre[4],          "9.57"),
        ("POST cases",            "%d" % post[0],           "79,998"),
        ("POST variants",         "%d" % post[1],           "2,563"),
        ("POST dominant share",   "%.1f%%" % post[2],       "64.5%"),
        ("POST top-20 share",     "%.1f%%" % post[3],       "86.7%"),
        ("POST mean length",      "%.2f" % post[4],         "13.63"),
    ]
    for label, g, o in rows:
        print("  %-22s %18s %18s" % (label, g, o))
    print("\n  variant counts scale with case count; the shares are what should match.\n")


if __name__ == "__main__":
    main()
