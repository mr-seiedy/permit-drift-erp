# A calibrated generator for process concept drift benchmarks

This repository accompanies the paper *Detections Outnumber Drifts: Why
Benchmark F1 Does Not Transfer to Industrial Event Logs*.

It contains three things: a generator for event logs with known concept drifts,
calibrated against fifteen years of workflow data from an industrial ERP system;
the logs it produced for the paper, with their exact ground truth; and the
scripts that produced every number in the paper, including the re-analysis of
the published benchmark results.

Everything here runs on the Python standard library except where a step needs
the reference drift-detection implementations, which are noted below.

---

## The claim, in one command

The paper's central observation is that precision on a drift-detection benchmark
cannot exceed *T/k*, the ratio of true change points in the log to change points
the algorithm returns, and that on published benchmark logs the two are almost
the same number. That is checked against published data, not ours:

```
python analysis/analyse_reference_results.py --cdrift /path/to/cdrift-evaluation
```

It reads `algorithm_results.csv`, the complete published output of the reference
evaluation of Adams and colleagues, re-scores all 39,713 results with that
evaluation's own assignment function, and prints the seven tables of Section 1
of the paper. It runs in under a minute and needs no detector to be installed.

Clone the reference evaluation from
`https://github.com/cpitsch/cdrift-evaluation` and point `--cdrift` at it.

---

## The generator

```
python generator/gen_erp_drift.py --cases 20000 --seed 7 --out mylog
```

Writes four files:

| File | Contents |
|---|---|
| `log.csv` | one row per event, with `is_system_generated` at event level |
| `log.xes` | the same log in XES |
| `metadata.csv` | a configuration table declaring four activities long before their first event |
| `ground_truth.json` | the exact case index and timestamp of each change point |

Useful options:

```
--cases N            number of cases (default 20,000)
--seed N             random seed; output is deterministic given the seed
--gradual N          spread the addition drift over N cases (default 0, sudden)
--cp-replace-frac F  position of the replacement drift (default 0.18)
--cp-add-frac F      position of the addition drift (default 0.534)
```

Running it without `--quiet` prints the calibration table: the generated
variant concentration and mean case length against the values observed in the
source system.

### What it models that other generators do not

Three properties of the observed industrial system, each of which changes what
an evaluation concludes. Sections 1.5 and 5 of the paper give the evidence.

**Declaration is not production.** `metadata.csv` records the date each activity
was *defined*, not the date it began producing events. Four activities are
declared roughly twenty per cent of the log's horizon before their first event,
which is the proportion observed in the source system. Anyone annotating ground
truth from that table, rather than from `ground_truth.json`, will place the
change point far too early, and the size of that error is measurable.

**Automation is a property of events, not activities.** Three activities are
closed without ever being opened in 71.6, 80.4 and 86.4 per cent of their
events, and by a person in the rest. The `is_system_generated` column is per
event. Removing those activities wholesale, which is common practice, discards
real human work along with the automated kind.

**A process is not stationary between its drifts.** The deviation rate follows a
slow autoregressive walk, calibrated so that the baseline of the variant-distance
curve matches the observed one. Without it the baseline sits at half the observed
level, and Section 6.4 of the paper reports a finding that this artificially
clean baseline produced and that did not survive correction.

---

## Released logs

Under `logs/`, gzipped. Each folder holds the four files above.

| Folder | Cases | Drift shape |
|---|---|---|
| `n1000` | 1,000 | sudden |
| `n6000` | 6,000 | sudden |
| `n20000` | 20,000 | sudden |
| `n20000_gradual1000` | 20,000 | addition drift spread over 1,000 cases |

The 60,000-case log used in Section 6.1 is not included because of its size.
Reproduce it exactly with:

```
python generator/gen_erp_drift.py --cases 60000 --seed 7 --out logs/n60000
```

The generator is deterministic given a seed, so this reproduces the log used in
the paper byte for byte on any platform with the same Python version.

---

## Reproducing the experiments

### The corpus analysis (Section 1)

No detectors needed. See the first section above.

### The four experiments (Section 6)

These run the seven detection algorithms, so they need the reference
implementations and their dependencies:

```
git clone https://github.com/cpitsch/cdrift-evaluation
pip install POT strsimpy ruptures hotelling pulp scikit-learn deprecation pyarrow pm4py

python analysis/run_experiments.py --cdrift /path/to/cdrift-evaluation
python analysis/run_experiments.py --cdrift /path/to/cdrift-evaluation --only 3
```

Experiment 3 needs no detector and finishes in seconds; the others take some
hours at the larger log sizes. Results are written to `experiments/results.csv`.

### The variant-distance curve (Section 5.3)

```
python generator/tvd_curve.py --input logs/n20000/log.csv --sweep
```

Prints the peak, mean and ratio of the curve at four window settings and writes
the curve itself for plotting.

### Calibration from a source log (Section 5.1)

`analysis/calibrate.py` is the script that produced the generator's parameters
from the industrial log. The log itself cannot be released: it records the
internal workflow of a power generation company and permission to analyse it did
not extend to publishing it. The script is included so that the derivation is
inspectable and so that the same calibration can be performed on another system.
It emits aggregate statistics only: no case identifier, no user identifier, no
timestamp and no free text leave the source.

---

## Notes on dependencies

The reference implementations depend on `wasserstein`, which requires a C++
toolchain on Windows. `POT` is a drop-in substitute for the one function used,
and the analysis scripts install a shim automatically; the two agree to within
1e-17. `scipy.stats.wasserstein_distance` is *not* a substitute, because the
earth mover's distance here is solved against a Levenshtein cost matrix, which
SciPy does not accept.

With NumPy 2.x, `np.NaN` no longer exists and the reference code uses it. The
scripts patch it before importing.

---

## Citation

Details will be added on publication.

## Licence

MIT. See `LICENSE`.
