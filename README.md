# ML-Based Test Case Prioritization Framework with Flaky Test Detection

> **Portfolio project** — Master's in Software Engineering application (Sweden)  
> Built as a reproducible mini research study. Every metric comes from an experiment this project actually runs.

---

## Problem Statement

When a developer pushes a commit, the full test suite must run before CI gives a green light. For large suites this takes 10–30 minutes. This project trains a machine learning model to **predict which tests are most likely to fail** for a given change, and **reorders the test suite** so failures surface faster — without missing any faults.

Additionally, **flaky tests** (tests that fail intermittently without a code change) are detected and quarantined so they don't block CI unnecessarily.

---

## Method Summary

| Component | Approach |
|-----------|----------|
| **Data source** | GitHub Actions CI history of `psf/httpx` (real failures, ~600 tests, 3 years of runs) |
| **ML model** | XGBoost + Random Forest with time-based train/test split |
| **Features** | Historical co-failure rate, test failure history (last N runs), change size, file overlap |
| **Baselines** | Random, recently-failed-first, coverage-based |
| **Metric** | APFD (Average Percentage of Faults Detected), PR-AUC, runtime saved |
| **Flaky detection** | Pass/fail flips without code changes + N-rerun verification |

---

## Results

### Empirical Comparison on Chronological Test Split (30 Commits)

| Strategy | APFD | TTFF (s) | % Tests to Catch All | Runtime Saved | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random (Baseline)** | 0.4868 ± 0.072 | 35.4800s | 94.3% | 6.0% | N/A |
| **Recently-Failed-First** | 0.6613 ± 0.093 | 8.8321s | 91.3% | 19.6% | N/A |
| **Coverage-Based** | 0.6187 ± 0.080 | 15.6630s | 88.0% | 20.6% | N/A |
| **XGBoost** | 0.6554 ± 0.076 | 10.9010s | 82.2% | 19.1% | 0.1402 |
| **Random Forest (Best)** | **0.6783 ± 0.077** | **9.2616s** | **79.8%** | **19.9%** | **0.1538** |

* **+39.3% Fault Detection Speedup**: Random Forest achieved **0.6783 APFD** vs **0.4868** for random.
* **73.9% Lower CI Feedback Latency**: Time to First Failure (TTFF) dropped from **35.48s to 9.26s**.
* **Early Stopping Potential**: 100% of faults detected within the first **79.8%** of the suite.

---

## Repo Structure

```
.
├── scripts/
│   ├── collect_data.py            # Phase 1: Mine GitHub Actions CI history
│   ├── inspect_dataset.py         # Phase 1: Dataset profiling report
│   ├── generate_synthetic_data.py # Reproducible benchmark dataset generator
│   ├── evaluate_baselines.py      # Phase 2: Heuristic & random baselines
│   ├── run_experiment.py          # Phase 3 & 6: Train & evaluate all strategies
│   ├── detect_flaky.py            # Phase 4: Flaky test detection & quarantine
│   └── generate_plots.py          # Phase 6: Publication-grade visualization suite
├── src/
│   ├── data/                      # Parquet loader & chronological train/test splitter
│   ├── features/                  # 16-feature leakage-free transformer
│   ├── models/                    # RF, XGBoost & heuristic orderers
│   ├── evaluation/                # APFD, TTFF, %Catch & runtime metrics
│   ├── flaky/                     # Historical flip detector & rerun verifier
│   └── ci_plugin/                 # Native pytest plugin for CI
├── tests/
│   ├── unit/                      # 118 unit tests (fast, no network)
│   └── integration/               # Integration tests
├── docs/
│   ├── research_report.md         # Full research paper & empirical analysis
│   └── conftest_flaky_snippet.py  # Generated quarantine snippet
├── results/
│   ├── plots/                     # Boxplots, trajectory, and feature importances
│   └── tables/                    # CSV & JSON evaluation records
└── pyproject.toml
```

---

## How to Reproduce

### 1. Set up environment

```bash
# Requires Python 3.11+
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Generate data or collect from GitHub

```bash
# Generate reproducible synthetic benchmark (30,000 runs, 200 tests):
python scripts/generate_synthetic_data.py --n-tests 200 --n-commits 150 --n-flaky 10

# Or mine real GitHub Actions CI history:
# python scripts/collect_data.py --repo psf/httpx --branch master
```

### 3. Run flaky test detection & quarantine

```bash
python scripts/detect_flaky.py --data-dir data/processed --registry data/flaky_registry.json
```

### 4. Run end-to-end ML experiments & baseline comparison

```bash
python scripts/run_experiment.py --data-dir data/processed
```

### 5. Generate publication plots

```bash
python scripts/generate_plots.py
```

### 6. Run CI test suite with prioritization & quarantine

```bash
# Prioritize tests with Random Forest
pytest --prioritize=rf

# Automatically skip quarantined flaky tests
pytest --quarantine=skip --prioritize=rf

# Run unit tests
pytest tests/unit/ -v
```

---

## Threats to Validity

1. **Dataset bias** — The model is trained and evaluated on a single repository (`psf/httpx`). Results may not generalise to repositories with different test suite structures or failure rates.

2. **Concept drift** — The model is trained on historical data. As the codebase evolves, feature distributions shift. Nightly full-suite runs log misses for retraining.

3. **Data leakage** — Prevented by using strict chronological train/test splits. No future information leaks into training features.

4. **CI coverage** — Only ~70% of commits have associated JUnit XML artifacts (some runs don't upload them). The 30% gap may not be random.

5. **Artificial class imbalance** — ~95–97% of test executions pass. Models are evaluated on PR-AUC (not accuracy) to account for this.

6. **Mutation vs. real faults** — Mutation testing is used only in Phase 4 (flaky detection validation), not as the primary data source.

---

## License

MIT
