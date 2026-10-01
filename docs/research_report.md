# ML-Based Test Case Prioritization with Flaky Test Detection
## An Empirical Study on Accelerating CI Feedback Loops

**Author**: Adithya  
**Target Program**: Master's in Software Engineering (Sweden) / Industry Portfolio  
**Status**: Completed & Fully Reproducible  

---

## Abstract

Continuous Integration (CI) test suites frequently suffer from slow feedback cycles and non-deterministic (flaky) test failures. In this work, we design, implement, and empirically evaluate a complete machine learning-based Test Case Prioritization (TCP) framework combined with automated flaky test detection and quarantine. Using chronological time-series splitting (80/20 train/test) to strictly prevent future information leakage, we engineer 16 commit-level and historical test-execution features (including failure recency, co-failure scoring, and file modification overlap). We benchmark Random Forest and XGBoost classifiers against three established baselines: Random, Recently-Failed-First, and Coverage-Based Prioritization.

Our empirical results demonstrate that **Random Forest prioritization achieves an APFD of 0.6783** and **reduces Time-to-First-Failure (TTFF) by 73.9%** (from 35.48s down to 9.26s) compared to random ordering. Furthermore, our historical flip-without-change detector combined with an $N$-rerun verification strategy accurately isolates flaky tests, saving false-failure build interruptions in CI. Finally, we package the framework into a native `pytest` plugin (`ml-test-prioritization`) for seamless CI integration.

---

## 1. Problem Formulation & Research Questions

1. **RQ1 (Fault Detection Acceleration)**: Can ML-based test prioritization improve the Average Percentage of Faults Detected (APFD) and reduce Time-to-First-Failure (TTFF) compared to heuristic and random baselines?
2. **RQ2 (Feature Importance)**: Which feature categories (historical failure streaks, commit change size, co-change file overlap) are the strongest predictors of test failures?
3. **RQ3 (Flakiness Mitigation)**: How effectively can historical flip analysis quarantine non-deterministic test failures without manual intervention?

---

## 2. Experimental Methodology

### 2.1 Dataset & Leakage Prevention
To evaluate the framework, we simulate and mine realistic CI execution traces with:
- **Chronological Split**: Strict cutoff at commit timestamp (80% train, 20% test). No future execution outcomes leak into feature transformers or test-level priors.
- **Class Imbalance**: Test failure rates are realistically low (~7.2%). Models are evaluated using Precision-Recall AUC (PR-AUC) and cost-sensitive loss weighting (`scale_pos_weight` and `class_weight="balanced"`).

### 2.2 Feature Engineering (16 Features)
The `FeatureEngineer` extracts four feature families:
1. **History & Recency**: `hist_fail_rate_window`, `hist_last_failed`, `hist_consecutive_passes`, `hist_weighted_score` (exponential decay $\alpha=0.7$).
2. **Commit Metadata**: `files_changed`, `lines_added`, `lines_deleted`, `change_size`, `test_files_changed`, `src_files_changed`.
3. **Test Properties**: `duration_s`, `duration_log1p`, `test_global_fail_rate`, `test_is_high_risk`.
4. **Co-Change & Author Context**: `cochange_score` (Jaccard similarity between modified source files and historical failing tests) and `author_fail_rate`.

---

## 3. Empirical Results

### 3.1 Prioritization Performance Summary (Test Split: 30 Commits)

| Strategy | APFD (Mean ± Std) ↑ | TTFF Mean (s) ↓ | % Tests to Catch All ↓ | Runtime Saved ↑ | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random (Baseline)** | 0.4868 ± 0.0719 | 35.4800s | 94.3% | 6.0% | N/A |
| **Recently-Failed-First** | 0.6613 ± 0.0928 | 8.8321s | 91.3% | 19.6% | N/A |
| **Coverage-Based** | 0.6187 ± 0.0803 | 15.6630s | 88.0% | 20.6% | N/A |
| **XGBoost** | 0.6554 ± 0.0758 | 10.9010s | 82.2% | 19.1% | 0.1402 |
| **Random Forest (Best)** | **0.6783 ± 0.0769** | **9.2616s** | **79.8%** | **19.9%** | **0.1538** |

### 3.2 Key Findings
1. **Accelerated Feedback**: Random Forest achieved an **APFD of 0.6783**, improving fault detection speed over random testing by **+39.3%**.
2. **CI Latency Reduction**: Time to first failure dropped from **35.48s to 9.26s**, allowing developers to receive failure signals in less than 30% of standard baseline time.
3. **Early Stopping Potential**: To catch 100% of faults, Random Forest requires executing only **79.8%** of the suite (compared to 94.3% for random), enabling significant CI cost reductions.
4. **Top Predictive Features**: Feature importance analysis identified `cochange_score` (Gini=0.466) and `test_global_fail_rate` (Gini=0.105) as the primary drivers of failure prediction.

---

## 4. Flaky Test Quarantine & Runtime Verification

The framework integrates a two-stage flakiness mitigation pipeline:
1. **Historical Flip-Without-Change Detector**: Computes `flip_rate` (failing then passing without source code modifications) and `fail_rate_no_change`. Tests scoring above threshold $\theta = 0.10$ are stored in `data/flaky_registry.json`.
2. **Quarantine Execution**: Quarantined tests are tagged with `@pytest.mark.flaky` or automatically skipped via `--quarantine=skip`, preventing false build failures.
3. **Runtime $N$-Rerun Verifier**: Failed tests can be dynamically re-run up to $N=3$ times on the same commit commit state to verify whether failure is deterministic or flaky.

---

## 5. Native `pytest` Plugin Integration

The framework provides the `ml_test_prioritization` plugin:

```bash
# Order tests using Random Forest model
pytest --prioritize=rf

# Order tests using Coverage-based heuristic
pytest --prioritize=coverage

# Run non-flaky tests in blocking CI job
pytest -m "not flaky" --prioritize=rf

# Skip quarantined tests automatically
pytest --quarantine=skip --prioritize=rf
```

---

## 6. Threats to Validity

1. **Construct Validity**: APFD measures fault detection assuming all faults have equal severity. In production, business-critical tests might be weighted differently.
2. **Internal Validity**: Strict chronological time splits and leakage-free feature scaling were applied to avoid lookahead bias.
3. **External Validity**: Results evaluated on synthetic datasets mirroring real-world Python library structures (`httpx`). Real-world repositories may experience distinct commit frequencies and failure dynamics.
