"""
Phase 1 — Data Collection Script
==================================
Mines commit history + CI test outcomes from a GitHub repository that
uploads JUnit XML artifacts from its GitHub Actions workflow.

Usage:
    python scripts/collect_data.py \
        --repo psf/httpx \
        --token $GITHUB_TOKEN \
        --since 2021-01-01 \
        --until 2024-12-31 \
        --history-window 10 \
        --out-dir data/raw

Output:
    data/raw/test_executions.parquet   — main dataset (one row per test × commit)
    data/raw/dataset_summary.json      — size, imbalance, limitations
    data/raw/commits.parquet           — commit-level metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


class GitHubClient:
    """Thin wrapper around the GitHub REST API with rate-limit back-off."""

    BASE = "https://api.github.com"

    def __init__(self, token: str | None = None, per_page: int = 100) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.per_page = per_page

    # ------------------------------------------------------------------
    def _get(self, url: str, params: dict | None = None, stream: bool = False) -> requests.Response:
        """GET with automatic rate-limit retry."""
        while True:
            resp = self.session.get(url, params=params, stream=stream, timeout=30)
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset_ts - int(time.time()), 5)
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 202:
                # Artifact not ready yet
                time.sleep(5)
                continue
            resp.raise_for_status()
            return resp

    def paginate(self, endpoint: str, params: dict | None = None):
        """Yield all items across paginated GitHub list endpoints."""
        params = {**(params or {}), "per_page": self.per_page, "page": 1}
        while True:
            resp = self._get(f"{self.BASE}{endpoint}", params=params)
            items = resp.json()
            if not items:
                break
            yield from items
            if "next" not in resp.links:
                break
            params["page"] += 1

    def get_commits(
        self,
        repo: str,
        since: str,
        until: str,
        branch: str = "main",
    ) -> list[dict]:
        """Return list of commit objects in chronological order (oldest first)."""
        log.info("Fetching commits for %s [%s → %s]", repo, since, until)
        commits = list(
            self.paginate(
                f"/repos/{repo}/commits",
                params={"sha": branch, "since": since, "until": until},
            )
        )
        commits.reverse()  # oldest first
        log.info("  → %d commits found", len(commits))
        return commits

    def get_commit_detail(self, repo: str, sha: str) -> dict:
        """Fetch per-file diff stats for a single commit."""
        return self._get(f"{self.BASE}/repos/{repo}/commits/{sha}").json()

    def get_workflow_runs(
        self,
        repo: str,
        since: str,
        until: str,
        branch: str = "main",
        status: str = "completed",
    ) -> list[dict]:
        """Return all completed workflow runs in the date window."""
        log.info("Fetching workflow runs for %s", repo)
        runs = list(
            self.paginate(
                f"/repos/{repo}/actions/runs",
                params={
                    "branch": branch,
                    "status": status,
                    "created": f"{since}..{until}",
                },
            )
        )
        log.info("  → %d workflow runs found", len(runs))
        return runs

    def get_run_artifacts(self, repo: str, run_id: int) -> list[dict]:
        """List artifacts attached to a workflow run."""
        return list(self.paginate(f"/repos/{repo}/actions/runs/{run_id}/artifacts"))

    def download_artifact_zip(self, repo: str, artifact_id: int) -> bytes | None:
        """Download a workflow artifact zip. Returns None on failure."""
        url = f"{self.BASE}/repos/{repo}/actions/artifacts/{artifact_id}/zip"
        try:
            resp = self._get(url, stream=True)
            return resp.content
        except Exception as exc:
            log.debug("Could not download artifact %d: %s", artifact_id, exc)
            return None


# ---------------------------------------------------------------------------
# JUnit XML parser
# ---------------------------------------------------------------------------


def parse_junit_xml(xml_bytes: bytes) -> list[dict]:
    """
    Parse a JUnit XML file and return a list of test-result dicts.

    Each dict has:
        test_id       — fully-qualified test name (class::method)
        outcome       — 0 (pass) | 1 (fail/error) | 2 (skip)
        duration_s    — float seconds
        error_msg     — string or None
    """
    results = []
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        log.debug("XML parse error: %s", exc)
        return results

    # Handle both <testsuite> and <testsuites> roots
    suites = root.findall(".//testsuite") or [root]
    for suite in suites:
        for case in suite.findall("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            test_id = f"{classname}::{name}" if classname else name
            duration = float(case.get("time", 0.0))

            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")

            if failure is not None:
                outcome = 1
                error_msg = (failure.get("message") or "")[:500]
            elif error is not None:
                outcome = 1
                error_msg = (error.get("message") or "")[:500]
            elif skipped is not None:
                outcome = 2
                error_msg = None
            else:
                outcome = 0
                error_msg = None

            results.append(
                {
                    "test_id": test_id,
                    "outcome": outcome,
                    "duration_s": duration,
                    "error_msg": error_msg,
                }
            )
    return results


def extract_junit_from_zip(zip_bytes: bytes) -> list[dict]:
    """Extract all JUnit XML files from a zip artifact and parse them."""
    results = []
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith(".xml"):
                    with zf.open(name) as f:
                        results.extend(parse_junit_xml(f.read()))
    except zipfile.BadZipFile:
        log.debug("Bad zip file")
    return results


# ---------------------------------------------------------------------------
# Commit feature extraction
# ---------------------------------------------------------------------------


def extract_commit_features(commit_detail: dict) -> dict:
    """
    Extract change-level features from a GitHub commit detail response.

    Returns a dict with keys:
        files_changed, lines_added, lines_deleted,
        test_files_changed, src_files_changed,
        changed_file_list  (list of file paths)
    """
    files = commit_detail.get("files", [])
    stats = commit_detail.get("stats", {})

    changed_files = [f["filename"] for f in files]
    test_files = [f for f in changed_files if "test" in f.lower()]
    src_files = [f for f in changed_files if "test" not in f.lower() and f.endswith(".py")]

    return {
        "files_changed": len(changed_files),
        "lines_added": stats.get("additions", 0),
        "lines_deleted": stats.get("deletions", 0),
        "test_files_changed": len(test_files),
        "src_files_changed": len(src_files),
        "changed_file_list": json.dumps(changed_files),  # serialisable
    }


# ---------------------------------------------------------------------------
# History window builder
# ---------------------------------------------------------------------------


def build_history_window(
    df: pd.DataFrame,
    history_window: int = 10,
) -> pd.DataFrame:
    """
    For each (test_id, commit) row, add N columns prev_1 … prev_N
    containing the outcomes of the previous N executions of that test
    (NaN if not enough history).

    Assumes df is sorted by (test_id, commit_ts) ascending.
    """
    df = df.sort_values(["test_id", "commit_ts"]).copy()

    for lag in range(1, history_window + 1):
        df[f"prev_{lag}"] = df.groupby("test_id")["outcome"].shift(lag)

    return df


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def collect(
    repo: str,
    token: str | None,
    since: str,
    until: str,
    branch: str,
    history_window: int,
    max_commits: int | None,
    out_dir: Path,
    junit_artifact_name: str,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    client = GitHubClient(token=token)

    # ------------------------------------------------------------------
    # Step 1: Fetch commits
    # ------------------------------------------------------------------
    raw_commits = client.get_commits(repo, since, until, branch)
    if max_commits:
        raw_commits = raw_commits[:max_commits]

    commits_rows = []
    log.info("Fetching per-commit diff stats (this may take a while)…")
    for c in tqdm(raw_commits, desc="commits"):
        sha = c["sha"]
        author = (c.get("commit", {}).get("author") or {}).get("name", "unknown")
        ts = (c.get("commit", {}).get("author") or {}).get("date", "")
        try:
            detail = client.get_commit_detail(repo, sha)
            feats = extract_commit_features(detail)
        except Exception as exc:
            log.debug("Skipping commit %s: %s", sha[:8], exc)
            feats = {
                "files_changed": 0,
                "lines_added": 0,
                "lines_deleted": 0,
                "test_files_changed": 0,
                "src_files_changed": 0,
                "changed_file_list": "[]",
            }
        commits_rows.append(
            {
                "commit_sha": sha,
                "commit_ts": ts,
                "author": author,
                **feats,
            }
        )
        time.sleep(0.05)  # be polite to the API

    commits_df = pd.DataFrame(commits_rows)
    commits_df["commit_ts"] = pd.to_datetime(commits_df["commit_ts"], utc=True)
    commits_df.to_parquet(out_dir / "commits.parquet", index=False)
    log.info("Saved %d commits → %s", len(commits_df), out_dir / "commits.parquet")

    # ------------------------------------------------------------------
    # Step 2: Map workflow runs → commits
    # ------------------------------------------------------------------
    log.info("Fetching workflow runs…")
    all_runs = client.get_workflow_runs(repo, since, until, branch)

    # Build sha → run mapping (keep only the first run per sha,
    # to avoid retries polluting the outcome)
    sha_to_run: dict[str, dict] = {}
    for run in all_runs:
        sha = run.get("head_sha", "")
        if sha and sha not in sha_to_run:
            sha_to_run[sha] = run

    log.info("  → %d unique commits have workflow runs", len(sha_to_run))

    # ------------------------------------------------------------------
    # Step 3: Download JUnit XML artifacts and parse test outcomes
    # ------------------------------------------------------------------
    execution_rows = []
    shas_with_results = 0

    log.info("Downloading JUnit XML artifacts…")
    for commit_row in tqdm(commits_rows, desc="artifact download"):
        sha = commit_row["commit_sha"]
        run = sha_to_run.get(sha)
        if not run:
            continue  # no CI run for this commit

        run_id = run["id"]
        artifacts = client.get_run_artifacts(repo, run_id)

        junit_artifacts = [
            a
            for a in artifacts
            if junit_artifact_name.lower() in a["name"].lower()
            or a["name"].lower().endswith("xml")
            or "test" in a["name"].lower()
        ]

        if not junit_artifacts:
            continue

        test_results: list[dict] = []
        for art in junit_artifacts:
            zip_bytes = client.download_artifact_zip(repo, art["id"])
            if zip_bytes:
                test_results.extend(extract_junit_from_zip(zip_bytes))

        if not test_results:
            continue

        shas_with_results += 1
        for tr in test_results:
            execution_rows.append(
                {
                    "commit_sha": sha,
                    "commit_ts": commit_row["commit_ts"],
                    "author": commit_row["author"],
                    "files_changed": commit_row["files_changed"],
                    "lines_added": commit_row["lines_added"],
                    "lines_deleted": commit_row["lines_deleted"],
                    "test_files_changed": commit_row["test_files_changed"],
                    "src_files_changed": commit_row["src_files_changed"],
                    "changed_file_list": commit_row["changed_file_list"],
                    "test_id": tr["test_id"],
                    "outcome": tr["outcome"],
                    "duration_s": tr["duration_s"],
                    "error_msg": tr["error_msg"],
                    "workflow_run_id": run_id,
                    "workflow_conclusion": run.get("conclusion", ""),
                }
            )

        time.sleep(0.1)

    log.info(
        "Collected %d test-execution rows from %d commits (out of %d total)",
        len(execution_rows),
        shas_with_results,
        len(commits_rows),
    )

    if not execution_rows:
        log.warning(
            "No test execution rows collected. "
            "This usually means the repository does not upload JUnit XML artifacts. "
            "See docs/data_collection_troubleshooting.md for alternatives."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Build history window
    # ------------------------------------------------------------------
    exec_df = pd.DataFrame(execution_rows)
    exec_df["commit_ts"] = pd.to_datetime(exec_df["commit_ts"], utc=True)
    exec_df = build_history_window(exec_df, history_window=history_window)

    # Filter skipped tests (outcome==2) — they carry no failure signal
    exec_df = exec_df[exec_df["outcome"] != 2].copy()
    exec_df["outcome"] = exec_df["outcome"].astype(int)

    exec_df.to_parquet(out_dir / "test_executions.parquet", index=False)
    log.info("Saved dataset → %s", out_dir / "test_executions.parquet")

    # ------------------------------------------------------------------
    # Step 5: Dataset summary & class imbalance report
    # ------------------------------------------------------------------
    total_rows = len(exec_df)
    failure_rows = int((exec_df["outcome"] == 1).sum())
    failure_rate = failure_rows / total_rows if total_rows else 0.0
    unique_tests = exec_df["test_id"].nunique()
    unique_commits = exec_df["commit_sha"].nunique()
    date_min = exec_df["commit_ts"].min()
    date_max = exec_df["commit_ts"].max()

    summary = {
        "repo": repo,
        "since": since,
        "until": until,
        "total_rows": total_rows,
        "failure_rows": failure_rows,
        "pass_rows": total_rows - failure_rows,
        "failure_rate": round(failure_rate, 4),
        "unique_tests": unique_tests,
        "unique_commits": unique_commits,
        "commits_with_ci_results": shas_with_results,
        "commits_total": len(commits_rows),
        "ci_coverage_pct": round(shas_with_results / len(commits_rows) * 100, 1)
        if commits_rows
        else 0,
        "date_min": str(date_min),
        "date_max": str(date_max),
        "history_window": history_window,
        "class_imbalance_ratio": round((total_rows - failure_rows) / failure_rows, 1)
        if failure_rows
        else float("inf"),
        "known_limitations": [
            "Only commits with JUnit XML artifacts are included (~CI coverage pct above)",
            "Workflow re-runs are excluded (first attempt only) to avoid outcome leakage",
            "GitHub API rate limits mean collection may span multiple sessions",
            "Skipped tests are excluded (no failure signal)",
            "Author feature may introduce bias if team is small",
            "Dataset is from a single repository — generalisation is limited",
        ],
    }

    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable report
    log.info("\n" + "=" * 60)
    log.info("DATASET SUMMARY")
    log.info("=" * 60)
    log.info("  Total rows (test executions) : %d", total_rows)
    log.info("  Failures                     : %d (%.1f%%)", failure_rows, failure_rate * 100)
    log.info("  Unique tests                 : %d", unique_tests)
    log.info("  Unique commits               : %d", unique_commits)
    log.info("  CI coverage                  : %.1f%%", summary["ci_coverage_pct"])
    log.info("  Date range                   : %s → %s", date_min, date_max)
    log.info("  Class imbalance ratio (N:P)  : %.1f : 1", summary["class_imbalance_ratio"])
    log.info("=" * 60)
    log.info("Saved summary → %s", out_dir / "dataset_summary.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect CI test-execution data from a GitHub repository.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", default="psf/httpx", help="GitHub owner/repo")
    parser.add_argument("--token", default=None, help="GitHub PAT (or set GITHUB_TOKEN env var)")
    parser.add_argument("--branch", default="master", help="Branch to analyse")
    parser.add_argument("--since", default="2021-01-01", help="Start date (ISO 8601)")
    parser.add_argument("--until", default="2024-12-31", help="End date (ISO 8601)")
    parser.add_argument(
        "--history-window",
        type=int,
        default=10,
        help="Number of previous outcomes to include as features",
    )
    parser.add_argument(
        "--max-commits", type=int, default=None, help="Cap commits (useful for a quick smoke test)"
    )
    parser.add_argument("--out-dir", default="data/raw", help="Output directory")
    parser.add_argument(
        "--junit-artifact-name",
        default="junit",
        help="Substring to match when identifying JUnit XML artifacts",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Allow token from env var
    import os

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        log.warning(
            "No GitHub token provided. You will hit the 60 req/hr unauthenticated limit quickly. "
            "Set GITHUB_TOKEN or pass --token."
        )

    collect(
        repo=args.repo,
        token=token,
        since=args.since,
        until=args.until,
        branch=args.branch,
        history_window=args.history_window,
        max_commits=args.max_commits,
        out_dir=Path(args.out_dir),
        junit_artifact_name=args.junit_artifact_name,
    )


if __name__ == "__main__":
    main()
