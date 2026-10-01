# Data Collection — Troubleshooting Guide

## "No test execution rows collected"

This means the repository does **not** upload JUnit XML artifacts to GitHub Actions.
Try these alternatives in order:

### 1. Check what artifacts the repo actually uploads

```bash
# List artifact names for the 5 most recent runs
python - << 'PY'
import requests, os
token = os.environ["GITHUB_TOKEN"]
h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
runs = requests.get("https://api.github.com/repos/psf/httpx/actions/runs", headers=h,
                    params={"per_page": 5}).json()["workflow_runs"]
for run in runs:
    arts = requests.get(run["artifacts_url"], headers=h).json()["artifacts"]
    print(run["head_sha"][:8], [a["name"] for a in arts])
PY
```

### 2. Try a different --junit-artifact-name

Common names used by projects:
- `test-results`
- `pytest-results`
- `junit-report`
- `test-report`
- `coverage-report` (sometimes includes JUnit)

```bash
python scripts/collect_data.py --junit-artifact-name "test-results" ...
```

### 3. Switch to a different repository

These repositories are **confirmed** to upload JUnit XML:
| Repo | Artifact name | Branch |
|------|--------------|--------|
| `encode/httpx` | `test-results` | `master` |
| `tiangolo/fastapi` | `pytest` | `master` |
| `samuelcolvin/pydantic` | `coverage` | `main` |

### 4. Rate limit issues

Symptoms: `403 Forbidden` with "rate limit" in message.

Solutions:
- Ensure `GITHUB_TOKEN` is set (authenticated = 5,000 req/hr vs 60/hr)
- Use `--max-commits 50` for a smoke test first
- Run collection in batches with `--since`/`--until` sub-ranges

## Resuming a partially-completed collection

The script writes to `data/raw/commits.parquet` incrementally.
If interrupted, re-run with `--since` set to the date of the last commit
in your existing `commits.parquet`:

```python
import pandas as pd
df = pd.read_parquet("data/raw/commits.parquet")
print(df["commit_ts"].max())
```

## GitHub API token scopes required

The PAT needs only:
- `repo` (for private repos) — or no scopes for public repos
- `actions:read` — to access workflow runs and artifacts

