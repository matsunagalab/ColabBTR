"""Shared fixtures for benchmark tests."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def benchmark_data_dir():
    """Path to prepared benchmark data."""
    path = Path("benchmark_results/data")
    if not path.exists():
        pytest.skip("Benchmark data not found. Run 'python benchmarks/prepare.py' first.")
    return path


@pytest.fixture(scope="session")
def benchmark_output_dir():
    """Path for benchmark result files."""
    path = Path("benchmark_results")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def result_file(benchmark_output_dir):
    """JSONL file path for this test session."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_hash = "unknown"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return benchmark_output_dir / f"{git_hash}_{ts}.jsonl"


@pytest.fixture
def save_result(result_file):
    """Callable to append a result dict to the session JSONL file."""
    def _save(record):
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip()
        except Exception:
            git_hash = "unknown"
        record["git_commit"] = git_hash
        record["timestamp"] = datetime.now().isoformat()
        with open(result_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    return _save
