"""Tests for FHI-aims run_stats parsing."""

import gzip
import shutil
from pathlib import Path

import pytest

from atomate2.aims.schemas.calculation import _parse_run_stats

TEST_DIR = Path(__file__).parents[1] / "test_data" / "aims"


@pytest.fixture
def relax_si_aims_out(tmp_path) -> Path:
    """Decompress the completed relax-si aims.out.gz fixture to a plain-text file."""
    gz_path = TEST_DIR / "relax-si" / "outputs" / "aims.out.gz"
    out_path = tmp_path / "aims.out"
    with gzip.open(gz_path, "rt") as f_in, open(out_path, "w") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def test_parse_run_stats_completed_run(relax_si_aims_out):
    """A completed run should yield fully populated, plausible run_stats."""
    stats = _parse_run_stats(relax_si_aims_out)

    expected_keys = {
        "CPU time (sec)",
        "Elapsed time (sec)",
        "cores",
        "Minimum memory used (kb)",
        "Maximum memory used (kb)",
        "Average memory used (kb)",
    }
    assert set(stats) == expected_keys

    # Timing
    assert stats["CPU time (sec)"] == 68.393
    assert stats["Elapsed time (sec)"] == 68.399

    # Cores
    assert isinstance(stats["cores"], int)
    assert stats["cores"] == 4

    # Memory: min <= avg <= max, all positive
    mem_min = stats["Minimum memory used (kb)"]
    mem_max = stats["Maximum memory used (kb)"]
    mem_avg = stats["Average memory used (kb)"]
    assert mem_min == 18526.0
    assert mem_max == 18888.0
    assert mem_avg == 18712.0
    assert mem_min <= mem_avg <= mem_max


def test_parse_run_stats_missing_file_data(tmp_path):
    """A file with none of the expected patterns should degrade to all None."""
    aims_out = tmp_path / "aims.out"
    aims_out.write_text("nothing useful here\n")

    stats = _parse_run_stats(aims_out)
    assert stats == {
        "CPU time (sec)": None,
        "Elapsed time (sec)": None,
        "cores": None,
        "Minimum memory used (kb)": None,
        "Maximum memory used (kb)": None,
        "Average memory used (kb)": None,
    }


def test_parse_run_stats_incomplete_run(tmp_path):
    """A run killed mid-execution has no timing/memory block; degrade to None."""
    aims_out = tmp_path / "aims.out"
    aims_out.write_text(
        "===================================================================================\n"
        "=   BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES\n"
        "=   RANK 1 PID 66572 RUNNING AT andrey-Latitude-5411\n"
        "=   KILLED BY SIGNAL: 9 (Killed)\n"
        "===================================================================================\n"
    )

    stats = _parse_run_stats(aims_out)
    assert stats["CPU time (sec)"] is None
    assert stats["Elapsed time (sec)"] is None
    assert stats["Minimum memory used (kb)"] is None
    assert stats["Maximum memory used (kb)"] is None
    assert stats["Average memory used (kb)"] is None
