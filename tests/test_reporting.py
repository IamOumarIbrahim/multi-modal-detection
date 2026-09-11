"""Tests for results table filler, human confirmation gate, and README immutability."""

import hashlib
import shutil
from pathlib import Path
import pytest
from mmsar.reporting.fill_readme_tables import fill_readme_tables


REAL_README_PATH = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture(scope="module")
def real_readme_hash() -> str:
    """Record initial SHA256 hash of real README.md to ensure it is byte-identical."""
    assert REAL_README_PATH.exists(), f"Real README.md not found at {REAL_README_PATH}"
    return hashlib.sha256(REAL_README_PATH.read_bytes()).hexdigest()


def test_refuse_without_confirmation(tmp_path: Path) -> None:
    # Copy real README.md to tmp_path
    copy_path = tmp_path / "README.md"
    shutil.copyfile(REAL_README_PATH, copy_path)
    initial_bytes = copy_path.read_bytes()

    # Calling without confirmation must raise PermissionError and leave file untouched
    with pytest.raises(PermissionError, match="confirm=True is required"):
        fill_readme_tables(
            readme_path=copy_path,
            metrics={"yolo11n::rgb stream ($c_{\\text{rgb}})::precision": "0.85"},
            confirm=False,
            dry_run=False,
        )

    assert copy_path.read_bytes() == initial_bytes, (
        "File was unexpectedly modified when confirm=False"
    )


def test_dry_run_produces_diff_without_writing(tmp_path: Path) -> None:
    copy_path = tmp_path / "README.md"
    shutil.copyfile(REAL_README_PATH, copy_path)
    initial_bytes = copy_path.read_bytes()

    metrics = {
        "yolo11n::rgb stream ($c_{\\text{rgb}})::precision": "0.915"
    }

    diff = fill_readme_tables(
        readme_path=copy_path,
        metrics=metrics,
        confirm=False,
        dry_run=True,
    )

    assert diff, "Dry-run should produce a non-empty diff string"
    assert "0.915" in diff, "Diff should contain the replacement value"
    assert copy_path.read_bytes() == initial_bytes, (
        "File was unexpectedly modified during dry-run"
    )


def test_fill_confirmed_replaces_exact_cells(tmp_path: Path) -> None:
    copy_path = tmp_path / "README.md"
    shutil.copyfile(REAL_README_PATH, copy_path)

    metrics = {
        "yolo11n::rgb stream ($c_{\\text{rgb}})::precision": "0.924",
        "baseline::desert false alarms": "14",
    }

    diff = fill_readme_tables(
        readme_path=copy_path,
        metrics=metrics,
        confirm=True,
        dry_run=False,
    )

    assert diff, "Diff should be non-empty"
    updated_text = copy_path.read_text(encoding="utf-8")

    assert "0.924" in updated_text, "Cell value 0.924 not found in updated table"
    assert "14" in updated_text, "Cell value 14 not found in updated table"

    # Verify everything outside of results section is byte-identical
    original_text = REAL_README_PATH.read_text(encoding="utf-8")
    orig_before, orig_after = (
        original_text.split("### Results")[0],
        original_text.split("## Quick Reproduction")[1],
    )
    upd_before, upd_after = (
        updated_text.split("### Results")[0],
        updated_text.split("## Quick Reproduction")[1],
    )

    assert orig_before == upd_before, "Content before Results was modified"
    assert orig_after == upd_after, "Content after Results was modified"


def test_partial_fill_with_yolo26n_skipped(tmp_path: Path) -> None:
    """When YOLO26n is skipped due to time constraints, YOLO11n results fill cleanly and YOLO26n stays TBD."""
    copy_path = tmp_path / "README.md"
    shutil.copyfile(REAL_README_PATH, copy_path)

    # Metrics for YOLO11n only (YOLO26n omitted)
    metrics_primary_only = {
        "yolo11n::rgb stream::precision": "0.910",
        "yolo11n::baseline::frame precision": "0.895",
    }

    diff = fill_readme_tables(
        readme_path=copy_path,
        metrics=metrics_primary_only,
        confirm=True,
        dry_run=False,
    )
    assert diff, "Expected non-empty diff for partial fill"
    updated_text = copy_path.read_text(encoding="utf-8")

    # Primary model values filled
    assert "0.910" in updated_text
    assert "0.895" in updated_text

    # YOLO26n rows still retain TBD
    lines = updated_text.splitlines()
    yolo26_table1_line = next(l for l in lines if "YOLO26n" in l and "RGB Stream" in l)
    assert "TBD" in yolo26_table1_line, "YOLO26n row should have remained TBD when skipped"

    yolo26_table2_line = next(l for l in lines if "YOLO26n" in l and "Baseline" in l)
    assert "TBD" in yolo26_table2_line, "YOLO26n baseline should have remained TBD when skipped"


def test_full_fill_with_both_models(tmp_path: Path) -> None:
    """When time permits running both models, both YOLO11n and YOLO26n results are populated."""
    copy_path = tmp_path / "README.md"
    shutil.copyfile(REAL_README_PATH, copy_path)

    metrics_both = {
        "yolo11n::rgb stream::precision": "0.910",
        "yolo26n::rgb stream::precision": "0.935",
        "yolo11n::baseline::frame precision": "0.895",
        "yolo26n::baseline::frame precision": "0.920",
    }

    diff = fill_readme_tables(
        readme_path=copy_path,
        metrics=metrics_both,
        confirm=True,
        dry_run=False,
    )
    assert diff, "Expected non-empty diff for full fill"
    updated_text = copy_path.read_text(encoding="utf-8")

    # Both model values filled
    assert "0.910" in updated_text
    assert "0.935" in updated_text
    assert "0.895" in updated_text
    assert "0.920" in updated_text

    # Verify values are placed in respective lines
    lines = updated_text.splitlines()
    yolo11_line = next(l for l in lines if "YOLO11n" in l and "RGB Stream" in l)
    assert "0.910" in yolo11_line

    yolo26_line = next(l for l in lines if "YOLO26n" in l and "RGB Stream" in l)
    assert "0.935" in yolo26_line


def test_real_repo_readme_is_byte_identical(real_readme_hash: str) -> None:
    """Assert real repository's README.md is byte-identical before and after tests."""
    current_hash = hashlib.sha256(REAL_README_PATH.read_bytes()).hexdigest()
    assert current_hash == real_readme_hash, (
        "CRITICAL: Real repository README.md was modified during the test session!"
    )

