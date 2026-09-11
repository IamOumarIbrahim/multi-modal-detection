"""Tests for dataset manifest schema, storage, and progress tracking."""

from pathlib import Path
import pytest
from mmsar.manifest.schema import Bin, Manifest
from mmsar.manifest.store import load_manifest, save_manifest, DEFAULT_MANIFEST_PATH


def test_real_manifest_seeded_counts() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / DEFAULT_MANIFEST_PATH
    assert manifest_path.exists(), f"Manifest file does not exist at {manifest_path}"

    manifest = load_manifest(manifest_path)

    # Assert desert.positive.count == 5 and desert.hard_negative.count == 3
    assert manifest.get_bin("desert", "positive").count == 5
    assert manifest.get_bin("desert", "hard_negative").count == 3

    # Assert every other bin == 0
    for cond, scenarios in manifest.bins.items():
        for scen, b in scenarios.items():
            if (cond, scen) not in [("desert", "positive"), ("desert", "hard_negative")]:
                assert b.count == 0, f"Expected bin {cond}/{scen} to have count 0, got {b.count}"

    # Check totals
    assert manifest.total_count == 8
    assert manifest.total_target == 45


def test_manifest_roundtrip_save_load(tmp_path: Path) -> None:
    manifest = Manifest.create_empty()
    manifest.get_bin("forest", "positive").count = 2

    temp_file = tmp_path / "manifest_test.json"
    save_manifest(manifest, temp_file)

    loaded = load_manifest(temp_file)
    assert loaded.get_bin("forest", "positive").count == 2
    assert loaded.get_bin("altitude", "clear_negative").count == 0
    assert loaded.total_count == 2
