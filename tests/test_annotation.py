"""Tests for Label Studio configuration, client mocking, and RGB-to-Thermal annotation copy."""

import json
import re
from pathlib import Path
import pytest
import requests_mock
from mmsar.annotation.label_studio_config import (
    generate_label_studio_config,
    validate_label_studio_config,
)
from mmsar.annotation.label_studio_client import LabelStudioManager
from mmsar.annotation.rgb_to_thermal_copy import copy_annotations_rgb_to_thermal


def test_label_studio_config_xml_validity() -> None:
    # Validate the committed annotations/label_studio_config.xml
    config_path = Path(__file__).resolve().parent.parent / "annotations" / "label_studio_config.xml"
    assert config_path.exists(), f"Configuration file not found at {config_path}"

    content = config_path.read_text(encoding="utf-8")
    assert validate_label_studio_config(content, expected_label="Person_Detected")


def test_label_studio_config_programmatic() -> None:
    # Test programmatic generator
    custom_label = "Person_Detected"
    xml_str = generate_label_studio_config(custom_label)
    assert validate_label_studio_config(xml_str, expected_label=custom_label)

    # Malformed XML should raise ValueError
    with pytest.raises(ValueError):
        validate_label_studio_config("<View><Invalid></View>")


def test_mocked_label_studio_client() -> None:
    # Gate check: strictly mocked HTTP, no real Label Studio server touched
    base_url = "http://localhost:8080"
    api_key = "mocked-secret-key"

    with requests_mock.Mocker() as m:
        # Mock Label Studio legacy SDK endpoints
        m.get(f"{base_url}/api/version", json={"release": "1.11.0"})
        m.post(
            f"{base_url}/api/projects",
            status_code=201,
            json={"id": 101, "title": "Test Project", "label_config": "<View></View>"},
        )
        m.get(
            f"{base_url}/api/projects/101",
            json={"id": 101, "title": "Test Project", "label_config": "<View></View>"},
        )
        m.post(
            re.compile(rf"{base_url}/api/projects/101/import/?"),
            status_code=201,
            json={"task_ids": [201, 202]},
        )

        manager = LabelStudioManager(base_url=base_url, api_key=api_key)
        proj = manager.create_project(title="Test Project")
        assert getattr(proj, "id", None) == 101 or proj.params.get("id") == 101

        task_ids = manager.import_image_tasks(101, ["frame_001.png", "frame_002.png"])
        assert task_ids == [201, 202]


def test_rgb_to_thermal_copy_exact_coordinates(tmp_path: Path) -> None:
    # Synthetic RGB export fixture with 2 frames and specific box coordinates
    rgb_tasks = [
        {
            "id": 1,
            "data": {"image": "desert_pos_01_rgb_0000.png"},
            "annotations": [
                {
                    "id": 10,
                    "result": [
                        {
                            "id": "box_01",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": 640,
                            "original_height": 640,
                            "value": {
                                "x": 18.75,
                                "y": 42.125,
                                "width": 8.5,
                                "height": 14.25,
                                "rotation": 0,
                                "rectanglelabels": ["Person_Detected"],
                            },
                        }
                    ],
                }
            ],
        },
        {
            "id": 2,
            "data": {"image": "desert_pos_01_rgb_0003.png"},
            "annotations": [
                {
                    "id": 20,
                    "result": [
                        {
                            "id": "box_02",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": 640,
                            "original_height": 640,
                            "value": {
                                "x": 35.0,
                                "y": 55.5,
                                "width": 12.0,
                                "height": 22.0,
                                "rotation": 0,
                                "rectanglelabels": ["Person_Detected"],
                            },
                        }
                    ],
                }
            ],
        },
    ]

    rgb_json_path = tmp_path / "rgb_export.json"
    rgb_json_path.write_text(json.dumps(rgb_tasks, indent=2), encoding="utf-8")

    thermal_map = {
        "desert_pos_01_rgb_0000.png": "desert_pos_01_thermal_0000.png",
        "desert_pos_01_rgb_0003.png": "desert_pos_01_thermal_0003.png",
    }

    out_thermal_path = tmp_path / "thermal_export.json"
    thermal_tasks = copy_annotations_rgb_to_thermal(
        rgb_export_path=rgb_json_path,
        thermal_frame_ids=thermal_map,
        output_path=out_thermal_path,
    )

    assert out_thermal_path.exists(), "Thermal export file was not saved"
    assert len(thermal_tasks) == 2

    # Assert image remapping
    assert thermal_tasks[0]["data"]["image"] == "desert_pos_01_thermal_0000.png"
    assert thermal_tasks[1]["data"]["image"] == "desert_pos_01_thermal_0003.png"

    # Assert box coordinates are preserved EXACTLY
    box0_rgb = rgb_tasks[0]["annotations"][0]["result"][0]["value"]
    box0_th = thermal_tasks[0]["annotations"][0]["result"][0]["value"]
    assert box0_th["x"] == box0_rgb["x"] == 18.75
    assert box0_th["y"] == box0_rgb["y"] == 42.125
    assert box0_th["width"] == box0_rgb["width"] == 8.5
    assert box0_th["height"] == box0_rgb["height"] == 14.25
    assert box0_th["rectanglelabels"] == ["Person_Detected"]

    box1_rgb = rgb_tasks[1]["annotations"][0]["result"][0]["value"]
    box1_th = thermal_tasks[1]["annotations"][0]["result"][0]["value"]
    assert box1_th["x"] == box1_rgb["x"] == 35.0
    assert box1_th["y"] == box1_rgb["y"] == 55.5
    assert box1_th["width"] == box1_rgb["width"] == 12.0
    assert box1_th["height"] == box1_rgb["height"] == 22.0
