"""
Tests for locator_config.py (config persistence, docs/PRODUCTION_READINESS.md
#4). All file I/O happens in a temp directory - never touches a real
lego_locator_config.json in the repo.
"""
import json
import os

import pytest

import locator_config as cfgmod


def test_load_missing_file_returns_hardcoded_defaults(tmp_path):
    path = str(tmp_path / "nope.json")
    cfg = cfgmod.load(path)
    assert cfg == cfgmod.DEFAULTS
    assert cfg is not cfgmod.DEFAULTS   # must be a copy, not the shared dict


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "cfg.json")
    written = {**cfgmod.DEFAULTS, "s_floor": 200, "fov": 47.5,
              "osc_host": "10.0.0.5"}
    cfgmod.save(written, path)
    loaded = cfgmod.load(path)
    assert loaded == written


def test_save_ignores_unknown_keys(tmp_path):
    path = str(tmp_path / "cfg.json")
    cfgmod.save({**cfgmod.DEFAULTS, "totally_made_up_key": 123}, path)
    with open(path) as f:
        on_disk = json.load(f)
    assert "totally_made_up_key" not in on_disk


def test_load_ignores_unknown_keys_in_file(tmp_path):
    path = str(tmp_path / "cfg.json")
    with open(path, "w") as f:
        json.dump({"s_floor": 99, "some_future_field": "x"}, f)
    cfg = cfgmod.load(path)
    assert cfg["s_floor"] == 99
    assert "some_future_field" not in cfg


def test_load_partial_file_fills_remaining_from_defaults(tmp_path):
    path = str(tmp_path / "cfg.json")
    with open(path, "w") as f:
        json.dump({"osc_port": 9999}, f)
    cfg = cfgmod.load(path)
    assert cfg["osc_port"] == 9999
    assert cfg["fov"] == cfgmod.DEFAULTS["fov"]


def test_load_corrupt_json_falls_back_to_defaults(tmp_path):
    path = str(tmp_path / "cfg.json")
    with open(path, "w") as f:
        f.write("{ not valid json ]")
    cfg = cfgmod.load(path)
    assert cfg == cfgmod.DEFAULTS


def test_load_non_dict_json_falls_back_to_defaults(tmp_path):
    path = str(tmp_path / "cfg.json")
    with open(path, "w") as f:
        json.dump([1, 2, 3], f)
    cfg = cfgmod.load(path)
    assert cfg == cfgmod.DEFAULTS


def test_save_is_atomic_no_tmp_file_left_behind(tmp_path):
    path = str(tmp_path / "cfg.json")
    cfgmod.save(cfgmod.DEFAULTS, path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_save_overwrites_existing_file_cleanly(tmp_path):
    path = str(tmp_path / "cfg.json")
    cfgmod.save({**cfgmod.DEFAULTS, "fov": 10.0}, path)
    cfgmod.save({**cfgmod.DEFAULTS, "fov": 90.0}, path)
    assert cfgmod.load(path)["fov"] == 90.0
