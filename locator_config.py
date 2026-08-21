"""
locator_config.py - persisted tuning for lego_locator_xyz.py, so a rig's
calibration (colour-detection sliders, assumed FOV, OSC endpoint, outline
height, settle time) survives between runs instead of resetting to hardcoded
defaults every session.

Addresses docs/PRODUCTION_READINESS.md #4 ("no config persistence").

File: lego_locator_config.json, written next to the script (repo root by
default). NOT meant to be committed - it's a per-rig/per-lighting snapshot,
not shared code (same spirit as the Unreal Saved/Intermediate folders this
repo DOES commit, except this one is genuinely local-only; there's no
.gitignore in this repo per README convention, so this is left untracked by
choice - just don't `git add` it).

Load order: hardcoded DEFAULTS -> file (if present and valid) -> CLI flags
(argparse defaults are seeded from the loaded config, so an explicit CLI
flag always wins over whatever the file says).
"""
import json
import os

DEFAULT_PATH = "lego_locator_config.json"

# Mirrors the hardcoded defaults this replaces: lego_locator.py's
# DEFAULT_S_FLOOR/DEFAULT_V_FLOOR/DEFAULT_MIN_AREA_100 and
# lego_locator_xyz.py's argparse defaults (--fov/--outline-height/
# --osc-host/--osc-port/--settle).
DEFAULTS = {
    "s_floor": 140,
    "v_floor": 70,
    "min_area_100": 8,
    "fov": 60.0,
    "outline_height": 2.0,
    "osc_host": "127.0.0.1",
    "osc_port": 7000,
    "settle": 3.0,
}


def load(path=DEFAULT_PATH):
    """DEFAULTS merged with whatever `path` has, ignoring unknown keys and
    falling back to DEFAULTS entirely on a missing/corrupt/unreadable file -
    a bad config file must never crash the tracker, just be ignored."""
    cfg = dict(DEFAULTS)
    if not os.path.exists(path):
        return cfg
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return cfg
    if isinstance(data, dict):
        for k, v in data.items():
            if k in DEFAULTS:
                cfg[k] = v
    return cfg


def save(cfg, path=DEFAULT_PATH):
    """Write only the known keys (ignores extras the caller's dict might
    carry), atomically (write to a temp file, then rename) so a crash or
    Ctrl-C mid-write can't leave a truncated/corrupt config behind."""
    clean = {k: cfg[k] for k in DEFAULTS if k in cfg}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(clean, f, indent=2)
    os.replace(tmp, path)
    return clean
