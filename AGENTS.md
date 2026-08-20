# AGENTS.md

Guide for anyone — human or AI agent — working in this repo.

## What this project is

`MechTwin03` is a real-time physical/digital twin built for the Rice Unreal jam.
A webcam computer-vision script (`detect.py`) tracks colored blocks on a
surface and streams their position/rotation/size over OSC to an Unreal Engine
5.8 project (`MT03_RealTimeLayout`), which mirrors them live in-engine.

- `detect.py` — OpenCV tracker, sends OSC to `127.0.0.1:7000`
- `send_test.py` — minimal OSC sender for testing the link without a camera
- `inspect_bp.py` — debug script run *inside* Unreal's embedded Python, for
  probing Blueprint graphs
- `MT03_RealTimeLayout/` — the Unreal project (engine association: 5.8)
- `Video/`, `mt03 outputs/` — raw recordings and edit project files (tracked
  via Git LFS, see `.gitattributes`)

See `README.md` for setup/run instructions and `docs/ROADMAP.md` for where
things are headed.

## How collaboration works here

- **Requesting work from someone (or some agent):** don't ask in passing —
  drop a file in `/tasks` (see `tasks/README.md` for the format). That's the
  single place to check for "what does someone want me to do."
- **"Tell evan ..." / any note addressed to someone who isn't you:** write it
  as a task file in `/tasks`, then **commit and push it immediately, without
  waiting to be asked.** The whole point is that the addressed person sees it
  on their end — a local, unpushed file doesn't do that. This is standing
  authorization to `git add`/`commit`/`push` for `/tasks` changes specifically;
  it does not extend to other files or to force-pushes/history rewrites.
- **Roadmap / direction questions:** check `docs/ROADMAP.md` first before
  asking; update it when priorities shift instead of letting it go stale.
- **How to run anything:** `README.md` is the source of truth. If you had to
  figure out a run step that wasn't documented, add it back to the README.

## Working in this repo

- This repo uses **Git LFS** for large media (`*.mp4`, `*.mov`, and the big
  Unreal `CachedAssetRegistry` cache files). Run `git lfs install` once per
  machine before cloning/pulling, or large files will come through as pointer
  text instead of real content.
- Unreal binary assets (`.uasset`, `.umap`) are not human-diffable — don't try
  to resolve merge conflicts in them by hand; coordinate in `/tasks` instead
  of editing the same asset at the same time.
- No `.gitignore` is in place — Unreal's `Intermediate`/`Saved`/
  `DerivedDataCache` folders are committed as-is for this project. Be aware
  those folders churn on every editor open and will show up as diffs.
- Keep task files and roadmap updates in the same PR as the work they
  describe when practical, so history stays legible.
