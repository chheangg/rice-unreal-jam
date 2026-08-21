---
status: in-progress # open | in-progress | done
requested-by: ibrahimaasim
assigned-to: evan
date: 2026-08-21
---

## Update 2026-08-21 (Claude, third pass) - C++ implementation written, needs a compile pass

Wrote an actual implementation instead of leaving this as a Blueprint task:
`MT03_RealTimeLayout/Source/MT03_RealTimeLayout/LegoOscSubsystem.h/.cpp`
(+ the `.Target.cs`/`.Build.cs` files and `Modules` entry in the `.uproject`
needed to make this a C++ project at all - it was pure Blueprint before).

It's a `UWorldSubsystem`, so **no Blueprint wiring or manual actor placement
is needed** - Unreal instantiates one automatically per World the moment a
level starts (PIE or packaged). On `Initialize()` it starts an OSC server on
`127.0.0.1:7000` and on each `/outline` message it rebuilds that piece's
mesh via `GeometryScript`'s `AppendExtrudedPolygon`, using the vertices
(already absolute world-space, already carrying true rotation) and the
fixed `height_cm` - same design as the extrude plan below, just as compiled
code instead of a graph.

**This has NOT been compiled or opened in the Editor** - written without
engine access, so treat it as a strong first draft, not verified working.
The most likely things to need a small fix on first compile (each is
flagged inline in the `.cpp` with a "CHECK IF THIS DOESN'T COMPILE" comment
naming the Blueprint-node equivalent to cross-reference): the exact
`UOSCManager::CreateOSCServer` parameter list, the `OnOscMessageReceived`
delegate signature on `UOSCServer`, `FOSCAddress::GetFullPath()`, and
`AppendExtrudedPolygon`'s parameter list. All are real, documented 5.x APIs
- the risk is signature drift between engine versions, not made-up
functions - so a compile error here should be a quick one-line fix, not a
redesign.

**To pick this up:** open the project in 5.8 (first open will prompt to
build the new C++ module - accept it), fix whatever the compiler flags, hit
Play, and confirm pieces show up as their real extruded outline instead of
nothing/a cube. If `BP_OSCreciver` (the pre-existing Blueprint) still
spawns its own placeholder per `/obj`, decide whether to disable/delete it
or leave both running - two receivers reacting to the same OSC stream would
double up actors, so pick one.

## Original task (Blueprint version - superseded above if the C++ path works)

## What

`lego_locator_xyz.py --osc` now sends a second OSC message per confirmed
piece, alongside the existing `/obj`:

```
/outline -> [name, n_points, x1_mm, y1_mm, x2_mm, y2_mm, ..., xn_mm, yn_mm, height_cm]
```

- `name` - same piece name as `/obj` (e.g. `yel1`, `red`)
- `n_points` - how many (x,y) pairs follow (piece contour simplified via
  `cv2.approxPolyDP`, capped at 16 points)
- the `x,y` pairs are the piece's REAL outline vertices, in mm, already in
  floor-frame world space (same frame as `/obj`'s x/y) - they already carry
  the piece's true rotation, so no separate angle needs to be applied
- `height_cm` - a fixed flat extrusion height (currently 2.0cm, see
  `--outline-height` in `lego_locator_xyz.py`), NOT a real scanned height -
  just enough to give the mesh some thickness

Build a Blueprint (or extend the existing OSC receiver) that, on `/outline`,
uses UE5's **Geometry Script** plugin to extrude that 2D polygon into real
mesh (Geometry Script has a built-in "append extrude polygon"-style function
for exactly this - 2D points + a height -> solid mesh at runtime). This
replaces the placeholder cube/block currently spawned per piece with
something that actually looks like the built Lego shape's footprint.

## Why

The team wants a piece placed in front of the camera to show up in Unreal as
its actual outline instead of a generic block, without needing a real 3D
scan/import pipeline (no time for photogrammetry mid-hackathon). A flat
extrusion of the real detected 2D silhouette is a good-enough visual match
for a demo. Full context/back-and-forth on why this approach (vs. a real
scan) is in the chat that spawned this task; short version: `/obj`'s cube
placeholder was always meant to be swappable (see `docs/ROADMAP.md`'s
"Option 1 vs Option 2" note) and this is the cheap version of Option 2.

## Notes

- `/obj` is unchanged - keep using it for position/rotation/size of whatever
  actor represents the piece. `/outline` is purely for building/updating that
  actor's mesh.
- Only confirmed (settled 3s) pieces send `/outline`, same gating as `/obj`.
- If a piece moves, a NEW `/outline` arrives each frame with (very likely)
  the same point count - simplest approach is probably to rebuild the
  Geometry Script mesh each time rather than trying to diff/patch it,
  unless that's visibly too expensive.
- **`/outline` always has >= 3 points now** (as of this session) - a
  degenerate simplification (a sliver contour collapsing to a line/point)
  falls back to the min-area-rect corners instead, and the script skips
  sending `/outline` entirely if it still can't produce a real polygon. You
  should never see `n_points` < 3 on this address.
- **Units - read carefully, this is an easy mistake:** the `x,y` pairs are
  in **millimeters**, but `height_cm` is exactly what its name says -
  **centimeters**. They are NOT the same unit within one message. Unreal's
  default world unit is 1 unit = 1cm, so: divide every `x_mm`/`y_mm` by 10
  to get the actor-space value in Unreal units, then use `height_cm`
  directly (no further conversion) for the extrusion depth. **Do NOT reuse
  the "÷50" scale divisor mentioned elsewhere in `docs/ROADMAP.md`'s
  "Carried-over knowledge" section** - that number is from the OLDER,
  unrelated `detect.py` pipeline, which sends raw PIXEL coordinates (not
  metric), and ÷50 was an arbitrary fit for pixel-space, not a unit
  conversion. `/obj` and `/outline` from `lego_locator_xyz.py` are real
  metric millimeters/centimeters - the correct, exact conversion is ÷10 for
  mm, not ÷50 or any other pixel-era constant. If `/obj`'s receiver
  Blueprint is currently using ÷50 on `lego_locator_xyz.py`'s `/obj`
  output, that's very likely ALSO wrong for the same reason and worth
  checking while you're in there.
- **Floor-frame axis orientation is fixed within a run, not globally
  meaningful:** `/obj`'s and `/outline`'s X/Y live in a floor-relative frame
  whose in-plane axes are derived from the fitted floor plane's normal, not
  from any real-world compass direction - so "floor X" doesn't point any
  particular way in the room, and can differ between separate runs of the
  tracker (though it's now held continuous frame-to-frame WITHIN a run - a
  bug where it could flip 180 degrees mid-run was fixed this session, see
  `docs/PRODUCTION_READINESS.md`). Practical effect: don't hardcode an
  assumption about which screen direction "positive X" corresponds to when
  wiring up the receiver/camera framing - eyeball it once per session
  instead, since it can differ each time the tracker restarts.
- I (Claude) can't test the Unreal side myself - no Unreal Editor access in
  this environment, and this needs a live Editor session. Flagging that up
  front so it's clear this task needs manual verification once picked up.

## Update 2026-08-21 (Claude, second pass) - plugin blocker fixed, implementation sketch, cheap fallback added

`/obj` now also carries `shape` (`square`/`rectangle`/`circle`/`cross`/`?`)
as its 8th field, appended at the end (see README). That's a much cheaper
signal than this task's outline extrude if you want a quick "pick one of 4
pre-made meshes by category" fallback (`docs/ROADMAP.md`'s original "Option
1") while the real extrude pipeline below is still being built/debugged -
worth wiring up first if you want *something* other than a plain cube
on-screen quickly, since it needs no Geometry Script/plugin work at all,
just a Switch-on-String node keyed off index 7.


While re-checking this task I found a concrete blocker and fixed the part of
it that's a plain text edit (not an Editor-GUI action, so within what I can
safely do):

- `MT03_RealTimeLayout.uproject` only had `ModelingToolsEditorMode` enabled
  (editor-only tooling UI) - the actual **`GeometryScripting`** plugin,
  which is what exposes the Blueprint-callable mesh-generation function
  library (including polygon extrude) to a running level/PIE, was NOT
  enabled at all. Without it, the Geometry Script nodes this task needs
  either wouldn't appear in the Blueprint node picker or wouldn't work
  outside the editor's own modeling tools. Added it to the `Plugins` array
  with no `TargetAllowList` restriction (so it's available in both Editor
  and packaged Game, not just the Editor). **This needs the project
  reopened in Unreal once to let it compile/register the plugin** - that
  first-open step still requires a human at the Editor, I can't trigger it.

- Implementation sketch for whoever picks this up (conceptual - I have not
  verified exact node names/pins against a live 5.8 editor, so treat these
  as "search for something like this in the node picker," not copy-paste
  exact signatures):
  1. On the Blueprint actor that currently gets the placeholder cube, add a
     `Dynamic Mesh Component` (or use one already present for the modeling
     tools workflow).
  2. On `/outline` received: build a 2D polygon input from the `x_mm/10,
     y_mm/10` pairs (divide by 10 per the units note below), in order - the
     Geometry Script polygon-extrude family of functions (something like
     "Append Extruded Polygon" under the Geometry Script mesh-primitive
     function library) takes a 2D point list + an extrude height and
     produces a solid mesh. Feed it `height_cm` directly (already in the
     correct unit, see below).
  3. Set the Dynamic Mesh Component's mesh to the result each time a new
     `/outline` arrives for that piece's ID (rebuild-per-message, per the
     note below about not diffing).
  4. Keep `/obj` driving the actor's transform (position/rotation) exactly
     as it does today - `/outline`'s vertices are already rotated into
     world space, so do NOT also apply `/obj`'s angle on top of the
     extruded mesh, or the shape will double-rotate.
  5. One thing worth deciding early: whether the extruded mesh should be
     centered at the actor's own transform or built directly in world
     space and the actor left at identity - since the outline vertices
     already carry absolute floor-frame position, building world-space and
     leaving the actor transform at origin is probably simplest and avoids
     fighting `/obj`'s position update against the mesh's own baked-in
     position. Pick whichever is less fuss once you see it in the editor;
     this isn't testable without one.
