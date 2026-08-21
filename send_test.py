"""
send_test.py - fake tracker, for testing the Unreal side without a camera.

Sends the SAME OSC message the real trackers send (`detect.py` /
`detect_xyz.py`), on the same address, at a realistic frame rate - so the
Unreal receiver Blueprint can be built and debugged with no camera, no
lighting rig, and no Lego on the table.

OSC message layout (address /obj), identical to detect_xyz.py:
    [name, x, y, angle, sizeX, sizeY, z]
With --no-z it sends the 6-element form instead, matching detect.py:
    [name, x, y, angle, sizeX, sizeY]

Differences from the real tracker, on purpose:
  - `angle` really rotates 0..359 here. The current colour-only tracker
    always sends 0, but ArUco will send a true angle, so the receiver's
    rotation path needs exercising now.
  - IDs can appear and disappear mid-run (--churn), which is what the
    ID-based receiver has to cope with: spawn on a name it hasn't seen,
    stop updating one that goes quiet.

Run:
    python send_test.py                  # 4 blocks, 30 Hz, circling
    python send_test.py --ids 6          # 6 blocks
    python send_test.py --churn 3        # add/drop a block every 3s
    python send_test.py --no-z           # old 6-element message
    python send_test.py --rate 5         # slow, easier to read in UE logs

### --outline mode: testing ULegoOscSubsystem / the mesh-extrude receiver

The modes above only ever send `/obj`, in pixel coordinates - useful for
the OLD placeholder-cube receiver, but it never exercises `/outline` at
all, which is the only address `ULegoOscSubsystem`
(`MT03_RealTimeLayout/Source/.../LegoOscSubsystem.cpp`) listens to. Use
`--outline` instead to fake `lego_locator_xyz.py`'s real wire format
(metric mm/cm, world-space) and rotating polygons with concave shapes (an
L, a plus/cross) as well as convex ones - so you can confirm the C++
receiver builds and updates real extruded meshes with NO camera, NO Lego
pieces, and NO lighting rig, just Unreal running and this script:

    python send_test.py --outline                 # 4 fake pieces, rotating
    python send_test.py --outline --ids 2 --rate 5 # slower, easier to watch

Ctrl-C to stop.
"""

import argparse
import math
import random
import time

from pythonosc.udp_client import SimpleUDPClient

# Local (unrotated, centered-at-origin) polygon templates for --outline mode,
# in mm - deliberately including concave shapes (L, plus) alongside convex
# ones (square, rectangle), since a convex-only test wouldn't catch a
# receiver-side extrude that silently assumes convexity.
OUTLINE_SHAPES = {
    "square": (
        [(-30, -30), (30, -30), (30, 30), (-30, 30)],
        "square",
    ),
    "rectangle": (
        [(-45, -20), (45, -20), (45, 20), (-45, 20)],
        "rectangle",
    ),
    "L": (
        [(-30, -30), (30, -30), (30, -10), (-10, -10), (-10, 30), (-30, 30)],
        "?",   # classify_shape() has no "L" category - "?" is what the
               # real pipeline would send for one too
    ),
    "plus": (
        [(-10, -30), (10, -30), (10, -10), (30, -10), (30, 10), (10, 10),
         (10, 30), (-10, 30), (-10, 10), (-30, 10), (-30, -10), (-10, -10)],
        "cross",
    ),
}
OUTLINE_HEIGHT_CM = 2.0   # matches lego_locator_xyz.py's --outline-height default


class FakeOutlineBlock:
    """One piece orbiting the floor-frame origin, spinning, with a fixed
    local polygon - mirrors lego_locator_xyz.py's real /obj + /outline
    output (mm/cm, world-space vertices already carrying true rotation)."""

    def __init__(self, name, index, total, shape_name):
        spread = (index + 0.5) / total
        self.name = name
        self.orbit_radius_mm = 150.0 + 40.0 * index
        self.phase = 2 * math.pi * spread
        self.speed = 0.3 + 0.08 * index         # rad/s, desynced per piece
        self.spin_speed = 0.5 + 0.2 * index      # own rotation, separate from orbit
        self.local_poly, self.shape_label = OUTLINE_SHAPES[shape_name]

    def sample(self, t):
        orbit_a = self.phase + self.speed * t
        cx = self.orbit_radius_mm * math.cos(orbit_a)
        cy = self.orbit_radius_mm * math.sin(orbit_a)
        angle_deg = math.degrees(self.spin_speed * t) % 360.0
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        world_poly = []
        for (lx, ly) in self.local_poly:
            wx = cx + (lx * cos_a - ly * sin_a)
            wy = cy + (lx * sin_a + ly * cos_a)
            world_poly.append((wx, wy))
        return cx, cy, angle_deg, world_poly

# Frame the fake blocks move inside - matches a default 640x480 webcam, so
# the pixel coordinates land in the range Unreal is already scaled for
# (previous working scale divisor was /50).
FRAME_W, FRAME_H = 640, 480

# Names the real tracker currently sends; extra blocks get tag-style names.
BASE_NAMES = ["yel1", "yel2", "yel3", "red"]


def make_names(count):
    names = list(BASE_NAMES[:count])
    while len(names) < count:
        names.append(f"tag{len(names)}")
    return names


class FakeBlock:
    """One block circling its own centre, with a wobbling size and depth."""

    def __init__(self, name, index, total):
        self.name = name
        # spread the blocks around the frame so they don't overlap
        spread = (index + 0.5) / total
        self.cx = FRAME_W * (0.2 + 0.6 * spread)
        self.cy = FRAME_H * 0.5
        self.radius = 60 + 30 * math.sin(index)
        self.phase = 2 * math.pi * spread
        self.speed = 0.6 + 0.15 * index          # rad/s, so they desync
        self.base_sx = random.randint(40, 90)
        self.base_sy = random.randint(40, 90)

    def sample(self, t):
        a = self.phase + self.speed * t
        x = int(self.cx + self.radius * math.cos(a))
        y = int(self.cy + self.radius * math.sin(a))
        # true 0..359 rotation - the colour tracker sends 0, ArUco won't
        angle = int(math.degrees(a) % 360)
        sx = int(self.base_sx + 8 * math.sin(a * 2))
        sy = int(self.base_sy + 8 * math.cos(a * 2))
        # metric depth in meters, smaller = closer, same convention as DA3
        z = round(0.65 + 0.25 * math.sin(a), 3)
        return x, y, angle, sx, sy, z


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="UE host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=7000, help="UE OSC port (default 7000)")
    p.add_argument("--ids", type=int, default=4, help="how many blocks to fake (default 4)")
    p.add_argument("--rate", type=float, default=30.0, help="frames per second (default 30)")
    p.add_argument("--churn", type=float, default=0.0,
                   help="seconds between adding/dropping a block (0 = never, the default)")
    p.add_argument("--no-z", action="store_true",
                   help="send the 6-element message (detect.py) instead of 7 with z")
    p.add_argument("--outline", action="store_true",
                   help="test ULegoOscSubsystem/the mesh-extrude receiver instead: "
                        "send lego_locator_xyz.py's real mm/cm /obj + /outline "
                        "messages for rotating fake pieces (some concave), no "
                        "camera or Unreal Blueprint work needed to see them")
    p.add_argument("--quiet", action="store_true", help="don't print every frame")
    args = p.parse_args()

    client = SimpleUDPClient(args.host, args.port)

    if args.outline:
        shape_names = list(OUTLINE_SHAPES)
        names = make_names(args.ids)
        blocks = [FakeOutlineBlock(n, i, args.ids, shape_names[i % len(shape_names)])
                  for i, n in enumerate(names)]
        print(f"faking {len(blocks)} outline pieces -> {args.host}:{args.port} "
              f"/obj + /outline at {args.rate:g} Hz, Ctrl-C to stop")

        start = time.time()
        period = 1.0 / args.rate
        try:
            while True:
                now = time.time()
                t = now - start
                for b in blocks:
                    cx, cy, angle_deg, world_poly = b.sample(t)
                    client.send_message("/obj", [b.name, cx, cy, angle_deg,
                                                  60.0, 60.0, 0.0, b.shape_label])
                    pts_mm = [v for xy in world_poly for v in xy]
                    client.send_message("/outline", [b.name, len(world_poly)]
                                         + pts_mm + [OUTLINE_HEIGHT_CM])
                    if not args.quiet:
                        print(f"{b.name}/{b.shape_label} center=({cx:.0f},{cy:.0f})mm "
                              f"angle={angle_deg:.0f} n_points={len(world_poly)}")
                time.sleep(max(0.0, period - (time.time() - now)))
        except KeyboardInterrupt:
            print("\nstopped")
        return

    blocks = [FakeBlock(n, i, args.ids) for i, n in enumerate(make_names(args.ids))]
    active = list(blocks)          # currently reporting
    benched = []                   # gone quiet, available to come back

    print(f"faking {len(blocks)} blocks -> {args.host}:{args.port} /obj "
          f"at {args.rate:g} Hz ({'6' if args.no_z else '7'} values), Ctrl-C to stop")

    start = time.time()
    next_churn = start + args.churn if args.churn > 0 else None
    period = 1.0 / args.rate

    try:
        while True:
            now = time.time()
            t = now - start

            if next_churn is not None and now >= next_churn:
                # drop or restore one block, so the receiver has to handle a
                # name going quiet and a name showing up again later
                if benched and (len(active) <= 1 or random.random() < 0.5):
                    b = benched.pop(random.randrange(len(benched)))
                    active.append(b)
                    print(f"  [churn] {b.name} back")
                elif active:
                    b = active.pop(random.randrange(len(active)))
                    benched.append(b)
                    print(f"  [churn] {b.name} gone quiet")
                next_churn = now + args.churn

            for b in active:
                x, y, angle, sx, sy, z = b.sample(t)
                msg = [b.name, x, y, angle, sx, sy]
                if not args.no_z:
                    msg.append(z)
                client.send_message("/obj", msg)
                if not args.quiet:
                    print(f"{b.name} pos={x},{y} angle={angle} size={sx}x{sy} z={z}")

            time.sleep(max(0.0, period - (time.time() - now)))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
