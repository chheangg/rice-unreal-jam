"""
ue_multi_receiver.py - run INSIDE Unreal. Spawns one actor per tracked piece,
so the scene is no longer limited to the blueprint's four fixed cubes.

Why this exists rather than a fixed Blueprint
---------------------------------------------
`BP_OSCreciver` has five pre-placed cube components and a Switch on String with
four cases. It cannot show a fifth piece, and it cannot show a colour it has no
cube for - there is no spawn path in the graph. Adding one by hand is editor
work; adding one from Python is not possible either, which is what
`inspect_bp.py` established (UE's Python API exposes graphs but not node
wiring).

So this script skips the blueprint. It reads the OSC stream itself, keeps a
dict of name -> spawned actor, and spawns/moves/despawns as pieces come and go.
Any number of pieces, any colour, no graph surgery. `BP_OSCreciver` is left
alone and still works for its four cubes if you prefer it.

Two deliberate choices:

  * A RAW UDP SOCKET plus ~40 lines of OSC parsing, not the OSC plugin. Binding
    a UOSCServer's delegate from Python is fragile across engine versions, and
    OSC's wire format is trivial: an address string, a comma type-tag string,
    then the args, everything padded to 4 bytes. Parsing it here costs less
    than depending on plugin/Python interop behaving.
  * A SLATE TICK CALLBACK, so it runs in the editor with the level NOT in Play.
    The blueprint's server only exists during Play (it is created on
    BeginPlay), which is a large part of why "nothing shows up" was so easy to
    hit.

Run it (Unreal editor, Output Log -> Cmd dropdown set to "Python"):

    exec(open(r"D:\\Projects\\rice-unreal-jam\\ue_multi_receiver.py").read())

Then in a terminal, point the tracker at this receiver's port:

    python lego_locator_xyz.py 0 --osc-metric --osc-port 7001

To stop it:  stop_receiver()      (or just re-exec the file, it restarts clean)

Both message layouts are accepted, decided by the arg types on the wire:
    metric (floats):  [name, x_mm, y_mm, angle, long_cm, short_cm, z_mm]
    legacy (ints):    [name, cx_px, cy_px, angle, size_x_px, size_y_px]
The metric one is what you want here - real centimetres and a real position,
which the fixed-cube blueprint could never use.
"""

import socket
import struct
import time

try:
    import unreal
except ImportError:                     # so the parser can be tested outside UE
    unreal = None

PORT = 7001                 # NOT 7000: BP_OSCreciver already binds that in Play
ADDRESS = "/obj"
DROP_AFTER_S = 2.0          # despawn a piece unheard for this long
MAX_PER_TICK = 400          # datagrams to drain per tick, so a flood can't hang

# Tracker metres -> Unreal units. The tracker sends mm, UE is 1 uu = 1 cm.
MM_TO_UU = 0.1
WORLD_ORIGIN = (0.0, 0.0, 50.0)   # where the table's origin sits in the level
FLIP_Y = True               # camera Y grows downward, UE Y grows right
PIXEL_DIVISOR = 50.0        # legacy px messages: the ÷50 that worked before
# One knob to make the twin bigger than life. Real bricks are a few cm, so at
# 1.0 the scene is life-size and reads as tiny next to UE's 100 uu default
# cube; the demo usually wants 5-10. Positions and sizes scale together, so the
# layout stays faithful - only the ruler changes.
DEMO_SCALE = 1.0


def _actor_api():
    """EditorActorSubsystem where available, else the deprecated library."""
    if unreal is None:
        return None
    try:
        return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    except Exception:
        return unreal.EditorLevelLibrary


CUBE = "/Engine/BasicShapes/Cube.Cube"
MATERIALS = {               # by colour prefix of the piece name
    "yellow": "/Engine/BasicShapes/Mi_yellow.Mi_yellow",
    "red": "/Engine/BasicShapes/Mi_Pink.Mi_Pink",
    "orange": "/Engine/BasicShapes/Mi_Pink.Mi_Pink",
    "green": "/Engine/BasicShapes/Mi_Silver.Mi_Silver",
    "blue": "/Engine/BasicShapes/Mi_Silver.Mi_Silver",
}
TAG = "LegoTwin"            # so we only ever delete actors we spawned


# --------------------------------------------------------------------------
# OSC parsing. Pure functions, no `unreal` - testable from a normal terminal.
# --------------------------------------------------------------------------
def _padded(n):
    """OSC pads every field to a 4-byte boundary."""
    return (n + 3) & ~3


def _read_string(buf, i):
    end = buf.index(b"\x00", i)
    return buf[i:end].decode("utf-8", "replace"), i + _padded(end - i + 1)


def parse_osc(buf):
    """(address, [args]) for one OSC message, or None if it isn't one.

    Bundles are ignored: the tracker never sends them, and silently mis-parsing
    one would be worse than skipping it.
    """
    try:
        if buf[:1] != b"/":
            return None
        address, i = _read_string(buf, 0)
        if i >= len(buf) or buf[i:i + 1] != b",":
            return None
        tags, i = _read_string(buf, i)
        args = []
        for t in tags[1:]:
            if t == "i":
                args.append(struct.unpack_from(">i", buf, i)[0]); i += 4
            elif t == "f":
                args.append(struct.unpack_from(">f", buf, i)[0]); i += 4
            elif t == "s":
                v, i = _read_string(buf, i)
                args.append(v)
            elif t in "TF":
                args.append(t == "T")
            elif t == "d":
                args.append(struct.unpack_from(">d", buf, i)[0]); i += 8
            else:
                return None             # a type we don't speak; drop the whole
        return address, args
    except Exception:
        return None


def piece_from_args(args):
    """One /obj message -> {name, loc_uu, yaw, size_uu} in Unreal units.

    The two layouts are told apart by type, not by length: the metric one is
    floats, the legacy blueprint one is ints. That is the same distinction that
    made the blueprint read zeros, so it is worth keying off explicitly.
    """
    if len(args) < 6 or not isinstance(args[0], str):
        return None
    name = args[0]
    metric = any(isinstance(a, float) for a in args[1:])
    x, y, angle, s_long, s_short = args[1:6]
    z = args[6] if len(args) > 6 else 0.0

    if metric:
        ux = x * MM_TO_UU
        uy = y * MM_TO_UU * (-1.0 if FLIP_Y else 1.0)
        uz = z * MM_TO_UU
        sx, sy = s_long, s_short                 # already centimetres = uu
    else:
        ux = x / PIXEL_DIVISOR
        uy = y / PIXEL_DIVISOR * (-1.0 if FLIP_Y else 1.0)
        uz = 0.0
        sx = s_long / PIXEL_DIVISOR
        sy = s_short / PIXEL_DIVISOR
    ux, uy, uz = ux * DEMO_SCALE, uy * DEMO_SCALE, uz * DEMO_SCALE
    sx, sy = sx * DEMO_SCALE, sy * DEMO_SCALE

    return {"name": name,
            "loc": (WORLD_ORIGIN[0] + ux,
                    WORLD_ORIGIN[1] + uy,
                    WORLD_ORIGIN[2] + uz),
            "yaw": float(angle),
            # The basic cube mesh is 100 uu on a side, so scale = uu / 100.
            "scale": (max(sx, 1.0) / 100.0, max(sy, 1.0) / 100.0, 0.12)}


def color_of(name):
    """'yellow2' -> 'yellow'. Names are colour + instance number."""
    return name.rstrip("0123456789") or name


# --------------------------------------------------------------------------
# The Unreal half.
# --------------------------------------------------------------------------
class Receiver:
    def __init__(self, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.actors = {}                # piece name -> spawned actor
        self.last_seen = {}             # piece name -> time.time()
        self.handle = None
        self.api = _actor_api()
        self.mesh = unreal.load_asset(CUBE)
        self._mat_cache = {}
        self.port = port
        unreal.log(f"[twin] listening for OSC {ADDRESS} on udp/{port}")

    # ---- actor management ----
    def _material(self, color):
        if color not in self._mat_cache:
            path = MATERIALS.get(color)
            self._mat_cache[color] = unreal.load_asset(path) if path else None
        return self._mat_cache[color]

    def _spawn(self, name):
        actor = self.api.spawn_actor_from_class(
            unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
        actor.set_actor_label(f"{TAG}_{name}")
        comp = actor.static_mesh_component
        comp.set_static_mesh(self.mesh)
        # Spawned pieces are driven every frame, so they must not be static.
        comp.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        mat = self._material(color_of(name))
        if mat:
            comp.set_material(0, mat)
        unreal.log(f"[twin] + {name}")
        return actor

    def _despawn(self, name):
        actor = self.actors.pop(name, None)
        self.last_seen.pop(name, None)
        if actor:
            unreal.log(f"[twin] - {name}")
            self.api.destroy_actor(actor)

    def apply(self, piece):
        name = piece["name"]
        actor = self.actors.get(name)
        if actor is None or not actor.is_valid_lowlevel():
            actor = self._spawn(name)
            self.actors[name] = actor
        actor.set_actor_location_and_rotation(
            unreal.Vector(*piece["loc"]),
            unreal.Rotator(0.0, 0.0, piece["yaw"]),
            sweep=False, teleport=True)
        actor.set_actor_scale3d(unreal.Vector(*piece["scale"]))
        self.last_seen[name] = time.time()

    # ---- the tick ----
    def tick(self, _delta):
        for _ in range(MAX_PER_TICK):
            try:
                data, _addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            msg = parse_osc(data)
            if not msg or msg[0] != ADDRESS:
                continue
            piece = piece_from_args(msg[1])
            if piece:
                self.apply(piece)

        now = time.time()
        for name in [n for n, t in self.last_seen.items()
                     if now - t > DROP_AFTER_S]:
            self._despawn(name)

    # ---- lifecycle ----
    def start(self):
        self.handle = unreal.register_slate_post_tick_callback(self.tick)
        return self

    def stop(self):
        if self.handle is not None:
            unreal.unregister_slate_post_tick_callback(self.handle)
            self.handle = None
        for name in list(self.actors):
            self._despawn(name)
        self.sock.close()
        unreal.log("[twin] stopped")


def clean_orphans():
    """Delete pieces left behind by a previous run that was killed mid-tick."""
    api = _actor_api()
    if api is None:
        return
    try:
        actors = api.get_all_level_actors()
    except Exception:
        return
    for a in actors:
        try:
            if a and a.get_actor_label().startswith(f"{TAG}_"):
                api.destroy_actor(a)
        except Exception:
            pass


def start_receiver(port=PORT):
    """Restartable: stops any previous instance first, then re-exec is safe."""
    stop_receiver()
    clean_orphans()
    globals()["_RECEIVER"] = Receiver(port).start()
    return globals()["_RECEIVER"]


def stop_receiver():
    r = globals().get("_RECEIVER")
    if r is not None:
        r.stop()
        globals()["_RECEIVER"] = None


if unreal is not None:
    start_receiver()
