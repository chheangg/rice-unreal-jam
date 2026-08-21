"""
unreal_bridge.py - the one place that knows what BP_OSCreciver expects.

Why this file exists
--------------------
The trackers got smarter (colour-agnostic detection, metric XYZ, real sizes in
cm) but the Unreal receiver did not. `BP_OSCreciver` is still the original
blueprint, and it is picky in two ways that make a "correct-looking" OSC stream
land in Unreal as nothing at all:

  1. IT READS INTEGERS. The blueprint pulls each field with
     `Get OSC Message Integer at Index`. python-osc tags a Python float as OSC
     type 'f', and UE's GetInt32 does NOT coerce an 'f' - it fails and hands
     the graph a 0. So a stream of perfectly good floats moves every cube to
     the origin (i.e. to wherever the receiver actor is standing) and nothing
     appears to happen. Everything sent here is a Python `int`.

  2. IT ONLY KNOWS FOUR NAMES. Index 0 goes into a Switch on String with the
     cases `yel1`, `yel2`, `yel3`, `red` - each wired to one pre-placed cube
     component. There is no spawn path. A name the switch has never heard of
     (`yellow`, `blue`, `red/cross`, ...) falls through the default pin and is
     silently dropped. `SlotMap` below hands out those four names.

  3. The cubes are COMPONENTS of the receiver actor and are moved with
     SetRelativeLocation. If `BP_OSCreciver` is not dragged into the level, or
     the level is not in Play, there is nothing to move and nothing to see, no
     matter how good this stream is.

Message layout (unchanged from the original detect.py, on purpose):

    /obj  [name:str, cx:int, cy:int, angle:int, sizeX:int, sizeY:int]

cx/cy are PIXELS in the camera frame - the blueprint does its own
subtract-centre / scale maths on them. Do not "helpfully" upgrade them to
millimetres here; that maths lives in the graph.
"""

BP_YELLOW_SLOTS = ["yel1", "yel2", "yel3"]   # Cube_yellow1/2/3 in the BP
BP_RED_SLOT = "red"                          # Cube_pink in the BP

# Colours the blueprint has a cube for. Anything else is detected fine by the
# tracker and has nowhere to go in Unreal until someone adds a cube + a case.
SUPPORTED_COLORS = {"yellow", "red"}


class SlotMap:
    """Assigns each detected piece one of the blueprint's four fixed names.

    The rule is the original one from detect.py, kept so the demo behaves the
    way it did: yellows sorted small -> large become yel1/yel2/yel3, the
    biggest red becomes red. Sorting by size rather than by tracker id means a
    given physical brick keeps its cube as long as you don't swap two bricks of
    near-identical size, and it needs no state that can drift out of sync.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._warned = set()

    def assign(self, detections):
        """detections: [(color, area, payload_dict)] -> [(slot_name, payload)].

        payload_dict is passed straight back; this only decides names.
        """
        by_color = {}
        for color, area, payload in detections:
            by_color.setdefault(color, []).append((area, payload))

        out = []
        for area, payload in sorted(by_color.get("yellow", []))[:3]:
            out.append((BP_YELLOW_SLOTS[len(out)], payload))
        reds = sorted(by_color.get("red", []))
        if reds:
            out.append((BP_RED_SLOT, reds[-1][1]))     # biggest red

        for color in by_color:
            if color not in SUPPORTED_COLORS and color not in self._warned:
                self._warned.add(color)
                print(f"[bridge] '{color}' detected but BP_OSCreciver has no "
                      f"cube for it - add a component + a Switch case, or use "
                      f"yellow/red pieces for now")
        return out


class UnrealBridge:
    """Sends the legacy /obj message BP_OSCreciver actually parses."""

    def __init__(self, host="127.0.0.1", port=7000, verbose=False):
        from pythonosc.udp_client import SimpleUDPClient   # lazy: optional dep
        self.client = SimpleUDPClient(host, port)
        self.slots = SlotMap(verbose=verbose)
        self.verbose = verbose
        self.sent = 0
        print(f"[bridge] /obj -> {host}:{port} (int payload, "
              f"names {BP_YELLOW_SLOTS + [BP_RED_SLOT]})")

    def send_frame(self, detections):
        """detections: [(color, area, (cx, cy, angle, size_x, size_y))]."""
        for name, (cx, cy, angle, sx, sy) in self.slots.assign(detections):
            # int() on every field: an OSC 'f' where the blueprint wants an 'i'
            # reads back as 0 and pins the cube to the origin.
            msg = [name, int(round(cx)), int(round(cy)), int(round(angle)),
                   int(round(sx)), int(round(sy))]
            self.client.send_message("/obj", msg)
            self.sent += 1
            if self.verbose:
                print(f"[bridge] /obj {msg}")
