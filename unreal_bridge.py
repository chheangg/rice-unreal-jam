"""
unreal_bridge.py - the one place that knows what BP_OSCreciver expects.

Why this file exists
--------------------
The trackers got smarter (colour-agnostic detection, several pieces per colour,
metric XYZ, real sizes in cm) but the Unreal receiver did not. `BP_OSCreciver`
is still the original blueprint, and it is picky in ways that make a
"correct-looking" OSC stream land in Unreal as nothing at all:

  1. IT READS INTEGERS. The blueprint pulls each field with
     `Get OSC Message Integer at Index`. python-osc tags a Python float as OSC
     type 'f', and UE's GetInt32 does NOT coerce an 'f' - it fails and hands
     the graph a 0. So a stream of perfectly good floats moves every cube to
     the origin (i.e. to wherever the receiver actor is standing) and nothing
     appears to happen. Everything sent here is a Python `int`.

  2. IT ONLY KNOWS FOUR NAMES. Index 0 goes into a Switch on String with the
     cases `yel1`, `yel2`, `yel3`, `red` - each wired to one pre-placed cube
     component. There is no spawn path. A name the switch has never heard of
     (`yellow1`, `blue2`, ...) falls through the default pin and is silently
     dropped. `SlotMap` below leases those four names out to tracked pieces.

  3. The cubes are COMPONENTS of the receiver actor and are moved with
     SetRelativeLocation. If `BP_OSCreciver` is not dragged into the level, or
     the level is not in Play, there is nothing to move and nothing to see, no
     matter how good this stream is.

So Unreal can currently show FOUR pieces: three yellow and one red. The tracker
can follow as many as the camera can see, of any colour - `capacity_note()`
reports what is being dropped, rather than letting it vanish quietly. Lifting
the limit is a receiver change (spawn per name), not a tracker change.

Message layout (unchanged from the original detect.py, on purpose):

    /obj  [name:str, cx:int, cy:int, angle:int, sizeX:int, sizeY:int]

cx/cy are PIXELS in the camera frame - the blueprint does its own
subtract-centre / scale maths on them. Do not "helpfully" upgrade them to
millimetres here; that maths lives in the graph.
"""

# Which blueprint cube each colour may drive. The names are the Switch on
# String cases; the cubes behind them are Cube_yellow1/2/3 and Cube_pink.
BP_SLOTS_BY_COLOR = {
    "yellow": ["yel1", "yel2", "yel3"],
    "red": ["red"],
}

# Frames a piece may go missing before its cube is handed back. A brick that
# blinks out for two frames because a hand crossed it should come back to the
# SAME cube, otherwise the whole scene reshuffles on every flicker.
RELEASE_AFTER_MISSES = 30


class SlotMap:
    """Leases the blueprint's four fixed cube names to tracked pieces.

    Keyed by the tracker's stable per-piece name (`yellow1`, `yellow2`, ...),
    NOT by size order. The original detect.py sorted yellows small -> large and
    handed out yel1/yel2/yel3 in that order, which means two bricks of similar
    size swap cubes the moment a measurement wobbles - the "size-sort
    reshuffling" already on the roadmap as a thing to get rid of. A lease is
    held until the piece has been gone for RELEASE_AFTER_MISSES frames, so a
    brick keeps its cube across a brief occlusion and only genuinely departed
    pieces free one up.
    """

    def __init__(self):
        self.leases = {}            # piece name -> blueprint slot name
        self._misses = {}           # piece name -> frames unseen
        self._warned_color = set()
        self._warned_full = set()

    def _free_slots(self, color):
        taken = set(self.leases.values())
        return [s for s in BP_SLOTS_BY_COLOR.get(color, []) if s not in taken]

    def assign(self, detections):
        """detections: [(color, piece_name, payload)] -> [(slot_name, payload)].

        payload is passed straight back; this only decides which cube each
        piece drives.
        """
        seen = set()
        out = []
        for color, piece, payload in detections:
            seen.add(piece)
            slot = self.leases.get(piece)
            if slot is None:
                if color not in BP_SLOTS_BY_COLOR:
                    if color not in self._warned_color:
                        self._warned_color.add(color)
                        print(f"[bridge] '{color}' tracked but BP_OSCreciver "
                              f"has no cube for it - add a component + a "
                              f"Switch case, or use yellow/red for now")
                    continue
                free = self._free_slots(color)
                if not free:
                    if piece not in self._warned_full:
                        self._warned_full.add(piece)
                        n = len(BP_SLOTS_BY_COLOR[color])
                        print(f"[bridge] {piece}: all {n} {color} cube(s) in "
                              f"the blueprint are already leased, so this "
                              f"piece is tracked but not shown. The receiver "
                              f"needs to spawn per name to go past {n}.")
                    continue
                slot = free[0]
                self.leases[piece] = slot
            self._misses[piece] = 0
            out.append((slot, payload))

        # Hand back cubes from pieces that have really gone, not ones that
        # flickered for a frame.
        for piece in list(self.leases):
            if piece in seen:
                continue
            self._misses[piece] = self._misses.get(piece, 0) + 1
            if self._misses[piece] > RELEASE_AFTER_MISSES:
                del self.leases[piece]
                del self._misses[piece]
                self._warned_full.discard(piece)
        return out

    def capacity_note(self, detections):
        """'3 tracked, 2 shown (1 dropped)' - or None when everything fits."""
        shown = len({p for _c, p, _pl in detections if p in self.leases})
        total = len({p for _c, p, _pl in detections})
        if total <= shown:
            return None
        return f"{total} tracked, {shown} shown ({total - shown} over capacity)"


class UnrealBridge:
    """Sends the legacy /obj message BP_OSCreciver actually parses."""

    def __init__(self, host="127.0.0.1", port=7000, verbose=False):
        from pythonosc.udp_client import SimpleUDPClient   # lazy: optional dep
        self.client = SimpleUDPClient(host, port)
        self.slots = SlotMap()
        self.verbose = verbose
        self._last_note = None
        names = sum(BP_SLOTS_BY_COLOR.values(), [])
        print(f"[bridge] /obj -> {host}:{port} (int payload, "
              f"{len(names)} cubes: {names})")

    def send_frame(self, detections):
        """detections: [(color, piece_name, (cx, cy, angle, size_x, size_y))]."""
        for name, (cx, cy, angle, sx, sy) in self.slots.assign(detections):
            # int() on every field: an OSC 'f' where the blueprint wants an 'i'
            # reads back as 0 and pins the cube to the origin.
            msg = [name, int(round(cx)), int(round(cy)), int(round(angle)),
                   int(round(sx)), int(round(sy))]
            self.client.send_message("/obj", msg)
            if self.verbose:
                print(f"[bridge] /obj {msg}")

        # Say it once per change, not once per frame.
        note = self.slots.capacity_note(detections)
        if note != self._last_note:
            self._last_note = note
            if note:
                print(f"[bridge] {note}")
