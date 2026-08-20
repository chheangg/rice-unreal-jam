"""
platform_height.py — real-world X/Y/Z for a block seen by a SINGLE angled
camera (the DJI at ~45 deg), using the red platform border as a known-size
reference instead of a printed checkerboard.

THE GEOMETRY (why this isn't just "read pixel coordinates"):
--------------------------------------------------------------
At an angle, pixel distances don't correspond to real distances uniformly -
a block near the bottom of the frame (closer to the camera) moves more
pixels per real mm than one near the top. This is perspective/lens warp.
To get real, undistorted X/Y/Z we need the camera's actual pose in 3D space,
not just its pixel output.

  1. find_platform_corners() finds the 4 outer corners of the red border in
     the frame.
  2. solve_platform_pose() runs cv2.solvePnP with those corners (matched to
     PLATFORM_SIZE_MM, the border's real-world size) to get the camera's
     exact position + orientation relative to the platform. This only needs
     the platform border itself - no checkerboard, no second camera.
  3. X, Y (real mm, platform-relative): for a block's BASE pixel (where it
     touches the platform), we cast the camera ray through that pixel and
     intersect it with the platform's flat plane (Z=0). This is EXACT
     geometry - given a correct pose, it's not an approximation, and it's
     what correctly undoes the perspective warp regardless of camera angle.
     See PlatformHeightEstimator.ground_xy_mm().
  4. Z (real mm height above the platform): a ray alone can't tell you how
     far along it a point is without another clue, so height still needs
     the known real-world block size vs. how big it looks on screen (a
     known-size object at N pixels wide is at a computable distance). This
     is inherently approximate (REAL_WIDTH_MM below is a visual estimate,
     not measured) but is corrected by the pose the same way X/Y is.
     See PlatformHeightEstimator.point3d_mm().

PLATFORM_SIZE_MM is a rough visual estimate (~300mm square), not measured -
same tunable-constant pattern used elsewhere in this project. Pose is solved
once per run - restart the tracker if the DJI moves.

The platform border is RED - the same color tracked blocks use - so finding
its corners must NOT reuse the small-blob red detector as-is: the border is
a large thin rectangular outline, tracked red pieces are small solid blobs.
find_platform_corners() filters by contour area to avoid confusing the two.

COORDINATE SYSTEM (per spec): origin (0,0) is the platform's BOTTOM-LEFT
corner (as seen in the rotation-corrected, portrait-oriented frame); X
increases left-to-right; Y increases bottom-to-top; both range 0..
PLATFORM_SIZE_MM. Z is height above the platform surface, in mm.
find_platform_corners() identifies which of the 4 detected corners is
actually bottom-left/bottom-right/top-left/top-right (by position relative
to their centroid - image Y grows downward, so "bottom" = larger pixel Y),
so this holds regardless of which order cv2 happens to return them in.
"""

import numpy as np
import cv2

PLATFORM_SIZE_MM = 300.0   # rough estimate - inner play-area square, see chat

REAL_WIDTH_MM = {
    "yellow": 48.0,
    "red": 24.0,
}

# Same RED range as detect.py/detect_stereo.py - reused here to find the
# platform border's mask before filtering it down to just the large frame.
RED = [((0, 70, 50), (10, 255, 255)),
       ((170, 70, 50), (179, 255, 255))]

MIN_BORDER_AREA = 20000   # the frame's total red area is much bigger than a
                           # single tracked red block - tune if needed


def _classify_corners(box):
    """
    box: 4x2 array of corner pixels, any order. Returns them reordered as
    [bottom_left, bottom_right, top_left, top_right] by position relative to
    their centroid (image Y grows downward, so larger Y = more "bottom").
    """
    centroid = box.mean(axis=0)
    bl = br = tl = tr = None
    for p in box:
        is_bottom = p[1] > centroid[1]
        is_right = p[0] > centroid[0]
        if is_bottom and not is_right:
            bl = p
        elif is_bottom and is_right:
            br = p
        elif not is_bottom and not is_right:
            tl = p
        else:
            tr = p
    if bl is None or br is None or tl is None or tr is None:
        return None   # degenerate (e.g. 2 points on the same side of centroid)
    return np.array([bl, br, tl, tr], dtype=np.float32)


def find_platform_corners(hsv):
    """
    Returns the 4 outer corners of the red platform border, ordered
    [bottom_left, bottom_right, top_left, top_right] (see module docstring
    for the coordinate system this defines), or None if not found. Looks for
    the largest red contour above MIN_BORDER_AREA, distinct from small
    tracked red blocks by size.
    """
    mask = None
    for lo, hi in RED:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else mask | m
    # close small gaps between the border's brick segments so it reads as
    # one connected contour rather than several disjoint pieces.
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= MIN_BORDER_AREA]
    if not candidates:
        return None
    largest = max(candidates, key=cv2.contourArea)

    # The border is a thin frame, not a filled shape - its contour traces the
    # outline. minAreaRect over that outline gives the 4 outer corners.
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return _classify_corners(box)


def solve_platform_pose(K, dist, corners_px):
    """
    corners_px: 4x2 array [bottom_left, bottom_right, top_left, top_right] -
    exactly find_platform_corners()'s output. Returns (rvec, tvec) - the
    camera's pose relative to the platform plane (world Z=0 at the platform
    surface, world origin at the BOTTOM-LEFT corner - see module docstring).
    """
    size = PLATFORM_SIZE_MM
    objp = np.array([
        [0, 0, 0],          # bottom_left
        [size, 0, 0],       # bottom_right
        [0, size, 0],       # top_left
        [size, size, 0],    # top_right
    ], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, corners_px.astype(np.float32), K, dist)
    if not ok:
        return None, None
    return rvec, tvec


class PlatformHeightEstimator:
    """Real X/Y/Z for pixels in the DJI frame, given a solved platform pose."""

    def __init__(self, K, dist, rvec, tvec):
        self.K = K
        self.dist = dist
        self.tvec = tvec.flatten()
        self.R = cv2.Rodrigues(rvec)[0]
        self.R_inv = self.R.T
        self.fx = K[0, 0]
        # Camera's own position in world (platform) space - the ray-plane
        # intersection's "origin" term. Derived once, reused every call.
        self.cam_center_world = -self.R_inv @ self.tvec

    def _ray_world(self, px, py):
        """Undistorted camera ray through pixel (px,py), as a WORLD-space
        direction vector (not unit length, that's fine for what we use it
        for)."""
        pts = np.array([[[float(px), float(py)]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, self.K, self.dist)
        x_n, y_n = undist[0, 0]
        ray_cam = np.array([x_n, y_n, 1.0])
        return self.R_inv @ ray_cam

    def ground_xy_mm(self, px, py):
        """
        EXACT real-world (X, Y) in mm, platform-relative, for a pixel that
        lies ON the platform surface (Z=0) - e.g. a block's base/contact
        point, not its center. This is the perspective-warp-corrected
        position: same math regardless of where in frame or how angled the
        camera is, given a correct pose. Returns None if the ray is
        parallel to the platform (shouldn't happen for a real camera pose).
        """
        origin = self.cam_center_world
        direction = self._ray_world(px, py)
        if abs(direction[2]) < 1e-9:
            return None
        t = -origin[2] / direction[2]
        if t <= 0:
            return None   # platform is "behind" the ray - bad pixel/pose
        point = origin + t * direction
        return float(point[0]), float(point[1])

    def point3d_mm(self, cx, cy, pixel_width, color_name):
        """
        Approximate full 3D point (X, Y, Z) in mm for a block's CENTER pixel
        (cx, cy), using its known real-world width vs. apparent pixel width
        to estimate distance from the camera. Z (height) has no other way
        to be computed from one camera; X/Y here are LESS accurate than
        ground_xy_mm() (they drift if the width estimate is off), so prefer
        ground_xy_mm() for X/Y and just take .z from this for height.
        Returns None if inputs are degenerate.

        NOTE: this whole method exists only for objects WITHOUT an ArUco tag.
        If a marker is available, marker_world_xyz() is exact geometry (no
        guessed real-world size needed) and should be preferred entirely -
        see ArucoBlockTracker below.
        """
        real_width = REAL_WIDTH_MM.get(color_name)
        if real_width is None or pixel_width <= 1:
            return None
        depth_mm = self.fx * real_width / pixel_width
        pts = np.array([[[float(cx), float(cy)]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, self.K, self.dist)
        x_n, y_n = undist[0, 0]
        p_cam = np.array([x_n * depth_mm, y_n * depth_mm, depth_mm])
        return self.camera_point_to_world(p_cam)

    def camera_point_to_world(self, p_cam):
        """Transform a 3D point already expressed in CAMERA space into
        platform-world coordinates (mm, origin at platform bottom-left)."""
        p_world = self.R_inv @ (np.asarray(p_cam, dtype=np.float64) - self.tvec)
        return float(p_world[0]), float(p_world[1]), float(p_world[2])


# ============================================================================
# ArUco tag pose - EXACT block position/height, no guessed real-world size.
# See generate_aruco_tags.py: print a tag of MARKER_SIZE_MM and stick one on
# each tracked block. Because the tag's real size is known precisely (unlike
# an irregular LEGO silhouette), solvePnP gives an exact 3D position with no
# depth-from-apparent-size guessing at all - this is what point3d_mm() could
# never be, and is the fix for the Z drift/inaccuracy seen without tags.
# ============================================================================

def make_aruco_detector(dict_name="DICT_4X4_50"):
    """Build an ArUco detector (OpenCV >= 4.7 API - this project's cv2 is
    4.13, which no longer has the old estimatePoseSingleMarkers convenience
    function; solvePnP below is OpenCV's own recommended replacement)."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(d, params)


def detect_markers(gray, detector):
    """Returns {marker_id: corners_4x2} for every marker found this frame.
    Corner order per marker is cv2.aruco's own: TL, TR, BR, BL."""
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.flatten()):
        out[int(i)] = c.reshape(4, 2)
    return out


def solve_marker_camera_point(corners_px, K, dist, marker_size_mm):
    """
    corners_px: 4x2 array, cv2.aruco order (TL, TR, BR, BL).
    Returns the marker's exact CENTER position in CAMERA space (mm), or None
    if solvePnP fails. The object points below are centered at the marker's
    own origin, so solvePnP's tvec output directly IS the center position -
    no separate "read off the translation" step needed.
    """
    s = marker_size_mm / 2.0
    objp = np.array([
        [-s, s, 0],    # TL
        [s, s, 0],     # TR
        [s, -s, 0],    # BR
        [-s, -s, 0],   # BL
    ], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, corners_px.astype(np.float32), K, dist)
    if not ok:
        return None
    return tvec.flatten()
