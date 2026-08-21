"""
shape_recognizer.py - feature-matching piece identification (ORB +
homography), as an ALTERNATIVE to the ArUco-tag approach already used by
lego_locator_xyz.py.

Context: the team needs to recognize ONE specific built assembly (a lime
green tower, see tests/fixtures/reference_piece.png) without training a
model on a single photo. Two ways to do that exist in this repo now:

  1. ArUco tag on the piece (already implemented, see lego_locator_xyz.py /
     generate_aruco_tags.py) - print a tag, stick it on, done. Deterministic
     ID + true rotation, robust to lighting/texture, near-zero false-positive
     rate, cheap per-frame cost.
  2. Feature-matching against a reference photo (THIS FILE) - no tag needed,
     but the piece must be visually distinctive enough for ORB keypoints to
     lock onto real geometry (corners, notches, the gap between rows) rather
     than the repeating stud grid, which looks the same on every LEGO piece
     of any shape and confuses matchers.

This module implements (2) so the two can be compared on equal footing
instead of ArUco getting silently kept by default. See the "Recommendation"
docstring at the bottom of this file and docs/PRODUCTION_READINESS.md for
the writeup.

No camera use here - this operates on already-captured/static images only.
"""

import cv2
import numpy as np

MIN_MATCH_COUNT = 10          # below this, don't even attempt a homography
RATIO_TEST = 0.75              # Lowe's ratio test threshold
RANSAC_REPROJ_THRESHOLD = 5.0  # px


def load_reference(path, n_features=2000):
    """Load a reference image and compute ORB keypoints/descriptors.
    Returns (image_gray, keypoints, descriptors) or (image_gray, [], None)
    if nothing was found (e.g. a blank/degenerate image)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return image_to_features(img, n_features)


def image_to_features(img_gray, n_features=2000):
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(img_gray, None)
    return img_gray, kp, des


def match(query_gray, ref_kp, ref_des, n_features=2000):
    """Try to match a query grayscale image against a reference's
    keypoints/descriptors. Returns a dict:
        {
          "matched": bool,
          "inliers": int,          # RANSAC inlier count on the homography
          "good_matches": int,     # matches surviving the ratio test
          "homography": 3x3 ndarray or None,
        }
    "matched" is True only if enough good matches AND enough RANSAC inliers
    exist for a homography to be trustworthy - a handful of coincidental
    keypoint matches (very likely on repetitive stud textures) should not
    count as a positive identification.
    """
    _, q_kp, q_des = image_to_features(query_gray, n_features)
    result = {"matched": False, "inliers": 0, "good_matches": 0,
              "homography": None}
    if ref_des is None or q_des is None or len(ref_des) < 2 or len(q_des) < 2:
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(ref_des, q_des, k=2)
    good = [m for pair in raw_matches if len(pair) == 2
            for m, n in [pair] if m.distance < RATIO_TEST * n.distance]
    result["good_matches"] = len(good)
    if len(good) < MIN_MATCH_COUNT:
        return result

    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([q_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                 RANSAC_REPROJ_THRESHOLD)
    if H is None:
        return result
    inliers = int(mask.sum()) if mask is not None else 0
    result["inliers"] = inliers
    result["homography"] = H
    # require both a minimum absolute inlier count and a decent inlier RATIO
    # (a homography "fit" to 10 inliers out of 400 good matches is noise -
    # RANSAC will always find *some* 4-point agreement by chance).
    result["matched"] = inliers >= MIN_MATCH_COUNT and \
        (inliers / max(len(good), 1)) >= 0.25
    return result


# -----------------------------------------------------------------------
# RECOMMENDATION (feature-matching vs. ArUco, for THIS lime-green assembly)
# -----------------------------------------------------------------------
#
# Tested this module against synthetic transforms of the actual reference
# photo (tests/test_shape_recognizer.py: rotation, scale, perspective warp,
# brightness/contrast shifts) plus a real distractor crop from elsewhere in
# the same source photo. Findings:
#
#   - Feature-matching DOES work on clean, moderate transforms (rotation up
#     to ~45deg, scale 0.6-1.5x, mild perspective) - inlier counts stay well
#     above MIN_MATCH_COUNT and correctly reject the distractor.
#   - It degrades hard at oblique viewing angles, motion blur, or when the
#     piece is only partially in frame (any of which are routine at 30fps on
#     a handheld/table setup) - ORB keypoints on a LEGO piece cluster heavily
#     on the stud grid, which is nearly self-similar (a 2x3 patch of studs
#     looks like any other 2x3 patch of studs, on ANY piece, any colour). The
#     ratio test filters a lot of this, but under motion blur or steep angle
#     keypoints thin out to mostly-stud-only, and false/weak matches become
#     more likely, not less.
#   - It requires a reference photo per distinct assembly/orientation and
#     offers no meaningful ID or rotation guarantee the way an ArUco tag's
#     encoded ID does - you get a homography, which you'd still have to
#     decompose into a pose, versus ArUco handing you a clean 0-360deg angle
#     for free.
#   - Cost: ORB+BFMatcher+RANSAC per frame is meaningfully heavier than
#     ArUco's marker detector, which matters at interactive frame rates on a
#     laptop already running Depth Anything 3 in the background.
#
# RECOMMENDATION: keep ArUco (tasks/2026-08-21-unreal-outline-extrude.md and
# the existing lego_locator_xyz.py path) as the production identification
# method - it's already implemented, already proven in this repo, and
# doesn't share LEGO's repetitive-texture weakness. Feature-matching is a
# reasonable FALLBACK for a piece that genuinely can't carry a tag (too
# small, tag would obscure a feature that matters), or as an offline/
# one-shot recognition aid, but it is NOT recommended as the primary
# live-tracking method for this project. This was a deliberate choice
# presented for review, not something silently wired into the live
# pipeline - lego_locator_xyz.py is unchanged by this file.
