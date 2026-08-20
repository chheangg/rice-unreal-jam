"""
generate_aruco_tags.py — generates printable ArUco tags of an EXACTLY known
size, one per tracked block (yel1/yel2/yel3/red).

Why: the whole source of Z inaccuracy in detect_platform.py was guessing an
irregular LEGO silhouette's real-world width. A printed ArUco tag has an
exact, known size - stick one on each block, and cv2.solvePnP gives that
block's real 3D position directly from geometry, no guessing at all.

Run:  python generate_aruco_tags.py
Output: aruco_tags.png - print at 100% scale (NOT "fit to page"), cut out
        each tag, and stick one flat on top of each tracked block. Keep
        track of which tag ID goes on which block (yel1=0, yel2=1, yel3=2,
        red=3) - detect_platform.py assumes this mapping.
"""

import cv2
import numpy as np
from PIL import Image

ARUCO_DICT = "DICT_4X4_50"    # must match aruco_platform.py
MARKER_SIZE_MM = 30.0          # real printed size (outer edge) - keep this
                                # accurate, it's the whole point
DPI = 300
MM_PER_IN = 25.4
IDS = {0: "yel1", 1: "yel2", 2: "yel3", 3: "red"}

PAGE_W_IN, PAGE_H_IN = 8.5, 11.0


def main():
    px_per_mm = DPI / MM_PER_IN
    marker_px = int(MARKER_SIZE_MM * px_per_mm)
    page_w_px = int(PAGE_W_IN * DPI)
    page_h_px = int(PAGE_H_IN * DPI)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))

    canvas = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)
    margin = int(0.5 * DPI)
    gap = int(0.4 * DPI)
    cols = 2

    for i, (tag_id, label) in enumerate(IDS.items()):
        row, col = divmod(i, cols)
        x0 = margin + col * (marker_px + gap)
        y0 = margin + row * (marker_px + gap + 40)
        tag_img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, marker_px)
        canvas[y0:y0 + marker_px, x0:x0 + marker_px] = tag_img
        label_img = np.full((30, marker_px), 255, dtype=np.uint8)
        cv2.putText(label_img, f"id={tag_id} ({label}) {MARKER_SIZE_MM:.0f}mm",
                    (0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)
        canvas[y0 + marker_px:y0 + marker_px + 30, x0:x0 + marker_px] = label_img

    Image.fromarray(canvas, mode="L").save("aruco_tags.png", dpi=(DPI, DPI))
    print(f"Saved aruco_tags.png - {len(IDS)} tags at {MARKER_SIZE_MM}mm each.")
    print("Print at 100% scale (not 'Fit to Page'), cut out, stick one flat on "
          "each block per the id mapping in this script's IDS dict.")


if __name__ == "__main__":
    main()
