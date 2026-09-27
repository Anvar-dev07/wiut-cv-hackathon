"""
tools/calibrate_scene.py — click-to-calibrate the fixed camera scene, once.

Usage:
    python tools/calibrate_scene.py samples/some_clip.mp4 [--time 5.0]

Opens one frame in a window. For each polygon: left-click to add points,
'a' to accept the polygon, 'z' to undo the last point, ESC to cancel it.
Terminal prompts (y/n) drive how many crosswalks/lanes/stop-lines you add.
Saves scene_config.json in the current directory (used by solution.py).

Needs a display (run on your own machine, not on the headless eval box).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.calibration import Lane, SceneConfig, StopLine  # noqa: E402

WINDOW = "calibrate (click points, 'a'=accept, 'z'=undo, ESC=cancel)"


def grab_frame(video_path: str, t_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame at t={t_sec}s from {video_path}")
    return frame


def click_polygon(base_img: np.ndarray, instruction: str) -> list[list[float]]:
    points: list[list[float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([float(x), float(y)])

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"\n{instruction}\n  left-click = add point | 'a' = accept | 'z' = undo | ESC = cancel/skip")

    while True:
        disp = base_img.copy()
        if len(points) > 1:
            cv2.polylines(disp, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2)
        for p in points:
            cv2.circle(disp, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        cv2.putText(disp, instruction[:80], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('a'):
            return points
        if key == ord('z') and points:
            points.pop()
        if key == 27:  # ESC
            return []


def click_two_points(base_img: np.ndarray, instruction: str) -> list[list[float]]:
    pts = click_polygon(base_img, instruction + " (click exactly 2 points, then 'a')")
    return pts[:2]


def ask_yes_no(prompt: str) -> bool:
    return input(f"{prompt} (y/n): ").strip().lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to a sample .mp4 from the fixed camera")
    ap.add_argument("--time", type=float, default=5.0, help="which second to grab a frame from")
    ap.add_argument("--out", default="scene_config.json")
    args = ap.parse_args()

    frame = grab_frame(args.video, args.time)
    h, w = frame.shape[:2]
    scene = SceneConfig(frame_width=w, frame_height=h)

    road = click_polygon(frame, "1) Click the ROAD/carriageway boundary (the driveable area)")
    scene.road_polygon = road

    idx = 1
    while ask_yes_no("Add a crosswalk polygon?"):
        cw = click_polygon(frame, f"Crosswalk #{idx}: click its 4 corners")
        if cw:
            scene.crosswalks.append({"name": f"crosswalk_{idx}", "polygon": cw})
            idx += 1

    idx = 1
    while ask_yes_no("Add a lane (area + its allowed travel direction)?"):
        lane_poly = click_polygon(frame, f"Lane #{idx}: click its boundary polygon")
        if not lane_poly:
            continue
        arrow = click_two_points(frame, f"Lane #{idx}: click 2 points along its ALLOWED direction (from -> to)")
        if len(arrow) == 2:
            direction = [arrow[1][0] - arrow[0][0], arrow[1][1] - arrow[0][1]]
        else:
            direction = [0.0, -1.0]
        scene.lanes.append(Lane(name=f"lane_{idx}", polygon=lane_poly, direction=direction))
        idx += 1

    idx = 1
    while ask_yes_no("Add a stop line?"):
        pts = click_two_points(frame, f"Stop line #{idx}: click its 2 endpoints")
        if len(pts) == 2:
            scene.stop_lines.append(StopLine(name=f"stop_{idx}", p1=pts[0], p2=pts[1]))
            idx += 1

    cv2.destroyAllWindows()
    scene.save(args.out)
    print(f"\nSaved {args.out}: road={len(scene.road_polygon)} pts, "
          f"{len(scene.crosswalks)} crosswalks, {len(scene.lanes)} lanes, "
          f"{len(scene.stop_lines)} stop lines.")


if __name__ == "__main__":
    main()
