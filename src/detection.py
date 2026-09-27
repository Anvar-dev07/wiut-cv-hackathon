"""
detection.py — vehicle/pedestrian detection + tracking over a whole video.

Wraps Ultralytics YOLO with its built-in ByteTrack tracker. Produces one
flat list of "observations": one per (track, frame) pair, with position,
class, and a running velocity estimate. Everything downstream (events.py,
risk.py) works off this list, not off raw frames.

Why YOLO+ByteTrack: it's pretrained (no training needed), runs on CPU at
usable speed in the "nano" size, and open-weights (allowed by the rules).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# COCO class ids we care about (pretrained YOLO already knows these).
VEHICLE_COCO_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_COCO_ID = 0
TRACKED_COCO_IDS = list(VEHICLE_COCO_IDS.keys()) + [PERSON_COCO_ID]


@dataclass
class Observation:
    frame_idx: int
    t_sec: float
    track_id: int
    cls: str  # "car" | "motorcycle" | "bus" | "truck" | "person"
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0  # px/sec, filled in after the pass
    vy: float = 0.0
    speed: float = 0.0  # px/sec magnitude


def run_tracker(
    video_path: str,
    weights_path: str = "weights/yolo11n.pt",
    frame_stride: int = 2,
    conf: float = 0.25,
    imgsz: int = 640,
    max_frames: int | None = None,
):
    """Runs detection+tracking over the whole video, returns:
        observations: list[Observation] (one per track per processed frame)
        meta: {"fps", "width", "height", "n_frames", "duration"}

    frame_stride > 1 skips frames for speed; ByteTrack tolerates small gaps
    fine, and we still get a track_id per vehicle/pedestrian.
    """
    from ultralytics import YOLO  # local import: heavy, only needed here

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n_frames / fps if fps else 0.0

    observations: list[Observation] = []
    last_seen: dict[int, tuple[int, float, float, float]] = {}  # id -> (frame_idx, cx, cy, t)

    frame_idx = -1
    results_gen = model.track(
        source=video_path,
        conf=conf,
        imgsz=imgsz,
        classes=TRACKED_COCO_IDS,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )
    for result in results_gen:
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_stride > 1 and frame_idx % frame_stride != 0:
            continue
        t_sec = frame_idx / fps

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for tid, c, (x1, y1, x2, y2) in zip(ids, clss, xyxy):
            cls_name = VEHICLE_COCO_IDS.get(int(c), "person" if c == PERSON_COCO_ID else None)
            if cls_name is None:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1), (y2 - y1)

            vx = vy = speed = 0.0
            if tid in last_seen:
                pf, pcx, pcy, pt = last_seen[tid]
                dt = t_sec - pt
                if dt > 1e-6:
                    vx = (cx - pcx) / dt
                    vy = (cy - pcy) / dt
                    speed = float(np.hypot(vx, vy))
            last_seen[tid] = (frame_idx, cx, cy, t_sec)

            observations.append(
                Observation(
                    frame_idx=frame_idx, t_sec=t_sec, track_id=int(tid), cls=cls_name,
                    cx=cx, cy=cy, w=w, h=h, vx=vx, vy=vy, speed=speed,
                )
            )

    meta = {"fps": fps, "width": width, "height": height, "n_frames": n_frames, "duration": duration}
    return observations, meta


def group_by_track(observations: list[Observation]) -> dict[int, list[Observation]]:
    tracks: dict[int, list[Observation]] = {}
    for obs in observations:
        tracks.setdefault(obs.track_id, []).append(obs)
    for tid in tracks:
        tracks[tid].sort(key=lambda o: o.frame_idx)
    return tracks
