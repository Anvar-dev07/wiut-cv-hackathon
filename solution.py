"""
solution.py — required interface. See src/ for the actual pipeline:

    src/detection.py    YOLO + ByteTrack over a whole video
    src/calibration.py  fixed-camera scene geometry (road/lanes/crosswalks)
    src/events.py       track trajectories -> [start, end, label] events
    src/risk.py         causal, frame-by-frame accident risk (Part B)

Before this does anything useful you MUST calibrate the scene once:

    python tools/calibrate_scene.py samples/<any_clip>.mp4

which writes scene_config.json (road polygon, crosswalks, lane directions).
Without it, detect_events still runs (using whatever needs no calibration:
stopped_vehicle, congestion, accident, near_miss) but wrong_way and
jaywalking will be empty, since they need the road/lane/crosswalk polygons.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.calibration import SceneConfig
from src.detection import run_tracker
from src.events import build_events
from src.risk import OnlineTTCRiskEstimator

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0

WEIGHTS_PATH = "weights/yolo11n.pt"      # bundle this file in weights/ for submission
SCENE_CONFIG_PATH = "scene_config.json"  # produced by tools/calibrate_scene.py
FRAME_STRIDE = 2                          # process every 2nd frame for Part A speed


def detect_events(video_path: str) -> list[list]:
    scene = SceneConfig.load(SCENE_CONFIG_PATH)
    observations, meta = run_tracker(
        video_path, weights_path=WEIGHTS_PATH, frame_stride=FRAME_STRIDE,
    )
    return build_events(observations, meta, scene)


class RiskEstimator:
    """Thin wrapper: the real logic lives in src/risk.py so it can be
    unit-tested without going through the solution.py import path."""

    def __init__(self):
        self._impl = OnlineTTCRiskEstimator(weights_path=WEIGHTS_PATH)
        self.last_score = 0.0

    def reset(self, meta: dict) -> None:
        self._impl.reset(meta)
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.last_score = self._impl.step(frame, t_sec)
        return self.last_score
