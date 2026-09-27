"""
calibration.py — fixed-camera scene geometry.

The camera never moves, so we calibrate the scene ONCE (by looking at one
frame) and reuse the same polygons/lines for every video, sample or hidden.

Run `tools/calibrate_scene.py <a_sample_video.mp4>` to build
`scene_config.json` interactively (click points on a frame). This module
just loads that file and answers geometry questions:

    - is a point inside the road?
    - is a point inside any crosswalk?
    - which lane is a point in, and what is that lane's allowed direction?
    - how far is a point from a stop line?
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test. polygon = [[x, y], ...]."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def dist_point_to_segment(point, p1, p2) -> float:
    p = np.array(point, dtype=float)
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


@dataclass
class Lane:
    name: str
    polygon: list[list[float]]
    direction: list[float]  # unit vector of the ALLOWED travel direction

    def unit_direction(self) -> np.ndarray:
        d = np.array(self.direction, dtype=float)
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else d


@dataclass
class StopLine:
    name: str
    p1: list[float]
    p2: list[float]


@dataclass
class SceneConfig:
    road_polygon: list[list[float]] = field(default_factory=list)
    crosswalks: list[dict] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0

    # ---- queries -----------------------------------------------------
    def is_on_road(self, point) -> bool:
        if not self.road_polygon:
            return True  # no calibration yet -> don't filter anything out
        return point_in_polygon(point, self.road_polygon)

    def is_on_crosswalk(self, point) -> bool:
        return any(point_in_polygon(point, cw["polygon"]) for cw in self.crosswalks)

    def lane_for(self, point) -> Lane | None:
        for lane in self.lanes:
            if point_in_polygon(point, lane.polygon):
                return lane
        return None

    def min_dist_to_stop_line(self, point) -> float:
        if not self.stop_lines:
            return float("inf")
        return min(dist_point_to_segment(point, sl.p1, sl.p2) for sl in self.stop_lines)

    # ---- io ------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "SceneConfig":
        path = Path(path)
        if not path.exists():
            # Empty calibration: rules that need it will be skipped/loosened.
            return cls()
        data = json.loads(path.read_text())
        return cls(
            road_polygon=data.get("road_polygon", []),
            crosswalks=data.get("crosswalks", []),
            lanes=[Lane(**l) for l in data.get("lanes", [])],
            stop_lines=[StopLine(**s) for s in data.get("stop_lines", [])],
            frame_width=data.get("frame_width", 0),
            frame_height=data.get("frame_height", 0),
        )

    def save(self, path: str | Path) -> None:
        data = {
            "road_polygon": self.road_polygon,
            "crosswalks": self.crosswalks,
            "lanes": [vars(l) for l in self.lanes],
            "stop_lines": [vars(s) for s in self.stop_lines],
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
        Path(path).write_text(json.dumps(data, indent=2))
