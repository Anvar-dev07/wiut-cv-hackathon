"""
risk.py — causal, frame-by-frame accident-risk estimate (Part B).

Strategy (per the task's own hint): time-to-collision (TTC) between pairs of
tracked vehicles is a strong, simple signal, plus a bonus for hard braking.
Everything here only ever looks at frames already passed to step() — no
lookahead, no re-opening the video.

Running a full YOLO pass on every single frame is the "correct" way to feed
this, but it's expensive; DETECT_EVERY_N controls how often we actually run
the detector, reusing the last known boxes (extrapolated by velocity) on
skipped frames to keep step() itself fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VEHICLE_COCO_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DETECT_EVERY_N = 3          # run YOLO every Nth frame; extrapolate in between
HORIZON_S = 5.0              # must match RISK_HORIZON_SEC in solution.py
TTC_RISK_FLOOR_S = 1.0       # TTC below this -> risk saturates near 1
HARD_BRAKE_RATIO = 0.4
BRAKE_RISK_BONUS = 0.3


@dataclass
class _TrackState:
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    last_t: float = 0.0
    max_speed: float = 0.0


class OnlineTTCRiskEstimator:
    """The actual logic behind solution.RiskEstimator. Kept in src/ so it's
    testable on its own and easy to swap out."""

    def __init__(self, weights_path: str = "weights/yolo11n.pt"):
        self._weights_path = weights_path
        self._model = None  # lazy-loaded on first reset(), so import cost
        self._frame_count = 0
        self.tracks: dict[int, _TrackState] = {}

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._weights_path)

    def reset(self, meta: dict) -> None:
        self._ensure_model()
        self._frame_count = 0
        self.tracks = {}
        self.last_score = 0.0
        # ultralytics keeps tracker state internally keyed by model instance;
        # calling with a fresh persist=False on the first call of a new
        # video avoids leaking ids across videos.
        self._fresh_video = True

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self._frame_count += 1
        run_detector = (self._frame_count % DETECT_EVERY_N == 1) or self._frame_count == 1

        if run_detector:
            result = self._model.track(
                frame,
                classes=list(VEHICLE_COCO_IDS.keys()),
                tracker="bytetrack.yaml",
                persist=not self._fresh_video,
                verbose=False,
            )[0]
            self._fresh_video = False
            boxes = result.boxes
            seen_ids = set()
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    seen_ids.add(int(tid))
                    prev = self.tracks.get(int(tid))
                    if prev is not None:
                        dt = max(t_sec - prev.last_t, 1e-6)
                        vx = (cx - prev.cx) / dt
                        vy = (cy - prev.cy) / dt
                        speed = float(np.hypot(vx, vy))
                        prev.vx, prev.vy = vx, vy
                        prev.cx, prev.cy = cx, cy
                        prev.last_t = t_sec
                        prev.max_speed = max(prev.max_speed * 0.9, speed)
                    else:
                        self.tracks[int(tid)] = _TrackState(cx=cx, cy=cy, last_t=t_sec)
            # drop tracks we haven't seen in a while (simple pruning)
            stale = [tid for tid, st in self.tracks.items()
                     if tid not in seen_ids and (t_sec - st.last_t) > 2.0]
            for tid in stale:
                del self.tracks[tid]
        else:
            # extrapolate positions with last known velocity
            for st in self.tracks.values():
                dt = t_sec - st.last_t
                st.cx += st.vx * dt
                st.cy += st.vy * dt
                st.last_t = t_sec

        score = self._risk_from_tracks()
        self.last_score = score
        return score

    def _risk_from_tracks(self) -> float:
        ids = list(self.tracks.keys())
        if len(ids) < 2:
            return 0.0
        best = 0.0
        for i in range(len(ids)):
            a = self.tracks[ids[i]]
            for j in range(i + 1, len(ids)):
                b = self.tracks[ids[j]]
                rel_pos = np.array([b.cx - a.cx, b.cy - a.cy])
                rel_vel = np.array([b.vx - a.vx, b.vy - a.vy])
                dist = float(np.linalg.norm(rel_pos))
                closing_speed = -float(np.dot(rel_pos, rel_vel)) / (dist + 1e-6)
                if closing_speed <= 1e-3 or dist < 1e-3:
                    continue
                ttc = dist / closing_speed
                if 0 < ttc <= HORIZON_S:
                    risk = np.clip(1.0 - (ttc - TTC_RISK_FLOOR_S) / (HORIZON_S - TTC_RISK_FLOOR_S), 0.0, 1.0)
                    best = max(best, risk)
        # small bonus if anyone is braking hard right now
        for st in self.tracks.values():
            speed = float(np.hypot(st.vx, st.vy))
            if st.max_speed > 1e-3 and speed <= HARD_BRAKE_RATIO * st.max_speed:
                best = min(1.0, best + BRAKE_RISK_BONUS)
                break
        return float(best)
