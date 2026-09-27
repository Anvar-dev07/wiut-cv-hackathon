"""
events.py — turn tracked trajectories into [start_sec, end_sec, label] events.

Design: for each class, build a per-frame boolean "is this happening right
now, anywhere in frame" signal, then collapse consecutive True frames into
segments (segment_from_flags). This automatically satisfies "same-class
segments must not overlap" and matches the task's per-video, not per-object,
notion of an event.

Implemented now (tractable from tracks + scene calibration alone):
    stopped_vehicle, congestion, wrong_way, jaywalking, near_miss, accident

Deliberately left as TODO (need signals beyond position/velocity, e.g.
traffic-light colour, painted lane lines, U-turn geometry, or an
appearance-based obstacle/fire detector — see NOTES.md):
    red_light, illegal_u_turn, illegal_turn, solid_line_crossing,
    stop_line, road_obstacle, fire_smoke
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import SceneConfig
from .detection import Observation, group_by_track

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Tunables. Start here; retune against your own labels (my_labels.json).
STOPPED_SPEED_PX_S = 8.0        # below this -> "not moving" (scale with frame size)
STOPPED_MIN_DURATION_S = 10.0   # per the official definition
CONGESTION_SLOW_FRACTION = 0.6  # share of on-road vehicles that must be slow
CONGESTION_MIN_VEHICLES = 3
WRONG_WAY_COS_THRESHOLD = -0.3  # dot(velocity_unit, lane_direction) below this
WRONG_WAY_MIN_SPEED_PX_S = 15.0
NEAR_CLOSE_PX = 60.0            # bbox-center distance considered "close"
HARD_BRAKE_RATIO = 0.4          # speed must drop to <=40% of recent max
MIN_EVENT_DURATION_S = 0.5      # drop blips shorter than this
MERGE_GAP_S = 1.0               # merge same-class segments separated by <1s


def segments_from_flags(flag_times: list[tuple[float, bool]]) -> list[tuple[float, float]]:
    """flag_times: sorted [(t_sec, bool), ...] (one entry per processed frame).
    Returns merged [start, end] segments where the flag was True, after
    dropping short blips and merging near-adjacent segments.
    """
    if not flag_times:
        return []
    segments = []
    start = None
    prev_t = flag_times[0][0]
    for t, flag in flag_times:
        if flag and start is None:
            start = t
        if not flag and start is not None:
            segments.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None:
        segments.append((start, prev_t))

    # drop blips
    segments = [(s, e) for s, e in segments if e - s >= MIN_EVENT_DURATION_S]
    # merge close segments
    merged: list[list[float]] = []
    for s, e in segments:
        if merged and s - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def detect_stopped_and_congestion(observations, tracks, frame_times, scene: SceneConfig):
    """Returns (stopped_events, congestion_events)."""
    # Per-track stationary state machine.
    stopped_flags: dict[float, bool] = {t: False for t in frame_times}
    for tid, obs_list in tracks.items():
        if obs_list[0].cls not in VEHICLE_CLASSES:
            continue
        run_start = None
        for i, obs in enumerate(obs_list):
            if not scene.is_on_road((obs.cx, obs.cy)):
                run_start = None
                continue
            if obs.speed <= STOPPED_SPEED_PX_S:
                if run_start is None:
                    run_start = obs.t_sec
                elapsed = obs.t_sec - run_start
                if elapsed >= STOPPED_MIN_DURATION_S:
                    # mark every processed frame time in [run_start, obs.t_sec]
                    for t in frame_times:
                        if run_start <= t <= obs.t_sec:
                            stopped_flags[t] = True
            else:
                run_start = None

    # Congestion: fraction of on-road vehicles that are slow, each frame.
    by_frame_time = {}
    for obs in observations:
        by_frame_time.setdefault(obs.t_sec, []).append(obs)
    congestion_flags: dict[float, bool] = {}
    for t in frame_times:
        on_road_vehicles = [
            o for o in by_frame_time.get(t, [])
            if o.cls in VEHICLE_CLASSES and scene.is_on_road((o.cx, o.cy))
        ]
        if len(on_road_vehicles) >= CONGESTION_MIN_VEHICLES:
            slow = sum(1 for o in on_road_vehicles if o.speed <= STOPPED_SPEED_PX_S)
            congestion_flags[t] = (slow / len(on_road_vehicles)) >= CONGESTION_SLOW_FRACTION
        else:
            congestion_flags[t] = False

    sorted_times = sorted(frame_times)
    stopped_events = segments_from_flags([(t, stopped_flags[t]) for t in sorted_times])
    congestion_events = segments_from_flags([(t, congestion_flags[t]) for t in sorted_times])
    return stopped_events, congestion_events


def detect_wrong_way(tracks, frame_times, scene: SceneConfig):
    flags = {t: False for t in frame_times}
    if not scene.lanes:
        return []  # can't judge direction without calibrated lanes
    for tid, obs_list in tracks.items():
        if obs_list[0].cls not in VEHICLE_CLASSES:
            continue
        for obs in obs_list:
            if obs.speed < WRONG_WAY_MIN_SPEED_PX_S:
                continue
            lane = scene.lane_for((obs.cx, obs.cy))
            if lane is None:
                continue
            vel_unit = np.array([obs.vx, obs.vy]) / (obs.speed + 1e-9)
            cos_sim = float(np.dot(vel_unit, lane.unit_direction()))
            if cos_sim < WRONG_WAY_COS_THRESHOLD:
                flags[obs.t_sec] = True
    sorted_times = sorted(frame_times)
    return segments_from_flags([(t, flags[t]) for t in sorted_times])


def detect_jaywalking(tracks, frame_times, scene: SceneConfig):
    flags = {t: False for t in frame_times}
    for tid, obs_list in tracks.items():
        if obs_list[0].cls != "person":
            continue
        for obs in obs_list:
            on_road = scene.is_on_road((obs.cx, obs.cy))
            on_crossing = scene.is_on_crosswalk((obs.cx, obs.cy))
            if on_road and not on_crossing:
                flags[obs.t_sec] = True
    sorted_times = sorted(frame_times)
    return segments_from_flags([(t, flags[t]) for t in sorted_times])


def detect_accident_and_near_miss(tracks, frame_times, scene: SceneConfig):
    """Heuristic: two vehicle tracks get close while at least one brakes hard.
    Overlapping boxes -> accident candidate. Close-but-clear + hard brake ->
    near_miss candidate. This is a starting point, not a final solution —
    retune thresholds against your own labels.
    """
    accident_flags = {t: False for t in frame_times}
    near_miss_flags = {t: False for t in frame_times}

    # recent max speed per track, for "hard brake" detection
    recent_max_speed: dict[int, float] = {}

    by_time: dict[float, list] = {}
    for tid, obs_list in tracks.items():
        for obs in obs_list:
            by_time.setdefault(obs.t_sec, []).append((tid, obs))

    for t in sorted(frame_times):
        entries = by_time.get(t, [])
        # update rolling max speed + detect hard brakes this frame
        braking_ids = set()
        for tid, obs in entries:
            prev_max = recent_max_speed.get(tid, obs.speed)
            if prev_max > 1e-6 and obs.speed <= HARD_BRAKE_RATIO * prev_max and prev_max > WRONG_WAY_MIN_SPEED_PX_S:
                braking_ids.add(tid)
            recent_max_speed[tid] = max(obs.speed, prev_max * 0.9)  # slow decay

        vehicles = [(tid, obs) for tid, obs in entries if obs.cls in VEHICLE_CLASSES]
        for i in range(len(vehicles)):
            tid_a, a = vehicles[i]
            for j in range(i + 1, len(vehicles)):
                tid_b, b = vehicles[j]
                dist = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
                overlap = (
                    abs(a.cx - b.cx) < (a.w + b.w) / 2
                    and abs(a.cy - b.cy) < (a.h + b.h) / 2
                )
                if overlap:
                    accident_flags[t] = True
                elif dist < NEAR_CLOSE_PX and (tid_a in braking_ids or tid_b in braking_ids):
                    near_miss_flags[t] = True

    sorted_times = sorted(frame_times)
    accident_events = segments_from_flags([(t, accident_flags[t]) for t in sorted_times])
    near_miss_events = segments_from_flags([(t, near_miss_flags[t]) for t in sorted_times])
    return accident_events, near_miss_events


def build_events(observations: list[Observation], meta: dict, scene: SceneConfig) -> list[list]:
    tracks = group_by_track(observations)
    frame_times = sorted({o.t_sec for o in observations})
    if not frame_times:
        return []

    events: list[list] = []

    stopped, congestion = detect_stopped_and_congestion(observations, tracks, frame_times, scene)
    events += [[s, e, "stopped_vehicle"] for s, e in stopped]
    events += [[s, e, "congestion"] for s, e in congestion]

    wrong_way = detect_wrong_way(tracks, frame_times, scene)
    events += [[s, e, "wrong_way"] for s, e in wrong_way]

    jaywalking = detect_jaywalking(tracks, frame_times, scene)
    events += [[s, e, "jaywalking"] for s, e in jaywalking]

    accident, near_miss = detect_accident_and_near_miss(tracks, frame_times, scene)
    events += [[s, e, "accident"] for s, e in accident]
    events += [[s, e, "near_miss"] for s, e in near_miss]

    # clamp to video duration, just in case
    duration = meta.get("duration", 0.0)
    if duration:
        events = [[max(0.0, s), min(duration, e), label] for s, e, label in events if s < e]

    return events
