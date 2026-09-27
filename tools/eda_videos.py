"""
tools/eda_videos.py — quick look at every sample video: fps, size, duration,
plus one saved thumbnail per video so you can eyeball the camera angle.

Usage:
    python tools/eda_videos.py samples/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_dir")
    ap.add_argument("--out", default="eda_thumbnails")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {args.videos_dir}")
        return

    print(f"{'file':30s} {'fps':>6s} {'w':>6s} {'h':>6s} {'frames':>8s} {'duration_s':>10s}")
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n / fps if fps else 0.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"{vp.stem}_mid.jpg"), frame)
        cap.release()

        print(f"{vp.name:30s} {fps:6.1f} {w:6d} {h:6d} {n:8d} {duration:10.1f}")

    print(f"\nThumbnails saved to {out_dir}/ — look at them before calibrating the scene.")


if __name__ == "__main__":
    main()
