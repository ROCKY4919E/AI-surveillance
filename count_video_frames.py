import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Count total frames in a video file.")
    parser.add_argument("video", type=Path, help="Path to the video file.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = frame_count / fps if fps > 0 else 0.0

    capture.release()

    print(f"Video: {video_path}")
    print(f"Frames: {frame_count}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration (seconds): {duration:.2f}")


if __name__ == "__main__":
    main()
