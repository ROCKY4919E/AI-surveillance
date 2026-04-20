import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO



VISDRONE_PERSON_CLASSES = {"pedestrian", "people"}
VISDRONE_VEHICLE_CLASSES = {
    "bicycle",
    "bus",
    "car",
    "motor",
    "tricycle",
    "truck",
    "van",
    "awning-tricycle",
}
ACCIDENT_LABELS = {"accident", "normal"}


@dataclass
class PipelineConfig:
    source: str = "0"
    visdrone_weights: Path = Path("runs") / "visdrone" / "weights" / "best.pt"
    dfire_weights: Path = Path("runs") / "dfire" / "yolov8m_dfire" / "weights" / "best.pt"
    accident_weights: Path = (
        Path("runs") / "accident" / "yolov8m_accident" / "weights" / "best.pt"
    )
    device: str = "0"
    imgsz: int = 640
    accident_imgsz: int = 224
    visdrone_conf: float = 0.35
    dfire_conf: float = 0.35
    accident_conf: float = 0.70
    frame_stride: int = 2
    show: bool = False
    save_output: Path | None = None
    max_frames: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the integrated AI surveillance pipeline on a video source."
    )
    parser.add_argument("--source", default="0", help="Video path or camera index.")
    parser.add_argument(
        "--visdrone-weights",
        type=Path,
        default=PipelineConfig.visdrone_weights,
        help="Path to the VisDrone detection weights.",
    )
    parser.add_argument(
        "--dfire-weights",
        type=Path,
        default=PipelineConfig.dfire_weights,
        help="Path to the D-Fire detection weights.",
    )
    parser.add_argument(
        "--accident-weights",
        type=Path,
        default=PipelineConfig.accident_weights,
        help="Path to the accident classification weights.",
    )
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Detection image size.")
    parser.add_argument(
        "--accident-imgsz",
        type=int,
        default=224,
        help="Classification image size for the accident model.",
    )
    parser.add_argument(
        "--visdrone-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for VisDrone detections.",
    )
    parser.add_argument(
        "--dfire-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for D-Fire detections.",
    )
    parser.add_argument(
        "--accident-conf",
        type=float,
        default=0.70,
        help="Minimum confidence to raise an accident alert.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Run model inference every Nth frame to improve FPS on laptop hardware.",
    )
    parser.add_argument("--show", action="store_true", help="Display the annotated frames live.")
    parser.add_argument(
        "--save-output",
        type=Path,
        default=None,
        help="Optional output video path for annotated results.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to process the whole source.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else (base_dir / value).resolve()


def open_source(source_value: str):
    if source_value.isdigit():
        return cv2.VideoCapture(int(source_value))
    return cv2.VideoCapture(source_value)


def xyxy_to_tuple(box):
    x1, y1, x2, y2 = box
    return int(x1), int(y1), int(x2), int(y2)


def draw_box(frame, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def collect_all_detection_boxes(result):
    names = result.names
    detections = []
    if result.boxes is None:
        return detections

    xyxy_values = result.boxes.xyxy.cpu().tolist()
    cls_values = result.boxes.cls.cpu().tolist()
    conf_values = result.boxes.conf.cpu().tolist()

    for xyxy, cls_id, conf in zip(xyxy_values, cls_values, conf_values):
        detections.append(
            {
                "bbox": xyxy_to_tuple(xyxy),
                "label": names[int(cls_id)],
                "confidence": float(conf),
            }
        )
    return detections


def classify_accident(result):
    probs = result.probs
    if probs is None:
        return None

    names = result.names
    top_index = int(probs.top1)
    label = names[top_index]
    confidence = float(probs.top1conf)

    if label not in ACCIDENT_LABELS:
        return None

    return {"label": label, "confidence": confidence}


def create_writer(output_path: Path, fps: float, frame_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec_candidates = ["avc1", "H264", "mp4v"]

    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(
        f"Could not create a video writer for {output_path}. Tried codecs: {', '.join(codec_candidates)}"
    )


def fit_frame_to_screen(frame, max_width=1280, max_height=720):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame

    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def build_status_lines(frame_index, started_at, processed_frames, snapshot, accident_conf):
    elapsed = max(1e-6, time.time() - started_at)
    inference_fps = processed_frames / elapsed
    status_lines = [f"Frame: {frame_index}", f"FPS: {inference_fps:.1f}"]

    accident_prediction = snapshot["accident"]
    if (
        accident_prediction
        and accident_prediction["label"] == "accident"
        and accident_prediction["confidence"] >= accident_conf
    ):
        status_lines.append(f'ALERT accident {accident_prediction["confidence"]:.2f}')
    elif accident_prediction:
        status_lines.append(
            f'accident_model: {accident_prediction["label"]} {accident_prediction["confidence"]:.2f}'
        )

    for alert_name in snapshot["crowd_alerts"]:
        status_lines.append(f"ALERT {alert_name}")

    for detection in snapshot["dfire"]:
        if detection["label"] in {"fire", "smoke"}:
            status_lines.append(f'ALERT {detection["label"]} {detection["confidence"]:.2f}')

    for alert in snapshot["abandoned_alerts"]:
        status_lines.append(f"ALERT {alert['type']}")

    return status_lines


class PipelineRunner:
    def __init__(self, base_dir: Path, config: PipelineConfig):
        self.base_dir = base_dir
        self.config = config
        self.visdrone_weights = resolve_path(base_dir, config.visdrone_weights)
        self.dfire_weights = resolve_path(base_dir, config.dfire_weights)
        self.accident_weights = resolve_path(base_dir, config.accident_weights)

        for weights_path in (
            self.visdrone_weights,
            self.dfire_weights,
            self.accident_weights,
        ):
            if not weights_path or not weights_path.exists():
                raise FileNotFoundError(f"Missing weights file: {weights_path}")

        self.visdrone_model = YOLO(str(self.visdrone_weights))
        self.dfire_model = YOLO(str(self.dfire_weights))
        self.accident_model = YOLO(str(self.accident_weights))

    def run(self, progress_callback=None):
        config = self.config
        capture = open_source(config.source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {config.source}")

        if config.show:
            cv2.namedWindow("AI Surveillance Pipeline", cv2.WINDOW_NORMAL)

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        fps = source_fps if source_fps and source_fps > 0 else 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = None
        frame_index = 0
        processed_frames = 0
        event_counts = {
            "fire": 0,
            "smoke": 0,
            "accident": 0,
            "crowd_density": 0,
            "crowd_motion": 0,
            "abandoned_object": 0,
        }
        last_snapshot = {
            "visdrone": [],
            "dfire": [],
            "accident": None,
            "crowd_alerts": [],
            "abandoned_alerts": [],
        }
        started_at = time.time()
        crowd_detector = CrowdAnomalyDetector()
        abandoned_detector = AbandonedObjectDetector()
        resolved_output_path = resolve_path(self.base_dir, config.save_output)

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if config.max_frames and frame_index > config.max_frames:
                    break

                run_inference = processed_frames == 0 or frame_index % config.frame_stride == 0

                if run_inference:
                    visdrone_result = self.visdrone_model.predict(
                        frame,
                        imgsz=config.imgsz,
                        conf=config.visdrone_conf,
                        device=config.device,
                        verbose=False,
                    )[0]
                    dfire_result = self.dfire_model.predict(
                        frame,
                        imgsz=config.imgsz,
                        conf=config.dfire_conf,
                        device=config.device,
                        verbose=False,
                    )[0]
                    accident_result = self.accident_model.predict(
                        frame,
                        imgsz=config.accident_imgsz,
                        device=config.device,
                        verbose=False,
                    )[0]

                    visdrone_detections = collect_all_detection_boxes(visdrone_result)
                    dfire_detections = collect_all_detection_boxes(dfire_result)
                    person_boxes = [
                        item["bbox"]
                        for item in visdrone_detections
                        if item["label"] in VISDRONE_PERSON_CLASSES
                    ]
                    vehicle_boxes = [
                        item["bbox"]
                        for item in visdrone_detections
                        if item["label"] in VISDRONE_VEHICLE_CLASSES
                    ]

                    abandoned_detector.update(vehicle_boxes)
                    crowd_alerts = crowd_detector.detect(frame, person_boxes)
                    abandoned_alerts = abandoned_detector.detect(frame, person_boxes)
                    accident_prediction = classify_accident(accident_result)

                    last_snapshot = {
                        "visdrone": visdrone_detections,
                        "dfire": dfire_detections,
                        "accident": accident_prediction,
                        "crowd_alerts": crowd_alerts,
                        "abandoned_alerts": abandoned_alerts,
                    }
                    processed_frames += 1

                    for detection in dfire_detections:
                        if detection["label"] in event_counts:
                            event_counts[detection["label"]] += 1

                    if (
                        accident_prediction
                        and accident_prediction["label"] == "accident"
                        and accident_prediction["confidence"] >= config.accident_conf
                    ):
                        event_counts["accident"] += 1

                    for alert_name in crowd_alerts:
                        if alert_name in event_counts:
                            event_counts[alert_name] += 1

                    if abandoned_alerts:
                        event_counts["abandoned_object"] += len(abandoned_alerts)

                annotated = frame.copy()

                for detection in last_snapshot["visdrone"]:
                    draw_box(
                        annotated,
                        detection["bbox"],
                        f'{detection["label"]} {detection["confidence"]:.2f}',
                        (0, 255, 0),
                    )

                for detection in last_snapshot["dfire"]:
                    draw_box(
                        annotated,
                        detection["bbox"],
                        f'{detection["label"]} {detection["confidence"]:.2f}',
                        (0, 140, 255),
                    )

                for alert in last_snapshot["abandoned_alerts"]:
                    draw_box(annotated, alert["bbox"], alert["type"], (0, 0, 255))

                status_lines = build_status_lines(
                    frame_index,
                    started_at,
                    processed_frames,
                    last_snapshot,
                    config.accident_conf,
                )

                y = 30
                for line in status_lines:
                    cv2.putText(
                        annotated,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 28

                if writer is None and resolved_output_path is not None:
                    writer = create_writer(
                        resolved_output_path,
                        fps,
                        (annotated.shape[1], annotated.shape[0]),
                    )

                if writer is not None:
                    writer.write(annotated)

                if config.show:
                    preview_frame = fit_frame_to_screen(annotated)
                    cv2.imshow("AI Surveillance Pipeline", preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        break

                if progress_callback is not None:
                    progress_callback(
                        {
                            "frame_index": frame_index,
                            "total_frames": total_frames,
                            "processed_frames": processed_frames,
                            "status_lines": status_lines,
                            "event_counts": dict(event_counts),
                        }
                    )
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if config.show:
                cv2.destroyAllWindows()

        elapsed = time.time() - started_at
        return {
            "frames_read": frame_index,
            "inference_steps": processed_frames,
            "total_frames": total_frames,
            "elapsed_seconds": elapsed,
            "output_path": str(resolved_output_path) if resolved_output_path else None,
            "event_counts": event_counts,
        }


def run_pipeline(config: PipelineConfig, progress_callback=None):
    base_dir = Path(__file__).resolve().parent
    runner = PipelineRunner(base_dir, config)
    return runner.run(progress_callback=progress_callback)


def main():
    args = parse_args()
    config = PipelineConfig(
        source=args.source,
        visdrone_weights=args.visdrone_weights,
        dfire_weights=args.dfire_weights,
        accident_weights=args.accident_weights,
        device=args.device,
        imgsz=args.imgsz,
        accident_imgsz=args.accident_imgsz,
        visdrone_conf=args.visdrone_conf,
        dfire_conf=args.dfire_conf,
        accident_conf=args.accident_conf,
        frame_stride=args.frame_stride,
        show=args.show,
        save_output=args.save_output,
        max_frames=args.max_frames,
    )

    result = run_pipeline(config)
    print("Pipeline finished.")
    print(f"Frames read: {result['frames_read']}")
    print(f"Inference steps: {result['inference_steps']}")
    if result["output_path"]:
        print(f"Saved annotated video to: {result['output_path']}")


if __name__ == "__main__":
    main()
