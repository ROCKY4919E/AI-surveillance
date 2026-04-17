import cv2
import numpy as np


class AbandonedObjectDetector:
    """
    Rule-based abandoned object detector.

    Logic:
      - Track all bounding boxes that appear stationary over time
      - If a box stays in roughly the same position for `stationary_frames`
        consecutive frames AND no person is nearby, flag as abandoned object

    No model training required — works on top of existing object detections.
    """

    def __init__(self, stationary_frames=900, iou_threshold=0.5, person_proximity=80):
        """
        Args:
            stationary_frames: frames an object must be stationary before alert
                               (e.g. 900 frames @ 30fps = 30 seconds)
            iou_threshold: IoU overlap to consider two boxes the "same" object
            person_proximity: pixel distance; if a person is within this range,
                              object is NOT considered abandoned (owner nearby)
        """
        self.stationary_frames = stationary_frames
        self.iou_threshold = iou_threshold
        self.person_proximity = person_proximity

        # tracked_objects: list of dicts
        # {bbox, frame_count, alerted}
        self.tracked_objects = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame, person_boxes):
        """
        Call once per frame with the full list of detected bounding boxes
        that could be objects (vehicles or unattended items).

        For this project we pass vehicle_boxes as candidate stationary objects
        since a parked / abandoned vehicle is the most common scenario.
        You can also pass any other static box detections.

        Args:
            frame: current BGR frame (not used directly, kept for API consistency)
            person_boxes: list of (x1, y1, x2, y2) for detected persons this frame

        Returns:
            list of dicts: [{'type': 'abandoned_object', 'bbox': (x1,y1,x2,y2)}, ...]
        """
        alerts = []

        for obj in self.tracked_objects:
            if obj['frame_count'] >= self.stationary_frames and not obj['alerted']:
                # Check no person is nearby
                if not self._person_nearby(obj['bbox'], person_boxes):
                    alerts.append({
                        'type': 'abandoned_object',
                        'bbox': obj['bbox']
                    })
                    obj['alerted'] = True  # only alert once per object

        return alerts

    def update(self, candidate_boxes):
        """
        Update the tracker with boxes from the current frame.
        Call this every frame BEFORE calling detect().

        Args:
            candidate_boxes: list of (x1, y1, x2, y2) — typically vehicle_boxes
                             or any boxes you want to monitor for abandonment
        """
        matched_indices = set()

        for box in candidate_boxes:
            matched = False
            for i, obj in enumerate(self.tracked_objects):
                if self._iou(box, obj['bbox']) >= self.iou_threshold:
                    # Same object — increment counter, update bbox slightly
                    obj['frame_count'] += 1
                    obj['bbox'] = box  # update to latest position
                    matched_indices.add(i)
                    matched = True
                    break

            if not matched:
                # New object — start tracking
                self.tracked_objects.append({
                    'bbox': box,
                    'frame_count': 0,
                    'alerted': False
                })

        # Remove objects no longer detected (they moved or disappeared)
        self.tracked_objects = [
            obj for i, obj in enumerate(self.tracked_objects)
            if i in matched_indices
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(boxA, boxB):
        """Compute Intersection over Union between two boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = areaA + areaB - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _person_nearby(self, bbox, person_boxes):
        """Return True if any person centre is within proximity distance of bbox."""
        bx = (bbox[0] + bbox[2]) / 2
        by = (bbox[1] + bbox[3]) / 2

        for pb in person_boxes:
            px = (pb[0] + pb[2]) / 2
            py = (pb[1] + pb[3]) / 2
            dist = np.sqrt((bx - px) ** 2 + (by - py) ** 2)
            if dist < self.person_proximity:
                return True

        return False