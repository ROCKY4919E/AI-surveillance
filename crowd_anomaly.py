import cv2
import numpy as np


class CrowdAnomalyDetector:
    """
    Rule-based crowd anomaly detector.
    Triggers on two conditions:
      1. Occupancy — too many people detected in a single zone
      2. Flow spike — sudden large optical flow magnitude (fight, stampede)
    No model training required.
    """

    def __init__(self, occupancy_threshold=8, flow_threshold=15.0):
        """
        Args:
            occupancy_threshold: number of persons in frame to trigger crowd alert
            flow_threshold: average optical flow magnitude to trigger motion alert
        """
        self.occupancy_threshold = occupancy_threshold
        self.flow_threshold = flow_threshold
        self.prev_gray = None

    def detect(self, frame, person_boxes):
        """
        Args:
            frame: current BGR frame (numpy array)
            person_boxes: list of (x1, y1, x2, y2) tuples from object detector

        Returns:
            list of alert strings, e.g. ['crowd_density', 'crowd_motion']
        """
        alerts = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Condition 1: Occupancy threshold ---
        if len(person_boxes) >= self.occupancy_threshold:
            alerts.append('crowd_density')

        # --- Condition 2: Optical flow spike ---
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = float(np.mean(magnitude))

            if avg_magnitude > self.flow_threshold:
                alerts.append('crowd_motion')

        self.prev_gray = gray
        return alerts