"""YOLO-based obstacle and vehicle detection subsystem.

Uses the *Ultralytics* YOLOv8 implementation.  The first call to
:class:`ObstacleDetector` will download ``yolov8n.pt`` automatically if the
file is not already present on disk.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .config import ETS2Config

logger = logging.getLogger(__name__)


class ObstacleDetector:
    """Wraps a YOLOv8 model for efficient per-frame obstacle detection.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.

    Notes
    -----
    YOLO inference is the most CPU/GPU-intensive step.  To keep overall
    throughput near 30 FPS the detector only runs every
    :attr:`~ets2_driver.config.DetectionConfig.inference_every_n_frames`
    frames and returns a cached result on skipped frames.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._frame_counter: int = 0
        self._last_boxes: List[List[float]] = []

        # Lazy-import so the module can be imported without ultralytics installed
        # (e.g. in unit-test environments that mock the detector)
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(cfg.detection.model_path)
            logger.info("YOLO model loaded: %s", cfg.detection.model_path)
        except Exception as exc:
            logger.warning("Could not load YOLO model (%s). "
                           "Obstacle detection is disabled.", exc)
            self._model = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[List[float]]:
        """Return a list of bounding boxes for detected road obstacles.

        Each box is ``[x1, y1, x2, y2]`` in pixel coordinates relative to
        *frame*.  The result is cached and only refreshed every
        ``inference_every_n_frames`` calls.

        Parameters
        ----------
        frame:
            BGR image from the vision system.

        Returns
        -------
        List[List[float]]
            Possibly-empty list of ``[x1, y1, x2, y2]`` boxes.
        """
        self._frame_counter += 1
        skip = self.cfg.detection.inference_every_n_frames
        if self._frame_counter % skip != 0:
            return self._last_boxes

        if self._model is None:
            return []

        try:
            results = self._model(
                frame,
                conf=self.cfg.detection.conf_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return self._last_boxes

        boxes: List[List[float]] = []
        obstacle_ids = set(self.cfg.detection.obstacle_classes)
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls in obstacle_ids:
                boxes.append(box.xyxy[0].tolist())

        self._last_boxes = boxes
        return boxes

    # ------------------------------------------------------------------
    # Avoidance decision
    # ------------------------------------------------------------------

    def get_avoidance_action(
        self,
        obstacles: List[List[float]],
        frame_width: int,
    ) -> Tuple[str, float]:
        """Decide what avoidance action to take given detected obstacles.

        Parameters
        ----------
        obstacles:
            List of ``[x1, y1, x2, y2]`` bounding boxes.
        frame_width:
            Width of the full frame in pixels (used to detect left/right bias).

        Returns
        -------
        Tuple[str, float]
            A tuple ``(action, steer_override)`` where *action* is one of
            ``"none"``, ``"brake_hard"``, or ``"swerve_left"``/``"swerve_right"``,
            and *steer_override* is the vJoy-normalised steering value to apply
            (only meaningful when action starts with ``"swerve"``).
        """
        if not obstacles:
            return "none", 0.0

        dcfg = self.cfg.detection
        half = frame_width / 2
        swerve = dcfg.swerve_amount

        for x1, _y1, x2, _y2 in obstacles:
            width_px = x2 - x1
            center_x = (x1 + x2) / 2
            # y-coordinates (_y1, _y2) are not used here: proximity is inferred
            # from bounding-box width (larger width ≈ closer obstacle) rather
            # than from absolute vertical position, which varies with road slope.

            if width_px >= dcfg.brake_width_px:
                # Obstacle is large — brake hard and steer around it
                if center_x < half:
                    # Obstacle on left → steer right
                    return "swerve_right", swerve
                else:
                    # Obstacle on right → steer left
                    return "swerve_left", -swerve

        return "none", 0.0
