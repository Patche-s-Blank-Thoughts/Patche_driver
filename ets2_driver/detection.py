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

        # Debug / telemetry — set by get_avoidance_action()
        self.last_obstacle_sides: List[str] = []
        self.last_obstacle_distances: List[float] = []
        self.last_lane_target: str = "center"

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
    # Obstacle side classification
    # ------------------------------------------------------------------

    def classify_obstacle_side(self, x1: float, x2: float, frame_width: int) -> str:
        """Classify an obstacle as ``"left"``, ``"center"``, or ``"right"``.

        The frame is divided into three equal zones.  The obstacle's
        horizontal centre determines its zone.  This matches the
        :attr:`~ets2_driver.config.DetectionConfig.center_zone_fraction`
        setting.

        Parameters
        ----------
        x1, x2:
            Horizontal bounding-box edges in pixels.
        frame_width:
            Width of the full frame.

        Returns
        -------
        str
            ``"left"``, ``"center"``, or ``"right"``.
        """
        center_x = (x1 + x2) / 2.0
        zone = self.cfg.detection.center_zone_fraction
        left_bound = frame_width * (0.5 - zone / 2.0)
        right_bound = frame_width * (0.5 + zone / 2.0)
        if center_x < left_bound:
            return "left"
        if center_x > right_bound:
            return "right"
        return "center"

    # ------------------------------------------------------------------
    # Avoidance decision
    # ------------------------------------------------------------------

    def get_avoidance_action(
        self,
        obstacles: List[List[float]],
        frame_width: int,
    ) -> Tuple[str, float]:
        """Decide what avoidance action to take given detected obstacles.

        The decision uses three horizontal zones (left / center / right).
        When the path ahead is blocked the bot tries to move into the
        clearest adjacent lane.

        Parameters
        ----------
        obstacles:
            List of ``[x1, y1, x2, y2]`` bounding boxes.
        frame_width:
            Width of the full frame in pixels (used for zone classification).

        Returns
        -------
        Tuple[str, float]
            A tuple ``(action, steer_override)`` where *action* is one of
            ``"none"``, ``"brake_hard"``, ``"swerve_left"``, or
            ``"swerve_right"``, and *steer_override* is the vJoy-normalised
            steering value to apply (only meaningful when action starts with
            ``"swerve"``).
        """
        # Reset debug state
        self.last_obstacle_sides = []
        self.last_obstacle_distances = []

        if not obstacles:
            return "none", 0.0

        dcfg = self.cfg.detection
        swerve = dcfg.swerve_amount

        # Classify all large obstacles by zone
        zones_blocked: set[str] = set()
        for x1, _y1, x2, _y2 in obstacles:
            width_px = x2 - x1
            side = self.classify_obstacle_side(x1, x2, frame_width)
            # Normalised width as a distance proxy (larger = closer)
            norm_dist = width_px / max(1, frame_width)
            self.last_obstacle_sides.append(side)
            self.last_obstacle_distances.append(norm_dist)

            if width_px >= dcfg.brake_width_px:
                zones_blocked.add(side)
                logger.debug(
                    "Large obstacle zone=%s width=%.0fpx norm=%.3f",
                    side, width_px, norm_dist,
                )

        if not zones_blocked:
            return "none", 0.0

        # Lane-switch logic: try to move away from the blocked zone
        if "center" in zones_blocked:
            # Path directly ahead blocked — pick the clearest adjacent lane
            if "left" not in zones_blocked:
                logger.info("Path blocked center → switching LEFT")
                self.last_lane_target = "left"
                return "swerve_left", -swerve
            if "right" not in zones_blocked:
                logger.info("Path blocked center → switching RIGHT")
                self.last_lane_target = "right"
                return "swerve_right", swerve
            # Both sides blocked — brake hard
            logger.warning("All lanes blocked — braking hard")
            return "brake_hard", 0.0

        if "left" in zones_blocked and "right" not in zones_blocked:
            # Left only blocked → stay right
            self.last_lane_target = "right"
            return "swerve_right", swerve

        if "right" in zones_blocked and "left" not in zones_blocked:
            # Right only blocked → stay left
            self.last_lane_target = "left"
            return "swerve_left", -swerve

        # Both flanks but not centre — continue but log
        logger.debug("Flanks blocked but centre clear — no avoidance needed")
        return "none", 0.0
