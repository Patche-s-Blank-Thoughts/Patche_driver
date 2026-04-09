"""Vision subsystem — screen capture, lane detection, and GPS reading.

All functions are pure (no side-effects on global state) and accept the
active :class:`~ets2_driver.config.ETS2Config` so they are easily testable
without a live game window.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

from .config import ETS2Config

logger = logging.getLogger(__name__)


class VisionSystem:
    """Handles all image-acquisition and computer-vision tasks.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._sct = mss.mss()
        logger.info("VisionSystem initialised — monitor region: %s",
                    cfg.capture.as_dict)

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """Capture one frame from the configured screen region.

        Returns
        -------
        np.ndarray
            BGR image with shape ``(height, width, 3)``.
        """
        raw = self._sct.grab(self.cfg.capture.as_dict)
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ------------------------------------------------------------------
    # Lane / road detection
    # ------------------------------------------------------------------

    def detect_lane_center(self, frame: np.ndarray) -> int:
        """Return the horizontal pixel position of the detected lane centre.

        The method converts the frame to HSV colour space, isolates white lane
        markings, restricts the search to the lower portion of the image (the
        Region Of Interest), and returns the mean column index of matching
        pixels.  Falls back to the frame centre when no lane markings are
        visible.

        Parameters
        ----------
        frame:
            Full BGR frame from :meth:`get_frame`.

        Returns
        -------
        int
            Horizontal pixel index of the estimated lane centre.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.cfg.lane.lower_white)
        upper = np.array(self.cfg.lane.upper_white)
        mask = cv2.inRange(hsv, lower, upper)

        # Region-of-interest: lower portion of the frame
        roi_top = int(frame.shape[0] * self.cfg.lane.roi_top_fraction)
        roi = mask[roi_top:, :]

        ys, xs = np.where(roi > 0)
        if xs.size == 0:
            logger.debug("No lane pixels found — using frame centre")
            return frame.shape[1] // 2

        return int(np.mean(xs))

    def get_steering_error(self, frame: np.ndarray) -> int:
        """Return the signed horizontal offset of the lane centre from screen centre.

        A positive value means the lane centre is to the *right* of the screen
        centre (steer right); a negative value means steer left.

        Parameters
        ----------
        frame:
            Full BGR frame from :meth:`get_frame`.

        Returns
        -------
        int
            Signed pixel error.
        """
        lane_center = self.detect_lane_center(frame)
        screen_center = frame.shape[1] // 2
        return lane_center - screen_center

    # ------------------------------------------------------------------
    # GPS mini-map reading
    # ------------------------------------------------------------------

    def crop_gps(self, frame: np.ndarray) -> np.ndarray:
        """Crop the route-advisor mini-map from a full frame.

        Parameters
        ----------
        frame:
            Full BGR frame.

        Returns
        -------
        np.ndarray
            Cropped BGR sub-image containing the GPS mini-map.
        """
        g = self.cfg.gps
        return frame[g.top:g.bottom, g.left:g.right]

    def gps_direction(self, gps: np.ndarray) -> int:
        """Estimate lateral steering bias from the GPS route line.

        Detects the red route line drawn on the ETS2 mini-map and returns the
        signed offset of its centroid from the mini-map centre.  A positive
        value means the route continues to the right.

        Parameters
        ----------
        gps:
            Cropped GPS mini-map image (BGR).

        Returns
        -------
        int
            Signed pixel offset (0 when the route line is not visible).
        """
        if gps is None or gps.size == 0:
            return 0

        hsv = cv2.cvtColor(gps, cv2.COLOR_BGR2HSV)
        g = self.cfg.gps

        # Red wraps around the hue wheel — combine two ranges
        mask1 = cv2.inRange(
            hsv, np.array(g.lower_red1), np.array(g.upper_red1)
        )
        mask2 = cv2.inRange(
            hsv, np.array(g.lower_red2), np.array(g.upper_red2)
        )
        mask = cv2.bitwise_or(mask1, mask2)

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return 0

        return int(np.mean(xs) - gps.shape[1] / 2)

    def get_combined_error(self, frame: np.ndarray) -> float:
        """Blend lane-follow error with GPS guidance error.

        The GPS component is weighted by
        :attr:`~ets2_driver.config.GpsConfig.blend_weight` so it gently pulls
        the truck toward the planned route without overwhelming the immediate
        lane-following signal.

        Parameters
        ----------
        frame:
            Full BGR frame from :meth:`get_frame`.

        Returns
        -------
        float
            Weighted composite steering error (pixels).
        """
        lane_error = self.get_steering_error(frame)
        gps_img = self.crop_gps(frame)
        gps_error = self.gps_direction(gps_img)
        w = self.cfg.gps.blend_weight
        return lane_error * (1.0 - w) + gps_error * w

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def draw_debug(
        self,
        frame: np.ndarray,
        lane_center: int,
        error: float,
        obstacles: list,
    ) -> np.ndarray:
        """Overlay debug information onto a frame for live monitoring.

        Parameters
        ----------
        frame:
            Original BGR frame (will be copied before drawing).
        lane_center:
            Horizontal pixel index of the detected lane centre.
        error:
            Current composite steering error.
        obstacles:
            List of ``[x1, y1, x2, y2]`` bounding boxes from the detector.

        Returns
        -------
        np.ndarray
            Annotated BGR frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        screen_cx = w // 2

        # Lane-centre vertical line
        cv2.line(vis, (lane_center, h // 2), (lane_center, h),
                 (0, 255, 0), 2)
        # Screen-centre reference
        cv2.line(vis, (screen_cx, 0), (screen_cx, h), (255, 0, 0), 1)

        # Obstacle bounding boxes
        for box in obstacles:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Error text
        cv2.putText(
            vis, f"err={error:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        return vis
