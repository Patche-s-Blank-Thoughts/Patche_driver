"""Vision subsystem — screen capture, lane detection, and GPS reading.

All functions are pure (no side-effects on global state) and accept the
active :class:`~ets2_driver.config.ETS2Config` so they are easily testable
without a live game window.
"""

from __future__ import annotations

import collections
import logging
from typing import Deque, List, Optional, Tuple

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

    # Number of frames kept in the GPS-direction rolling median buffer.
    # A window of 5 frames at 30 FPS gives ~167 ms of smoothing — enough to
    # suppress single-frame mini-map redraw artefacts without lagging the signal.
    _GPS_MEDIAN_WINDOW: int = 5

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._sct = mss.mss()

        # Debug / telemetry — set on each call to detect_lane_center()
        self.last_lane_confidence: float = 0.0
        self.last_lane_candidates: List[int] = []

        # Rolling buffer for GPS direction readings; median over the last
        # _GPS_MEDIAN_WINDOW frames reduces noise from momentary map flicker.
        self._gps_history: Deque[int] = collections.deque(maxlen=self._GPS_MEDIAN_WINDOW)

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

    def _lane_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the white-pixel mask and the ROI slice.

        Parameters
        ----------
        frame:
            Full BGR frame.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(roi_mask, full_mask)`` — both are binary masks with the same
            spatial extent as *frame*.  ``roi_mask`` has pixels outside the
            region-of-interest zeroed out.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.cfg.lane.lower_white)
        upper = np.array(self.cfg.lane.upper_white)
        full_mask = cv2.inRange(hsv, lower, upper)

        roi_top = int(frame.shape[0] * self.cfg.lane.roi_top_fraction)
        roi_mask = np.zeros_like(full_mask)
        roi_mask[roi_top:, :] = full_mask[roi_top:, :]
        return roi_mask, full_mask

    def detect_lane_center(self, frame: np.ndarray) -> int:
        """Return the horizontal pixel position of the detected lane centre.

        The method converts the frame to HSV colour space, isolates white lane
        markings, restricts the search to the lower portion of the image (the
        Region Of Interest), and returns the mean column index of matching
        pixels.  Falls back to the frame centre when no lane markings are
        visible.

        As a side-effect this method updates:
        * :attr:`last_lane_confidence` — fraction of ROI pixels that are lane
          markings (higher is more reliable).
        * :attr:`last_lane_candidates` — up to three candidate centres
          (left-third, centre-third, right-third of the ROI).

        Parameters
        ----------
        frame:
            Full BGR frame from :meth:`get_frame`.

        Returns
        -------
        int
            Horizontal pixel index of the estimated lane centre.
        """
        roi_mask, _ = self._lane_mask(frame)
        roi_top = int(frame.shape[0] * self.cfg.lane.roi_top_fraction)
        roi = roi_mask[roi_top:, :]

        ys, xs = np.where(roi > 0)

        # Confidence: fraction of ROI pixels that matched
        roi_pixels = roi.size
        self.last_lane_confidence = float(xs.size / max(1, roi_pixels))

        # Candidate centres split into left / centre / right thirds
        w = roi.shape[1]
        candidates: List[int] = []
        for lo, hi in [(0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)]:
            segment_xs = xs[(xs >= lo) & (xs < hi)]
            if segment_xs.size > 0:
                candidates.append(int(np.mean(segment_xs)))
        self.last_lane_candidates = candidates

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

        Bounds-checks all coordinates against the actual frame dimensions so a
        misconfigured GPS region never causes an empty or out-of-range crop.
        Falls back to the full frame when the configured region is invalid.

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
        h, w = frame.shape[:2]

        top    = max(0, min(g.top,    h - 1))
        bottom = max(top + 1, min(g.bottom, h))
        left   = max(0, min(g.left,   w - 1))
        right  = max(left + 1, min(g.right, w))

        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            logger.warning(
                "GPS crop is empty (cfg: top=%d bottom=%d left=%d right=%d, "
                "frame: %dx%d) — falling back to full frame.",
                g.top, g.bottom, g.left, g.right, w, h,
            )
            return frame
        return crop

    def gps_direction(self, gps: np.ndarray) -> int:
        """Estimate lateral steering bias from the GPS route line.

        Detects the red route line drawn on the ETS2 mini-map and returns the
        signed offset of its centroid from the mini-map centre.  A positive
        value means the route continues to the right.

        A rolling median filter over the last few frames suppresses noise from
        momentary map redraw artefacts and produces a more reliable signal.

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
            # No route pixels visible — keep the last known reading
            if self._gps_history:
                return int(np.median(list(self._gps_history)))
            return 0

        # Use median x-position of route pixels to resist outlier noise
        raw_offset = int(np.median(xs) - gps.shape[1] / 2)

        # Append to rolling history and return the smoothed median
        self._gps_history.append(raw_offset)
        return int(np.median(list(self._gps_history)))

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
        lane_candidates: Optional[List[int]] = None,
        obstacle_sides: Optional[List[str]] = None,
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
        lane_candidates:
            Optional list of candidate lane-centre x positions (drawn as
            thin vertical lines).
        obstacle_sides:
            Optional per-obstacle side labels (``"left"``/``"center"``/
            ``"right"``).  When provided, each box is labelled with its zone.

        Returns
        -------
        np.ndarray
            Annotated BGR frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        screen_cx = w // 2

        # Lane-boundary safe zone (±50 px from screen centre)
        safe_left = screen_cx - 50
        safe_right = screen_cx + 50
        cv2.line(vis, (safe_left, h // 2), (safe_left, h), (0, 200, 100), 1)
        cv2.line(vis, (safe_right, h // 2), (safe_right, h), (0, 200, 100), 1)

        # Candidate lane centres (thin cyan lines)
        if lane_candidates:
            for cx in lane_candidates:
                cv2.line(vis, (cx, h // 2), (cx, h), (255, 200, 0), 1)

        # Lane-centre vertical line
        cv2.line(vis, (lane_center, h // 2), (lane_center, h),
                 (0, 255, 0), 2)
        # Screen-centre reference
        cv2.line(vis, (screen_cx, 0), (screen_cx, h), (255, 0, 0), 1)

        # Obstacle bounding boxes with side labels
        side_colors = {"left": (0, 165, 255), "center": (0, 0, 255), "right": (255, 0, 255)}
        for i, box in enumerate(obstacles):
            x1, y1, x2, y2 = (int(v) for v in box)
            side = obstacle_sides[i] if obstacle_sides and i < len(obstacle_sides) else "center"
            color = side_colors.get(side, (0, 0, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            # Side label above the box
            cv2.putText(
                vis, side.upper(),
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        # Lane confidence
        cv2.putText(
            vis, f"err={error:.1f}  conf={self.last_lane_confidence:.3f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        return vis

    def draw_gps_debug(self, gps: np.ndarray) -> np.ndarray:
        """Overlay debug information onto the GPS mini-map crop.

        Draws the detected red-route mask as a cyan tint and marks the
        estimated route centroid with a vertical line.

        Parameters
        ----------
        gps:
            Cropped GPS mini-map BGR image from :meth:`crop_gps`.

        Returns
        -------
        np.ndarray
            Annotated BGR image (copy of *gps* with overlays).
        """
        if gps is None or gps.size == 0:
            return gps

        vis = gps.copy()
        hsv = cv2.cvtColor(gps, cv2.COLOR_BGR2HSV)
        g = self.cfg.gps

        mask1 = cv2.inRange(hsv, np.array(g.lower_red1), np.array(g.upper_red1))
        mask2 = cv2.inRange(hsv, np.array(g.lower_red2), np.array(g.upper_red2))
        mask = cv2.bitwise_or(mask1, mask2)

        # Tint detected route pixels cyan
        vis[mask > 0] = (255, 255, 0)

        # Draw centroid line and GPS offset label
        h_gps, w_gps = vis.shape[:2]
        _, xs = np.where(mask > 0)
        if xs.size > 0:
            cx = int(np.median(xs))
            cv2.line(vis, (cx, 0), (cx, h_gps), (0, 255, 0), 1)
            offset = cx - w_gps // 2
            cv2.putText(
                vis, f"gps={offset:+d}px",
                (2, h_gps - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
            )

        # Centre reference line
        cv2.line(vis, (w_gps // 2, 0), (w_gps // 2, h_gps), (255, 0, 0), 1)
        return vis
