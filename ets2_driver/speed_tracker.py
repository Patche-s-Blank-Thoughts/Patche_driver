"""HUD OCR-based speed capture and velocity estimation for ETS2 AI Driver.

The ETS2 speedometer renders the current vehicle speed as large digits in
km/h on the HUD.  This module crops a configurable region of the captured
frame, runs pytesseract OCR to read the numeric value, applies exponential
smoothing to suppress noise, and tracks acceleration/deceleration trends
between frames.

If pytesseract is not installed the tracker silently returns ``0.0`` for all
frames, so the rest of the system degrades gracefully — the adaptive PID
scheduler will simply use the zero-speed profile (which mirrors the original
static gains).

Usage::

    tracker = SpeedTracker(cfg)
    # Inside the main driving loop:
    speed_kph = tracker.update(frame)   # call once per tick
    print(tracker.current_speed, tracker.velocity_trend)
"""

from __future__ import annotations

import collections
import logging
import re
from typing import Deque, Optional

import cv2
import numpy as np

from .config import ETS2Config

logger = logging.getLogger(__name__)


class SpeedTracker:
    """HUD OCR speed reader with exponential smoothing and velocity estimation.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.

    Notes
    -----
    OCR is intentionally throttled — only run every
    :attr:`~ets2_driver.config.SpeedTrackingConfig.ocr_every_n_frames` frames
    to keep CPU load low.  The smoothed speed value persists between OCR
    frames so the reported speed is always up-to-date.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._ocr_available: bool = False
        self._smoothed_speed: float = 0.0
        self._frame_counter: int = 0
        self._velocity_history: Deque[float] = collections.deque(
            maxlen=max(1, cfg.speed_tracking.velocity_window)
        )

        # Public telemetry — updated each call to :meth:`update`
        self.current_speed: float = 0.0   # smoothed speed in km/h
        self.velocity: float = 0.0        # smoothed acceleration (km/h per frame)
        self.velocity_trend: str = "steady"  # "accelerating" / "decelerating" / "steady"

        try:
            import pytesseract  # type: ignore  # noqa: F401
            self._ocr_available = True
            logger.info("SpeedTracker: pytesseract OCR available.")
        except ImportError:
            logger.info(
                "SpeedTracker: pytesseract not installed — speed OCR disabled. "
                "Speed will be reported as 0.0 km/h."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> float:
        """Read the speedometer from *frame* and update telemetry.

        Parameters
        ----------
        frame:
            Full BGR frame from :class:`~ets2_driver.vision.VisionSystem`.

        Returns
        -------
        float
            Current smoothed speed in km/h.
        """
        stcfg = self.cfg.speed_tracking
        if not stcfg.enabled:
            return self.current_speed

        self._frame_counter += 1

        # Only run OCR every N frames to reduce CPU load
        if self._frame_counter % stcfg.ocr_every_n_frames == 0:
            raw_speed = self._ocr_speed(frame)
            if raw_speed is not None:
                alpha = stcfg.smoothing_alpha
                self._smoothed_speed = (
                    alpha * self._smoothed_speed + (1.0 - alpha) * raw_speed
                )

        # Track per-frame velocity delta
        prev = self.current_speed
        self.current_speed = self._smoothed_speed
        delta = self.current_speed - prev
        self._velocity_history.append(delta)

        # Smoothed velocity (mean of recent deltas)
        self.velocity = (
            sum(self._velocity_history) / len(self._velocity_history)
            if self._velocity_history
            else 0.0
        )

        # Trend classification
        if self.velocity > 0.5:
            self.velocity_trend = "accelerating"
        elif self.velocity < -0.5:
            self.velocity_trend = "decelerating"
        else:
            self.velocity_trend = "steady"

        return self.current_speed

    # ------------------------------------------------------------------
    # Internal OCR helpers
    # ------------------------------------------------------------------

    def _ocr_speed(self, frame: np.ndarray) -> Optional[float]:
        """Crop the speedometer ROI from *frame* and return the raw reading."""
        stcfg = self.cfg.speed_tracking
        h, w = frame.shape[:2]

        # Clamp ROI coordinates to frame bounds
        top    = max(0, min(stcfg.roi_top,    h))
        bottom = max(0, min(stcfg.roi_bottom, h))
        left   = max(0, min(stcfg.roi_left,   w))
        right  = max(0, min(stcfg.roi_right,  w))

        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            return None

        return self._parse_speed_from_roi(roi)

    def _parse_speed_from_roi(self, roi: np.ndarray) -> Optional[float]:
        """Run OCR on the speedometer crop and return a numeric km/h value."""
        if not self._ocr_available:
            return None

        try:
            import pytesseract  # type: ignore

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Upscale small crops for better OCR accuracy (target 64 px tall)
            h, _w = gray.shape[:2]
            target_h = 64
            if h < target_h:
                scale = target_h / max(h, 1)
                gray = cv2.resize(
                    gray, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            # CLAHE contrast normalisation
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

            # Multiple thresholding strategies
            _, otsu = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, otsu_inv = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            tess_configs = [
                "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789",
                "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789",
            ]

            max_speed = self.cfg.speed_tracking.max_speed_kph
            for img in (otsu, otsu_inv):
                for cfg_str in tess_configs:
                    try:
                        raw = pytesseract.image_to_string(img, config=cfg_str).strip()
                    except Exception:
                        continue
                    nums = re.findall(r"\d+", raw)
                    if nums:
                        val = float(nums[0])
                        if 0.0 <= val <= max_speed:
                            return val

        except Exception as exc:
            logger.debug("Speed OCR failed: %s", exc)

        return None
