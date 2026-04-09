"""Speed-limit sign detector using OpenCV computer vision.

Detection pipeline
------------------
1. Hough circle transform locates candidate circular regions within a
   configurable region-of-interest (ROI) of the captured frame.
2. Each candidate is scored by:
   - Red pixel density in the annular border region (HSV-based).
   - White pixel density inside the circle (speed signs have white backgrounds).
3. Optionally, pytesseract OCR reads the digit(s) from the interior of the
   best candidate to return the exact speed limit value.

The detector is intentionally conservative: ``confidence_threshold`` filters
out weak matches so the driver never adapts to a false positive.

If pytesseract is not installed the detector still works; it reports only
``limit_kph=None`` with a confidence score derived purely from shape/colour
cues (useful for "a sign is present" awareness without the exact value).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import ETS2Config

logger = logging.getLogger(__name__)


@dataclass
class SpeedLimitResult:
    """Outcome of one speed-limit detection pass.

    Attributes
    ----------
    limit_kph:
        Detected speed limit in km/h, or ``None`` if the sign was not found or
        the OCR step could not be completed.
    confidence:
        Detection confidence in ``[0.0, 1.0]``.  Values above
        :attr:`~ets2_driver.config.SpeedLimitConfig.confidence_threshold` are
        accepted; lower values are discarded.
    bounding_circle:
        ``(x, y, radius)`` of the detected circle within the ROI, or ``None``.
    raw_ocr_text:
        Raw string returned by pytesseract (empty when OCR is unavailable).
    """

    limit_kph: Optional[int]
    confidence: float
    bounding_circle: Optional[Tuple[int, int, int]]
    raw_ocr_text: str


class SpeedLimitDetector:
    """Detects and reads speed-limit signs in ETS2 game frames.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.
    """

    #: km/h values that appear on ETS2 speed limit signs.
    #: Based on standard European road speed limits as rendered in ETS2.
    #: Used to snap imprecise OCR readings to the nearest realistic value.
    KNOWN_LIMITS = frozenset({20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130})

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._last_result: SpeedLimitResult = SpeedLimitResult(
            limit_kph=None, confidence=0.0, bounding_circle=None, raw_ocr_text=""
        )
        self._ocr_available: bool = False

        try:
            import pytesseract  # type: ignore  # noqa: F401
            self._ocr_available = True
            logger.info("SpeedLimitDetector: pytesseract OCR available.")
        except ImportError:
            logger.info(
                "SpeedLimitDetector: pytesseract not installed — OCR disabled. "
                "Speed limits will not be read from signs."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> SpeedLimitResult:
        """Detect a speed-limit sign in *frame* and return the result.

        Parameters
        ----------
        frame:
            Full BGR frame from :class:`~ets2_driver.vision.VisionSystem`.

        Returns
        -------
        SpeedLimitResult
            Best detection result.  ``confidence == 0.0`` means no sign found.
        """
        slcfg = self.cfg.speed_limit

        # --- Crop ROI ---
        roi = frame[
            slcfg.roi_top : slcfg.roi_bottom,
            slcfg.roi_left : slcfg.roi_right,
        ]
        if roi.size == 0:
            return self._no_result()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=slcfg.min_dist_px,
            param1=slcfg.hough_param1,
            param2=slcfg.hough_param2,
            minRadius=slcfg.min_radius_px,
            maxRadius=slcfg.max_radius_px,
        )

        if circles is None:
            return self._no_result()

        circles = np.round(circles[0, :]).astype(int)

        best: Optional[SpeedLimitResult] = None
        best_conf: float = 0.0

        for cx, cy, r in circles:
            result = self._evaluate_circle(roi, int(cx), int(cy), int(r))
            if result.confidence > best_conf:
                best_conf = result.confidence
                best = result

        if best is None or best_conf < slcfg.confidence_threshold:
            return self._no_result()

        self._last_result = best
        return best

    @property
    def last_result(self) -> SpeedLimitResult:
        """Most recent detection result (cached between calls)."""
        return self._last_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _no_result(self) -> SpeedLimitResult:
        empty = SpeedLimitResult(
            limit_kph=None, confidence=0.0, bounding_circle=None, raw_ocr_text=""
        )
        self._last_result = empty
        return empty

    def _evaluate_circle(
        self, roi: np.ndarray, cx: int, cy: int, r: int
    ) -> SpeedLimitResult:
        """Score a single candidate circle as a speed-limit sign."""
        h, w = roi.shape[:2]
        r_inner = max(1, int(r * 0.72))

        # Masks
        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_outer, (cx, cy), r, 255, -1)
        cv2.circle(mask_inner, (cx, cy), r_inner, 255, -1)
        ring_mask = cv2.subtract(mask_outer, mask_inner)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red border check (hue wraps at 0/180)
        red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)

        ring_total = int(np.count_nonzero(ring_mask))
        if ring_total == 0:
            return SpeedLimitResult(
                limit_kph=None, confidence=0.0, bounding_circle=None, raw_ocr_text=""
            )

        red_in_ring = int(np.count_nonzero(cv2.bitwise_and(red_mask, ring_mask)))
        red_ratio = red_in_ring / ring_total

        if red_ratio < 0.12:
            return SpeedLimitResult(
                limit_kph=None, confidence=0.0, bounding_circle=None, raw_ocr_text=""
            )

        # White interior check
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
        fill_total = int(np.count_nonzero(mask_inner))
        white_in_fill = int(
            np.count_nonzero(cv2.bitwise_and(white_mask, mask_inner))
        ) if fill_total > 0 else 0
        white_ratio = white_in_fill / fill_total if fill_total > 0 else 0.0

        confidence = (
            min(1.0, red_ratio / 0.30) * 0.5 + min(1.0, white_ratio / 0.50) * 0.5
        )

        # Optional OCR
        limit_kph: Optional[int] = None
        raw_text = ""
        if self._ocr_available and confidence > 0.3:
            limit_kph, raw_text = self._run_ocr(roi, cx, cy, r_inner)
            if limit_kph is not None:
                confidence = min(1.0, confidence + 0.3)

        return SpeedLimitResult(
            limit_kph=limit_kph,
            confidence=round(confidence, 3),
            bounding_circle=(cx, cy, r),
            raw_ocr_text=raw_text,
        )

    def _run_ocr(
        self, roi: np.ndarray, cx: int, cy: int, r: int
    ) -> Tuple[Optional[int], str]:
        """Crop the sign interior and run pytesseract to read the number."""
        try:
            import pytesseract  # type: ignore

            h, w = roi.shape[:2]
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            crop = roi[y1:y2, x1:x2]
            if crop.size == 0:
                return None, ""

            # Upscale small crops for better OCR accuracy
            diam = max(x2 - x1, y2 - y1)
            scale = max(1, 64 // (diam + 1) + 1)
            crop_up = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            cfg_str = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
            raw = pytesseract.image_to_string(thresh, config=cfg_str).strip()
            nums = re.findall(r"\d+", raw)
            if not nums:
                return None, raw

            val = int(nums[0])
            # Snap to nearest known ETS2 speed limit if within ±5 km/h
            closest = min(self.KNOWN_LIMITS, key=lambda x: abs(x - val))
            if abs(closest - val) <= 5:
                return closest, raw
            return val, raw

        except Exception as exc:
            logger.debug("OCR failed: %s", exc)
            return None, ""
