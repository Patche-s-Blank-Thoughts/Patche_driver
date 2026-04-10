"""Parking-lot detection and road-escape planner.

When the ETS2 truck starts in a parking lot (common at the beginning of a
session) it needs to find the exit and merge onto the main road before normal
lane-following can take over.

This module provides :class:`ParkingLotPlanner`, which:

1. Detects whether the current view shows a parking lot (low lane-marking
   density) or a proper road.
2. If a parking lot is detected, builds a multi-step escape plan — either by
   asking the LLM for a plan or by using a safe rule-based default.
3. Executes the plan step-by-step, overriding the normal control loop.
4. Hands back to normal lane-following once road markings are detected.

The planner integrates with the existing
:class:`~ets2_driver.llm_planner.LLMPlanner` infrastructure so no additional
model dependencies are required.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .gears import GearShifter

import cv2
import numpy as np

from .config import ETS2Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Navigation state constants
# ---------------------------------------------------------------------------

class ParkingState:
    """Named constants for the planner state machine."""
    UNKNOWN        = "UNKNOWN"
    IN_PARKING_LOT = "IN_PARKING_LOT"
    EXITING        = "EXITING"
    ON_ROAD        = "ON_ROAD"


# ---------------------------------------------------------------------------
# Individual navigation step
# ---------------------------------------------------------------------------

class NavigationStep:
    """One timed action in a parking-escape plan.

    Parameters
    ----------
    action:
        One of ``FORWARD``, ``TURN_LEFT``, ``TURN_RIGHT``, ``REVERSE``,
        ``STOP``.
    duration_s:
        How long (seconds) to execute this action before moving to the next.
    description:
        Optional human-readable label for logging.
    """

    def __init__(
        self,
        action: str,
        duration_s: float = 2.0,
        description: str = "",
    ) -> None:
        self.action = action.upper()
        self.duration_s = duration_s
        self.description = description
        self._start_time: Optional[float] = None

    @property
    def is_complete(self) -> bool:
        """True once the step's duration has elapsed since :meth:`start`."""
        if self._start_time is None:
            return False
        return time.monotonic() - self._start_time >= self.duration_s

    def start(self) -> None:
        """Mark the step as started (records current time)."""
        self._start_time = time.monotonic()

    def __repr__(self) -> str:
        return (
            f"NavigationStep(action={self.action!r}, "
            f"duration_s={self.duration_s:.1f}, "
            f"desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ParkingLotPlanner:
    """Detect a parking-lot start and plan an escape to the main road.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.

    Usage
    -----
    Call :meth:`update` on every frame (or every few frames).  It returns
    ``None`` when the planner is inactive (let normal lane-following run) or
    a ``(steer, throttle, brake)`` tuple when overriding the control loop.

    Once the planner detects road markings it transitions to
    :attr:`ParkingState.ON_ROAD` and returns ``None`` for all subsequent
    calls.
    """

    # How often (seconds) to re-evaluate the road/parking-lot state
    _CHECK_INTERVAL_S: float = 3.0

    # Fraction of ROI pixels that must be white-ish lane markings before we
    # consider the view to be "on a road".  Parking lots have very few or no
    # painted lane lines.
    _ROAD_DENSITY_THRESHOLD: float = 0.012

    # Stuck detection: if speed stays below this for _STUCK_FRAMES_THRESHOLD
    # consecutive frames while executing a forward-type action, the planner
    # inserts a REVERSE escape sequence to clear the obstacle.
    _STUCK_SPEED_THRESHOLD_KPH: float = 1.0
    _STUCK_FRAMES_THRESHOLD: int = 30        # ≈ 1 s at 30 fps
    _STUCK_INSERT_COOLDOWN_S: float = 8.0   # minimum gap between insertions

    def __init__(self, cfg: ETS2Config, gear_shifter: "Optional[GearShifter]" = None) -> None:
        self.cfg = cfg
        self._gear_shifter = gear_shifter
        self._state: str = ParkingState.UNKNOWN
        self._last_check: float = 0.0
        self._nav_plan: List[NavigationStep] = []
        self._current_step_idx: int = 0
        self._agent = None
        self._reverse_engaged: bool = False
        self._stuck_frames: int = 0
        self._last_stuck_insert: float = 0.0

        logger.info("ParkingLotPlanner initialised.")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current parking-state label (read-only)."""
        return self._state

    @property
    def is_active(self) -> bool:
        """True while the planner is overriding the control loop."""
        return self._state in (ParkingState.IN_PARKING_LOT, ParkingState.EXITING)

    # ------------------------------------------------------------------
    # Road / parking-lot detection
    # ------------------------------------------------------------------

    def detect_road_presence(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Check whether *frame* shows a road with visible lane markings.

        Uses the same HSV white-pixel thresholds as
        :class:`~ets2_driver.vision.VisionSystem` so the two modules stay in
        sync without duplicating configuration.

        Parameters
        ----------
        frame:
            Full BGR frame captured from the game window.

        Returns
        -------
        Tuple[bool, float]
            ``(on_road, density)`` — *density* is the fraction of ROI pixels
            that matched the lane-marking colour (higher = more road markings).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.cfg.lane.lower_white)
        upper = np.array(self.cfg.lane.upper_white)
        mask = cv2.inRange(hsv, lower, upper)

        h = frame.shape[0]
        roi_top = int(h * self.cfg.lane.roi_top_fraction)
        roi = mask[roi_top:, :]

        density = float(np.count_nonzero(roi)) / max(1, roi.size)
        on_road = density >= self._ROAD_DENSITY_THRESHOLD
        return on_road, density

    # ------------------------------------------------------------------
    # LLM-backed plan generation
    # ------------------------------------------------------------------

    def _ensure_agent(self) -> None:
        """Lazily load the shared LLM agent (same pattern as LLMPlanner)."""
        if self._agent is not None or not self.cfg.llm.enabled:
            return
        try:
            import config as repo_config  # type: ignore
            original = repo_config.MODEL_NAME
            repo_config.MODEL_NAME = self.cfg.llm.model_name
            from agent import Agent  # type: ignore
            self._agent = Agent()
            repo_config.MODEL_NAME = original
            logger.info("ParkingLotPlanner: LLM agent loaded (%s).", self.cfg.llm.model_name)
        except Exception as exc:
            logger.warning("ParkingLotPlanner: could not load LLM agent: %s", exc)

    def _llm_plan_escape(
        self, lane_density: float, gps_error: float
    ) -> List[NavigationStep]:
        """Ask the LLM for a parking-escape plan.

        Falls back silently to :meth:`_default_plan` when the LLM is
        unavailable or returns an unparseable response.

        Parameters
        ----------
        lane_density:
            Road-marking pixel density (low → parking lot).
        gps_error:
            Current GPS lateral offset in pixels.

        Returns
        -------
        List[NavigationStep]
            Ordered list of timed manoeuvre steps.
        """
        if self.cfg.llm.enabled:
            self._ensure_agent()

        if self._agent is not None:
            try:
                from models.chat import ChatMessage  # type: ignore
                prompt = (
                    "I am driving a truck in Euro Truck Simulator 2. "
                    f"Road lane-marking density is {lane_density:.4f} (very low = parking lot). "
                    f"GPS lateral offset: {gps_error:.0f}px. "
                    "I appear to be in a parking lot and need to reach the main road. "
                    "Give me a step-by-step escape plan using only these actions: "
                    "FORWARD, TURN_LEFT, TURN_RIGHT, REVERSE. "
                    "Format each step as ACTION:duration_in_seconds, one per line. "
                    "Example:\nFORWARD:3\nTURN_RIGHT:2\nFORWARD:5"
                )
                messages = [ChatMessage(role="user", content=prompt)]
                response = asyncio.run(self._agent.chat(messages))
                steps = self._parse_llm_steps(response.message)
                if steps:
                    logger.info(
                        "ParkingLotPlanner: LLM generated %d-step escape plan.",
                        len(steps),
                    )
                    return steps
            except Exception as exc:
                logger.warning(
                    "ParkingLotPlanner: LLM plan failed (%s) — using default plan.",
                    exc,
                )

        return self._default_plan()

    @staticmethod
    def _parse_llm_steps(text: str) -> List[NavigationStep]:
        """Extract ``ACTION:duration`` pairs from LLM free-text output."""
        pattern = re.compile(
            r"\b(FORWARD|TURN_LEFT|TURN_RIGHT|REVERSE|STOP):(\d+(?:\.\d+)?)\b",
            re.IGNORECASE,
        )
        steps = []
        for match in pattern.finditer(text):
            action = match.group(1).upper()
            duration = float(match.group(2))
            duration = max(0.5, min(duration, 10.0))  # clamp to safe range [0.5, 10.0] seconds
            steps.append(NavigationStep(action, duration))
        return steps

    @staticmethod
    def _default_plan() -> List[NavigationStep]:
        """Conservative rule-based escape plan when no LLM is available."""
        return [
            NavigationStep("FORWARD",    2.0, "Move forward to clear parking space"),
            NavigationStep("TURN_RIGHT", 2.5, "Turn toward parking lot exit"),
            NavigationStep("FORWARD",    3.0, "Drive toward exit"),
            NavigationStep("TURN_LEFT",  1.5, "Align with road"),
            NavigationStep("FORWARD",    4.0, "Reach main road"),
        ]

    # ------------------------------------------------------------------
    # Action → controls mapping
    # ------------------------------------------------------------------

    def _action_to_controls(self, action: str) -> Tuple[float, float, float]:
        """Convert a navigation action token to ``(steer, throttle, brake)``."""
        mapping = {
            "FORWARD":     (0.0,  0.4, 0.0),
            "TURN_LEFT":   (-0.5, 0.3, 0.0),
            "TURN_RIGHT":  (0.5,  0.3, 0.0),
            "REVERSE":     (0.0,  0.3, 0.0),  # throttle in reverse gear
            "STOP":        (0.0,  0.0, 0.8),
        }
        return mapping.get(action, (0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Main update API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        gps_error: float = 0.0,
        speed_kph: float = 0.0,
    ) -> Optional[Tuple[float, float, float]]:
        """Check the current state and return a control override if active.

        Call this once per frame from the driving loop *before*
        :meth:`~ets2_driver.driver.ETS2Driver._resolve_controls`.

        Parameters
        ----------
        frame:
            Current BGR game frame.
        gps_error:
            GPS lateral offset in pixels (from :class:`VisionSystem`).
        speed_kph:
            Current vehicle speed in km/h.  Used for stuck detection: if the
            truck stays at near-zero speed while executing a forward action,
            the planner inserts a reverse escape sequence.

        Returns
        -------
        Optional[Tuple[float, float, float]]
            ``(steer, throttle, brake)`` override when the planner is active,
            or ``None`` to let normal lane-following proceed.
        """
        now = time.monotonic()

        # Periodically re-evaluate the road/parking state
        if now - self._last_check >= self._CHECK_INTERVAL_S:
            self._last_check = now
            on_road, density = self.detect_road_presence(frame)

            if self._state == ParkingState.UNKNOWN:
                if on_road:
                    logger.info(
                        "ParkingLotPlanner: road detected on startup (density=%.4f) — "
                        "normal driving.",
                        density,
                    )
                    self._state = ParkingState.ON_ROAD
                else:
                    logger.info(
                        "ParkingLotPlanner: parking lot detected (density=%.4f) — "
                        "building escape plan.",
                        density,
                    )
                    self._state = ParkingState.IN_PARKING_LOT
                    self._nav_plan = self._llm_plan_escape(density, gps_error)
                    self._current_step_idx = 0

            elif self._state == ParkingState.EXITING:
                if on_road:
                    logger.info(
                        "ParkingLotPlanner: road reached (density=%.4f) — "
                        "handing back to normal driving.",
                        density,
                    )
                    self._state = ParkingState.ON_ROAD
                    self._nav_plan = []

        # If not active, let normal lane-following take over
        if not self.is_active:
            return None

        # If the plan is exhausted, transition to exiting state
        if not self._nav_plan or self._current_step_idx >= len(self._nav_plan):
            logger.info("ParkingLotPlanner: plan exhausted — switching to EXITING state.")
            self._state = ParkingState.EXITING
            return None

        # Execute the current step
        step = self._nav_plan[self._current_step_idx]
        if step._start_time is None:
            step.start()
            # Handle gear transitions when starting a new step.
            if step.action == "REVERSE":
                if self._gear_shifter is not None:
                    self._gear_shifter.reverse()
                self._reverse_engaged = True
                self._stuck_frames = 0
            elif self._reverse_engaged:
                # Returning to forward travel — shift out of reverse gear.
                if self._gear_shifter is not None:
                    self._gear_shifter.gear_up()
                self._reverse_engaged = False
            logger.info(
                "ParkingLotPlanner: step %d/%d — %s (%.1fs) %s",
                self._current_step_idx + 1,
                len(self._nav_plan),
                step.action,
                step.duration_s,
                step.description,
            )

        # Stuck detection: accumulate while a forward-type action is executing
        # at near-zero speed, then dynamically insert a reverse escape sequence.
        if step.action in ("FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            if speed_kph < self._STUCK_SPEED_THRESHOLD_KPH:
                self._stuck_frames += 1
            else:
                self._stuck_frames = 0

            if (
                self._stuck_frames >= self._STUCK_FRAMES_THRESHOLD
                and now - self._last_stuck_insert >= self._STUCK_INSERT_COOLDOWN_S
            ):
                logger.warning(
                    "ParkingLotPlanner: stuck (speed=%.1f kph, %d frames) — "
                    "inserting reverse sequence at step %d.",
                    speed_kph,
                    self._stuck_frames,
                    self._current_step_idx,
                )
                self._stuck_frames = 0
                self._last_stuck_insert = now
                # Reset the current step so it restarts cleanly after reversing.
                step._start_time = None
                reverse_steps = [
                    NavigationStep("REVERSE", 3.0, "Escape obstacle — reversing"),
                    NavigationStep("STOP", 0.5, "Stop before reattempting forward"),
                ]
                self._nav_plan = (
                    self._nav_plan[: self._current_step_idx]
                    + reverse_steps
                    + self._nav_plan[self._current_step_idx :]
                )
                return (0.0, 0.0, 0.3)  # brief brake while gear engages
        else:
            self._stuck_frames = 0

        if step.is_complete:
            self._current_step_idx += 1
            if self._current_step_idx >= len(self._nav_plan):
                logger.info("ParkingLotPlanner: all steps complete — switching to EXITING.")
                self._state = ParkingState.EXITING
            return None

        return self._action_to_controls(step.action)
