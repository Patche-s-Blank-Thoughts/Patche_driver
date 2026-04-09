"""Top-level ETS2Driver class — wires all subsystems together.

Architecture
------------
The driver exposes a single :meth:`ETS2Driver.run` method that executes a
blocking main loop at the configured FPS.  Internally it orchestrates:

* :class:`~ets2_driver.vision.VisionSystem` — screen capture + CV
* :class:`~ets2_driver.detection.ObstacleDetector` — YOLOv8 inference
* :class:`~ets2_driver.controller.VJoyController` — joystick axes
* :class:`~ets2_driver.controller.PIDSteering` — PID steering controller
* :class:`~ets2_driver.controller.SpeedController` — throttle/brake rules
* :class:`~ets2_driver.gears.GearShifter` — gear automation
* :class:`~ets2_driver.llm_planner.LLMPlanner` — optional LLM advisory

Execution flow per frame
------------------------
1. Capture frame.
2. Compute combined lane + GPS steering error.
3. Run YOLO detection (every N frames).
4. If obstacle detected → apply avoidance override.
5. Else → PID steer + rule-based speed control.
6. If LLM planner enabled and interval elapsed → query LLM and honour advisory.
7. Send computed axes to vJoy.
8. Optionally display debug overlay.
9. Sleep to maintain target FPS.
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Optional

import cv2

from .config import ETS2Config
from .controller import PIDSteering, SpeedController, VJoyController
from .detection import ObstacleDetector
from .gears import GearShifter
from .llm_planner import LLMPlanner
from .vision import VisionSystem

logger = logging.getLogger(__name__)


class ETS2Driver:
    """Production-ready autonomous driving bot for Euro Truck Simulator 2.

    Parameters
    ----------
    cfg:
        Optional custom :class:`~ets2_driver.config.ETS2Config` instance.
        If omitted, defaults are used (reads environment variables).

    Example
    -------
    >>> from ets2_driver import ETS2Driver
    >>> bot = ETS2Driver()
    >>> bot.run()   # blocks until Ctrl-C
    """

    def __init__(self, cfg: Optional[ETS2Config] = None) -> None:
        self.cfg = cfg or ETS2Config()
        self._running: bool = False

        # Seconds to pause before retrying after an unhandled per-frame error
        self._error_retry_delay: float = 0.1

        # Subsystems
        self.vision = VisionSystem(self.cfg)
        self.detector = ObstacleDetector(self.cfg)
        self.vjoy = VJoyController(self.cfg)
        self.pid = PIDSteering(self.cfg)
        self.speed_ctrl = SpeedController(self.cfg)
        self.gears = GearShifter(self.cfg)
        self.planner = LLMPlanner(self.cfg)

        # Register SIGINT handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            "ETS2Driver ready — FPS=%d, debug=%s, LLM=%s",
            self.cfg.loop.fps,
            self.cfg.loop.show_debug,
            self.cfg.llm.enabled,
        )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, _frame: object) -> None:
        """Gracefully stop the driver on SIGINT / SIGTERM."""
        logger.info("Signal %d received — stopping driver.", signum)
        self.stop()

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request the main loop to exit and release all vJoy axes."""
        self._running = False
        self.vjoy.release_all()
        logger.info("ETS2Driver stopped.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the blocking main driving loop.

        The loop runs at :attr:`~ets2_driver.config.LoopConfig.fps` and will
        exit cleanly when :meth:`stop` is called or Ctrl-C is pressed.
        """
        self._running = True
        frame_interval = 1.0 / max(1, self.cfg.loop.fps)
        frame_count: int = 0
        last_fps_log: float = time.monotonic()

        logger.info("Driving loop started.")

        while self._running:
            t_start = time.monotonic()
            frame_count += 1

            try:
                self._tick()
            except Exception as exc:
                logger.error("Unhandled error in driving tick: %s", exc, exc_info=True)
                # Brief pause before retrying — avoids tight error loops
                time.sleep(self._error_retry_delay)

            # FPS logging every 5 s
            now = time.monotonic()
            if now - last_fps_log >= 5.0:
                elapsed = now - last_fps_log
                logger.info("Running at ~%.1f FPS", frame_count / elapsed)
                frame_count = 0
                last_fps_log = now

            # Sleep to maintain target FPS
            elapsed_tick = time.monotonic() - t_start
            sleep_time = frame_interval - elapsed_tick
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Driving loop exited.")

    # ------------------------------------------------------------------
    # Per-frame logic
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Execute one frame of the driving pipeline."""

        # 1. Capture frame
        frame = self.vision.get_frame()
        frame_width = frame.shape[1]

        # 2. Compute steering error (lane + GPS blend)
        lane_error = self.vision.get_steering_error(frame)
        gps_img = self.vision.crop_gps(frame)
        gps_error = self.vision.gps_direction(gps_img)
        combined_error = self.vision.get_combined_error(frame)

        # 3. YOLO obstacle detection
        obstacles = self.detector.detect(frame)
        avoidance_action, steer_override = self.detector.get_avoidance_action(
            obstacles, frame_width
        )

        # 4. LLM advisory (runs at ≤ 2 Hz when enabled)
        llm_action = "CONTINUE"
        if self.planner.should_query():
            llm_action = self.planner.query(
                lane_error=float(lane_error),
                obstacle_action=avoidance_action,
                gps_error=float(gps_error),
                speed=None,  # extend with OCR speed reading if desired
            )

        # 5. Resolve final control outputs
        steer, throttle, brake = self._resolve_controls(
            combined_error=combined_error,
            avoidance_action=avoidance_action,
            steer_override=steer_override,
            llm_action=llm_action,
        )

        # 6. Send axis commands to vJoy
        self.vjoy.set_steering(steer)
        self.vjoy.set_throttle(throttle)
        self.vjoy.set_brake(brake)

        # 7. Gear shifting (speed estimation unavailable without OCR → pass None)
        self.gears.update(estimated_speed=None)

        # 8. Debug overlay
        if self.cfg.loop.show_debug:
            lane_center = self.vision.detect_lane_center(frame)
            debug_frame = self.vision.draw_debug(
                frame, lane_center, combined_error, obstacles
            )
            cv2.imshow("ETS2 AI Driver — Debug", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()

    def _resolve_controls(
        self,
        combined_error: float,
        avoidance_action: str,
        steer_override: float,
        llm_action: str,
    ) -> tuple[float, float, float]:
        """Determine final (steer, throttle, brake) triple.

        Priority order (highest → lowest):
        1. LLM ``STOP`` command.
        2. LLM ``BRAKE`` command.
        3. Obstacle avoidance (``swerve_*``).
        4. Normal lane-follow + speed control.
        5. LLM ``OVERTAKE`` / ``EXIT_*`` modifiers (future extension point).

        Parameters
        ----------
        combined_error:
            Weighted lane + GPS steering error in pixels.
        avoidance_action:
            String action token from :class:`~ets2_driver.detection.ObstacleDetector`.
        steer_override:
            Normalised steering value paired with *avoidance_action*.
        llm_action:
            High-level action token from the LLM planner.

        Returns
        -------
        tuple[float, float, float]
            ``(steer, throttle, brake)`` all in normalised ranges.
        """
        scfg = self.cfg.speed

        # --- LLM hard stop ---
        if llm_action == "STOP":
            return 0.0, 0.0, scfg.hard_brake

        # --- LLM moderate brake (e.g. red light) ---
        if llm_action == "BRAKE":
            steer = self.pid.compute(combined_error)
            return steer, 0.0, scfg.turn_brake

        # --- Obstacle avoidance ---
        if avoidance_action != "none":
            return steer_override, 0.0, scfg.hard_brake

        # --- Normal lane-following ---
        steer = self.pid.compute(combined_error)
        throttle, brake = self.speed_ctrl.compute(combined_error)
        return steer, throttle, brake
