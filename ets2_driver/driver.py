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

import collections
import logging
import signal
import time
from typing import Deque, List, Optional

import cv2

from .config import ETS2Config
from .controller import PIDSteering, SpeedController, VJoyController
from .dashboard import DashboardServer, TelemetryState
from .detection import ObstacleDetector
from .gears import GearShifter
from .llm_planner import LLMPlanner
from .speed_limit import SpeedLimitDetector
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
        self.speed_limit_detector = SpeedLimitDetector(self.cfg)
        self.dashboard = DashboardServer(self.cfg)

        # Rolling FPS measurement — keep the last 60 tick timestamps so that
        # the displayed FPS updates every frame rather than every 5 seconds.
        self._tick_times: Deque[float] = collections.deque(maxlen=60)
        self._current_fps: float = 0.0

        # Speed-limit frame-skip counter
        self._sl_frame_counter: int = 0

        # Register SIGINT handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            "ETS2Driver ready — FPS=%d, debug=%s, LLM=%s, dashboard=%s",
            self.cfg.loop.fps,
            self.cfg.loop.show_debug,
            self.cfg.llm.enabled,
            self.cfg.dashboard.enabled,
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
        last_fps_log: float = time.monotonic()

        # Start the dashboard server (non-blocking daemon thread)
        self.dashboard.start()

        logger.info("Driving loop started.")

        while self._running:
            t_start = time.monotonic()

            try:
                self._tick()
            except Exception as exc:
                logger.error("Unhandled error in driving tick: %s", exc, exc_info=True)
                # Brief pause before retrying — avoids tight error loops
                time.sleep(self._error_retry_delay)

            # Rolling FPS: record this tick's timestamp and compute rate over
            # the window stored in the deque (up to the last 60 ticks).
            now = time.monotonic()
            self._tick_times.append(now)
            if len(self._tick_times) >= 2:
                window = self._tick_times[-1] - self._tick_times[0]
                if window > 0:
                    self._current_fps = (len(self._tick_times) - 1) / window

            # Log FPS every 5 s (unchanged cadence, but uses smooth value)
            if now - last_fps_log >= 5.0:
                logger.info("Running at ~%.1f FPS", self._current_fps)
                last_fps_log = now

            # Sleep to maintain target FPS
            elapsed_tick = now - t_start
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

        # 4. Speed-limit sign detection (runs every N frames)
        self._sl_frame_counter += 1
        sl_skip = self.cfg.speed_limit.detection_every_n_frames
        if self._sl_frame_counter % sl_skip == 0:
            self.speed_limit_detector.detect(frame)
        sl_result = self.speed_limit_detector.last_result

        # 5. LLM advisory (runs at ≤ 2 Hz when enabled)
        llm_action = "CONTINUE"
        if self.planner.should_query():
            llm_action = self.planner.query(
                lane_error=float(lane_error),
                obstacle_action=avoidance_action,
                gps_error=float(gps_error),
                speed=None,  # extend with OCR speed reading if desired
            )

        # 6. Resolve final control outputs (speed limit respected if detected)
        steer, throttle, brake = self._resolve_controls(
            combined_error=combined_error,
            avoidance_action=avoidance_action,
            steer_override=steer_override,
            llm_action=llm_action,
            speed_limit=sl_result.limit_kph,
        )

        # 7. Send axis commands to vJoy
        self.vjoy.set_steering(steer)
        self.vjoy.set_throttle(throttle)
        self.vjoy.set_brake(brake)

        # 8. Gear shifting (speed estimation unavailable without OCR → pass None)
        self.gears.update(estimated_speed=None)

        # 9. Build reasoning log for the dashboard
        decision_reasons = self._build_decision_reasons(
            lane_error=lane_error,
            gps_error=gps_error,
            combined_error=combined_error,
            avoidance_action=avoidance_action,
            llm_action=llm_action,
            steer=steer,
            throttle=throttle,
            brake=brake,
            speed_limit=sl_result.limit_kph,
            sl_confidence=sl_result.confidence,
        )

        # 10. Push telemetry to the dashboard
        self.dashboard.push_telemetry(
            TelemetryState(
                frame=frame,
                gps_frame=gps_img if gps_img is not None and gps_img.size > 0 else None,
                steering=steer,
                throttle=throttle,
                brake=brake,
                lane_error=float(lane_error),
                gps_error=float(gps_error),
                combined_error=float(combined_error),
                speed_limit=sl_result.limit_kph,
                speed_limit_confidence=sl_result.confidence,
                avoidance_action=avoidance_action,
                llm_action=llm_action,
                decision_reasons=decision_reasons,
                obstacles=obstacles,
                fps=self._current_fps,
            )
        )

        # 11. Debug overlay (optional OpenCV window)
        if self.cfg.loop.show_debug:
            lane_center = self.vision.detect_lane_center(frame)
            debug_frame = self.vision.draw_debug(
                frame, lane_center, combined_error, obstacles
            )
            # Annotate speed limit on debug frame
            if sl_result.limit_kph is not None:
                cv2.putText(
                    debug_frame,
                    f"Limit: {sl_result.limit_kph} km/h ({sl_result.confidence:.0%})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
                )
            cv2.imshow("ETS2 AI Driver — Debug", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()

    # ------------------------------------------------------------------
    # Decision reasoning
    # ------------------------------------------------------------------

    def _build_decision_reasons(
        self,
        lane_error: float,
        gps_error: float,
        combined_error: float,
        avoidance_action: str,
        llm_action: str,
        steer: float,
        throttle: float,
        brake: float,
        speed_limit: Optional[int],
        sl_confidence: float,
    ) -> List[str]:
        """Build a human-readable list of factors driving the current decision.

        Returns
        -------
        List[str]
            Short strings shown in the dashboard decision log.
        """
        reasons: List[str] = []

        # Lane / GPS status
        abs_err = abs(combined_error)
        if abs_err < 20:
            reasons.append("Lane centred — smooth cruise.")
        elif abs_err < self.cfg.speed.turn_error_threshold:
            side = "right" if combined_error > 0 else "left"
            reasons.append(f"Minor drift {side} ({abs_err:.0f}px) — gentle correction.")
        elif abs_err < self.cfg.speed.emergency_brake_threshold:
            side = "right" if combined_error > 0 else "left"
            reasons.append(f"Significant drift {side} ({abs_err:.0f}px) — reducing speed.")
        else:
            side = "right" if combined_error > 0 else "left"
            reasons.append(f"⚠ Large drift {side} ({abs_err:.0f}px) — emergency brake!")

        # GPS
        if abs(gps_error) > 20:
            side = "right" if gps_error > 0 else "left"
            reasons.append(f"GPS: route veers {side} ({abs(gps_error):.0f}px).")

        # Obstacles
        if avoidance_action != "none":
            reasons.append(f"🚗 Obstacle detected → {avoidance_action}.")

        # Speed limit
        if speed_limit is not None:
            reasons.append(
                f"🚦 Speed sign: {speed_limit} km/h "
                f"({sl_confidence:.0%} conf)."
            )

        # LLM advisory
        if llm_action != "CONTINUE":
            reasons.append(f"🤖 LLM advisory: {llm_action}.")

        # Control summary
        reasons.append(
            f"Output → steer={steer:+.3f}  thr={throttle:.0%}  brk={brake:.0%}"
        )

        return reasons

    def _resolve_controls(
        self,
        combined_error: float,
        avoidance_action: str,
        steer_override: float,
        llm_action: str,
        speed_limit: Optional[int] = None,
    ) -> tuple[float, float, float]:
        """Determine final (steer, throttle, brake) triple.

        Priority order (highest → lowest):
        1. LLM ``STOP`` command.
        2. LLM ``BRAKE`` command.
        3. Obstacle avoidance (``swerve_*``).
        4. Normal lane-follow + speed control.
        5. Speed-limit cap on throttle.

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
        speed_limit:
            Detected speed limit in km/h, or ``None`` if no sign is visible.
            When set, the cruise throttle is scaled down proportionally for
            lower limits so the truck doesn't blast through a 30-zone.

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

        # --- Speed-limit cap ---
        # Scale the allowed throttle proportionally: at the reference (max)
        # speed the full cruise_throttle applies; lower limits reduce it
        # linearly.  Reference speed is configurable via ETS2_SL_MAX_SPEED.
        if speed_limit is not None and speed_limit > 0:
            ref_speed = max(1, self.cfg.speed_limit.max_reference_speed_kph)
            max_thr = scfg.cruise_throttle * min(1.0, speed_limit / ref_speed)
            throttle = min(throttle, max_thr)

        return steer, throttle, brake
