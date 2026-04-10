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

import atexit
import collections
import logging
import signal
import time
from typing import Deque, List, Optional

import cv2

from .config import ETS2Config
from .camera import CameraManager
from .controller import PIDSteering, SpeedController, VJoyController
from .dashboard import DashboardServer, TelemetryState
from .debug_state import DebugState
from .detection import ObstacleDetector
from .gears import GearShifter
from .llm_planner import LLMPlanner
from .parking_planner import ParkingLotPlanner
from .speed_limit import SpeedLimitDetector
from .speed_tracker import SpeedTracker
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
        self.speed_tracker = SpeedTracker(self.cfg)
        self.dashboard = DashboardServer(self.cfg)
        self.debug_state = DebugState()
        self.camera = CameraManager(self.cfg)
        self.parking_planner = ParkingLotPlanner(self.cfg, gear_shifter=self.gears)

        # Rolling FPS measurement — keep the last 60 tick timestamps so that
        # the displayed FPS updates every frame rather than every 5 seconds.
        self._tick_times: Deque[float] = collections.deque(maxlen=60)
        self._current_fps: float = 0.0

        # Speed-limit frame-skip counter
        self._sl_frame_counter: int = 0

        # Register SIGINT handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Safety-net: ensure vJoy axes are released even if the process exits
        # via sys.exit() or an uncaught exception rather than a signal.
        atexit.register(self._atexit_cleanup)

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
        """Request the main loop to exit and halt the vehicle via vJoy.

        Sends the brake command several times with a short delay between each
        attempt so ETS2 has time to process the stop before the Python process
        exits.  vJoy retains the last written axis values after the process
        terminates, so we must ensure the final state is *brake full /
        throttle zero* rather than whatever was active during the last tick.
        """
        self._running = False
        # Repeat the stop command to ensure ETS2 receives it.  Five attempts
        # at 50 ms intervals give the game 250 ms to respond — enough for
        # one or two rendered frames at typical FPS targets.
        for _ in range(5):
            self.vjoy.release_all()
            time.sleep(0.05)
        logger.info("ETS2Driver stopped.")

    def _atexit_cleanup(self) -> None:
        """Safety-net atexit callback: zero throttle and apply full brake.

        Called automatically by the Python interpreter when the process is
        about to exit (sys.exit, end of main, unhandled exception).  This
        supplements the signal handlers to cover cases where neither SIGINT
        nor SIGTERM fires (e.g. the process is ended by the test runner or an
        IDE).
        """
        try:
            for _ in range(3):
                self.vjoy.set_throttle(0.0)
                self.vjoy.set_brake(1.0)
        except Exception:
            pass  # Best-effort — do not raise during interpreter shutdown

    def export_debug(self, path: str = "debug_log.json") -> None:
        """Write the rolling 60-frame debug history to a JSON file.

        Parameters
        ----------
        path:
            File path to write the JSON log to (default ``"debug_log.json"``).
        """
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.debug_state.export_json())
            logger.info("Debug log exported to %s (%d frames).",
                        path, len(self.debug_state.history))
        except OSError as exc:
            logger.error("Failed to export debug log: %s", exc)

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

        # Start a new debug frame for this tick
        dbg = self.debug_state.new_frame()

        # 1. Capture frame
        frame = self.vision.get_frame()
        frame_width = frame.shape[1]

        # 2. Update speed tracker (HUD OCR; returns 0.0 if disabled/unavailable)
        current_speed = self.speed_tracker.update(frame)

        # 2. Compute steering error (lane + GPS blend)
        lane_error = self.vision.get_steering_error(frame)
        gps_img = self.vision.crop_gps(frame)
        gps_error = self.vision.gps_direction(gps_img)
        combined_error = self.vision.get_combined_error(frame)

        # Populate lane debug info from vision system
        dbg.lane_error = float(lane_error)
        dbg.lane_confidence = self.vision.last_lane_confidence
        dbg.lane_candidates = list(self.vision.last_lane_candidates)
        dbg.gps_error = float(gps_error)
        dbg.gps_blend_weight = self.cfg.gps.blend_weight
        dbg.combined_error = float(combined_error)

        # 3. YOLO obstacle detection
        obstacles = self.detector.detect(frame)
        avoidance_action, steer_override = self.detector.get_avoidance_action(
            obstacles, frame_width
        )

        # Populate obstacle debug info
        dbg.obstacle_count = len(obstacles)
        dbg.obstacle_sides = list(self.detector.last_obstacle_sides)
        dbg.obstacle_distances = list(self.detector.last_obstacle_distances)
        dbg.avoidance_action = avoidance_action
        dbg.lane_target = self.detector.last_lane_target

        # 4. Speed-limit sign detection (runs every N frames)
        self._sl_frame_counter += 1
        sl_skip = self.cfg.speed_limit.detection_every_n_frames
        if self._sl_frame_counter % sl_skip == 0:
            self.speed_limit_detector.detect(frame)
        sl_result = self.speed_limit_detector.last_result
        dbg.speed_limit = sl_result.limit_kph

        # 5. LLM advisory (runs at ≤ 2 Hz when enabled)
        llm_action = "CONTINUE"
        if self.planner.should_query():
            llm_action = self.planner.query(
                lane_error=float(lane_error),
                obstacle_action=avoidance_action,
                gps_error=float(gps_error),
                speed=current_speed if current_speed > 0.0 else None,
            )
        dbg.llm_action = llm_action

        # 6. Check parking-lot planner (takes priority over normal lane-follow
        #    when the truck is still in a parking lot at session start).
        parking_override = self.parking_planner.update(
            frame, gps_error=float(gps_error), speed_kph=current_speed
        )

        if parking_override is not None:
            steer, throttle, brake = parking_override
        else:
            # Normal resolve path
            steer, throttle, brake = self._resolve_controls(
                combined_error=combined_error,
                avoidance_action=avoidance_action,
                steer_override=steer_override,
                llm_action=llm_action,
                speed_limit=sl_result.limit_kph,
                current_speed=current_speed,
            )

        # Populate control debug info (raw PID values from controller)
        dbg.raw_steer = self.pid.last_raw
        dbg.pid_p = self.pid.last_p
        dbg.pid_i = self.pid.last_i
        dbg.pid_d = self.pid.last_d
        dbg.pid_integral = self.pid.integral
        dbg.raw_throttle = self.speed_ctrl.last_raw_throttle
        dbg.raw_brake = self.speed_ctrl.last_raw_brake
        dbg.in_coasting = self.speed_ctrl.in_coasting
        dbg.in_emergency = self.speed_ctrl.in_emergency
        dbg.steer = steer
        dbg.throttle = throttle
        dbg.brake = brake

        # Populate speed tracking and adaptive PID debug info
        dbg.current_speed = current_speed
        dbg.target_speed = float(sl_result.limit_kph) if sl_result.limit_kph is not None else None
        dbg.velocity = self.speed_tracker.velocity
        dbg.velocity_trend = self.speed_tracker.velocity_trend
        dbg.pid_gain_scale = self.pid.last_gain_scale
        dbg.acceleration = self.speed_tracker.velocity

        # Commit the frame to history
        self.debug_state.commit()

        # 7. Send axis commands to vJoy
        self.vjoy.set_steering(steer)
        self.vjoy.set_throttle(throttle)
        self.vjoy.set_brake(brake)

        # 8. Gear shifting — pass OCR speed if available
        self.gears.update(estimated_speed=current_speed if current_speed > 0.0 else None)

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
            current_speed=current_speed,
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
                debug_summary=self.debug_state.summary(),
                obstacle_sides=list(self.detector.last_obstacle_sides),
                lane_confidence=self.vision.last_lane_confidence,
                lane_target=self.detector.last_lane_target,
                current_speed=current_speed,
                target_speed=float(sl_result.limit_kph) if sl_result.limit_kph is not None else None,
                velocity_trend=self.speed_tracker.velocity_trend,
                pid_gain_scale=self.pid.last_gain_scale,
            )
        )

        # 11. Debug overlay (optional OpenCV window)
        if self.cfg.loop.show_debug:
            lane_center = self.vision.detect_lane_center(frame)
            debug_frame = self.vision.draw_debug(
                frame,
                lane_center,
                combined_error,
                obstacles,
                lane_candidates=self.vision.last_lane_candidates,
                obstacle_sides=self.detector.last_obstacle_sides,
            )
            # Annotate speed limit on debug frame
            if sl_result.limit_kph is not None:
                cv2.putText(
                    debug_frame,
                    f"Limit: {sl_result.limit_kph} km/h ({sl_result.confidence:.0%})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
                )
            # Annotate coasting / emergency state
            if self.speed_ctrl.in_emergency:
                cv2.putText(debug_frame, "EMERGENCY BRAKE",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif self.speed_ctrl.in_coasting:
                cv2.putText(debug_frame, "COASTING",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("ETS2 AI Driver — Debug", debug_frame)

            # Show GPS mini-map debug window
            gps_debug = self.vision.draw_gps_debug(gps_img)
            if gps_debug is not None and gps_debug.size > 0:
                cv2.imshow("ETS2 AI Driver — GPS Mini-map", gps_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop()
            else:
                # Keys 1–4 switch camera views
                self.camera.handle_key(key)

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
        current_speed: float = 0.0,
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

        # Coasting / emergency state from speed controller
        if self.speed_ctrl.in_emergency:
            reasons.append("🛑 Emergency brake active.")
        elif self.speed_ctrl.in_coasting:
            reasons.append("🌊 Coasting — approaching curve.")

        # GPS
        if abs(gps_error) > 20:
            side = "right" if gps_error > 0 else "left"
            reasons.append(f"GPS: route veers {side} ({abs(gps_error):.0f}px).")

        # Obstacles
        if avoidance_action != "none":
            side_info = ""
            if self.detector.last_obstacle_sides:
                unique_sides = list(dict.fromkeys(self.detector.last_obstacle_sides))
                side_info = f" [{', '.join(unique_sides)}]"
            reasons.append(f"🚗 Obstacle detected{side_info} → {avoidance_action}.")

        # Speed limit
        if speed_limit is not None:
            reasons.append(
                f"🚦 Speed sign: {speed_limit} km/h "
                f"({sl_confidence:.0%} conf)."
            )

        # LLM advisory
        if llm_action != "CONTINUE":
            reasons.append(f"🤖 LLM advisory: {llm_action}.")

        # PID debug summary
        reasons.append(
            f"PID raw={self.pid.last_raw:+.3f}  "
            f"P={self.pid.last_p:+.4f}  "
            f"I={self.pid.last_i:+.4f}  "
            f"D={self.pid.last_d:+.4f}"
        )

        # Speed telemetry and adaptive gain state
        if current_speed > 0.0:
            trend_icon = {"accelerating": "⬆", "decelerating": "⬇", "steady": "→"}.get(
                self.speed_tracker.velocity_trend, "→"
            )
            reasons.append(
                f"🚀 Speed: {current_speed:.0f} km/h {trend_icon}  "
                f"Gain curve: {self.pid.last_gain_scale:.2f}"
            )

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
        current_speed: float = 0.0,
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
        current_speed:
            Current vehicle speed in km/h from
            :class:`~ets2_driver.speed_tracker.SpeedTracker`.  Passed to the
            PID and speed controllers for adaptive gain scheduling and
            speed-aware ramp rates.

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
            steer = self.pid.compute(combined_error, speed_kph=current_speed)
            return steer, 0.0, scfg.turn_brake

        # --- Obstacle avoidance ---
        if avoidance_action != "none":
            return steer_override, 0.0, scfg.hard_brake

        # --- Normal lane-following ---
        steer = self.pid.compute(combined_error, speed_kph=current_speed)
        throttle, brake = self.speed_ctrl.compute(combined_error, speed_kph=current_speed)

        # --- Speed-limit cap ---
        # Scale the allowed throttle proportionally: at the reference (max)
        # speed the full cruise_throttle applies; lower limits reduce it
        # linearly.  Reference speed is configurable via ETS2_SL_MAX_SPEED.
        if speed_limit is not None and speed_limit > 0:
            # Clamp to [1, 200] km/h: 1 prevents division-by-zero, 200 is the
            # practical upper limit for ETS2 (exceeding it would make the cap
            # ineffective and produce unpredictable throttle scaling).
            ref_speed = max(1, min(200, self.cfg.speed_limit.max_reference_speed_kph))
            max_thr = scfg.cruise_throttle * min(1.0, speed_limit / ref_speed)
            throttle = min(throttle, max_thr)

        return steer, throttle, brake
