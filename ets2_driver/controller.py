"""vJoy-based input controller for steering, throttle, and brake axes.

The vJoy virtual joystick driver must be installed and the vJoy device
configured in *VJoyConf* before running.  Download from:
https://sourceforge.net/projects/vjoystick/

Axis mapping (configurable in VJoyConf):
    HID_USAGE_X  → Steering  (axis 0x30)
    HID_USAGE_Y  → Throttle  (axis 0x31)
    HID_USAGE_Z  → Brake     (axis 0x32)

In ETS2 go to  Options → Controls → Assign axes to match the above.
"""

from __future__ import annotations

import logging
from typing import Optional

from .adaptive_pid import AdaptivePIDGains
from .config import ETS2Config

logger = logging.getLogger(__name__)

# vJoy axis range is [1, 32_767].  The centre of the steering axis is 16_384.
_VJOY_MAX: int = 32_767
_VJOY_CENTRE: int = (_VJOY_MAX + 1) // 2  # 16_384


class VJoyController:
    """Wraps a pyvjoy device to provide normalised axis control.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.
    device_id:
        vJoy device ID (default 1).  Change if you have multiple vJoy devices.

    Notes
    -----
    If *pyvjoy* is not installed the controller runs in *headless* (no-op)
    mode so that the rest of the codebase can be tested without the vJoy
    driver.
    """

    def __init__(self, cfg: ETS2Config, device_id: int = 1) -> None:
        self.cfg = cfg
        self._device: Optional[object] = None

        try:
            import pyvjoy  # type: ignore
            self._device = pyvjoy.VJoyDevice(device_id)
            self._HID_X = pyvjoy.HID_USAGE_X
            self._HID_Y = pyvjoy.HID_USAGE_Y
            self._HID_Z = pyvjoy.HID_USAGE_Z
            logger.info("vJoy device %d acquired.", device_id)
        except Exception as exc:
            logger.warning(
                "pyvjoy not available or vJoy device %d not found (%s). "
                "Running in headless mode — inputs will be logged only.",
                device_id, exc,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_axis(self, hid_usage: int, normalised: float) -> None:
        """Write a normalised ``[-1, 1]`` value to a vJoy axis.

        Parameters
        ----------
        hid_usage:
            pyvjoy HID usage constant (X / Y / Z).
        normalised:
            Value in ``[-1, 1]``.  For ``[0, 1]`` axes (throttle/brake) use
            the dedicated :meth:`set_throttle` / :meth:`set_brake` methods.
        """
        raw = int(normalised * _VJOY_MAX)
        raw = max(1, min(_VJOY_MAX, raw))
        if self._device is not None:
            try:
                self._device.set_axis(hid_usage, raw)
            except Exception as exc:
                logger.error("vJoy axis write failed: %s", exc)
        else:
            logger.debug("Headless axis write: hid=%d raw=%d", hid_usage, raw)

    def _set_unipolar_axis(self, hid_usage: int, value: float, axis_name: str) -> None:
        """Write a normalised ``[0, 1]`` value to a vJoy axis.

        Maps the normalised value directly to ``[0, 32767]`` with no offset.

        Parameters
        ----------
        hid_usage:
            pyvjoy HID usage constant (Y / Z for throttle / brake).
        value:
            Pre-clamped value in ``[0.0, 1.0]``.
        axis_name:
            Human-readable axis name used in log messages (e.g. ``"throttle"``).
        """
        raw = int(value * _VJOY_MAX)
        raw = max(0, min(_VJOY_MAX, raw))
        if self._device is not None:
            try:
                self._device.set_axis(hid_usage, raw)
            except Exception as exc:
                logger.error("vJoy %s write failed: %s", axis_name, exc)
        else:
            logger.debug("Headless %s: value=%.3f raw=%d", axis_name, value, raw)

    # ------------------------------------------------------------------
    # Public control interface
    # ------------------------------------------------------------------

    def set_steering(self, value: float) -> None:
        """Set the steering axis.

        Parameters
        ----------
        value:
            Normalised steering in ``[-1.0, 1.0]``.
            ``-1.0`` = full left, ``0.0`` = centre, ``+1.0`` = full right.
        """
        value = max(-1.0, min(1.0, value))
        # Map [-1, 1] → [1, 32767] with 16384 as centre
        raw = int((value + 1.0) / 2.0 * _VJOY_MAX)
        raw = max(1, min(_VJOY_MAX, raw))
        if self._device is not None:
            try:
                self._device.set_axis(self._HID_X, raw)
            except Exception as exc:
                logger.error("vJoy steering write failed: %s", exc)
        else:
            logger.debug("Headless steering: value=%.3f raw=%d", value, raw)

    def set_throttle(self, value: float) -> None:
        """Set the throttle axis.

        Parameters
        ----------
        value:
            Normalised throttle in ``[0.0, 1.0]``.
        """
        value = max(0.0, min(1.0, value))
        self._set_unipolar_axis(getattr(self, "_HID_Y", 0x31), value, "throttle")

    def set_brake(self, value: float) -> None:
        """Set the brake axis.

        Parameters
        ----------
        value:
            Normalised brake pressure in ``[0.0, 1.0]``.
        """
        value = max(0.0, min(1.0, value))
        self._set_unipolar_axis(getattr(self, "_HID_Z", 0x32), value, "brake")

    def release_all(self) -> None:
        """Centre steering, release throttle and brake (safe-stop state)."""
        self.set_steering(0.0)
        self.set_throttle(0.0)
        self.set_brake(0.0)
        logger.info("All vJoy axes released to zero.")


class PIDSteering:
    """Discrete-time PID controller that outputs a normalised steering value.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.

    Notes
    -----
    Integral windup is clamped by
    :attr:`~ets2_driver.config.PidConfig.integral_max`.
    The derivative term uses a simple finite difference on the error signal.
    A configurable deadzone suppresses micro-corrections when the error is
    small.  Exponential output smoothing and a per-frame rate limiter prevent
    violent steering snaps.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._smoothed_steer: float = 0.0

        # Adaptive gain scheduler (uses speed to interpolate PID coefficients)
        self._gain_scheduler: AdaptivePIDGains = AdaptivePIDGains(cfg)

        # Debug / telemetry — readable after each call to compute()
        self.last_p: float = 0.0
        self.last_i: float = 0.0
        self.last_d: float = 0.0
        self.last_raw: float = 0.0
        self.last_gain_scale: float = 0.5  # 0.0=low-speed, 1.0=high-speed curve

    def reset(self) -> None:
        """Reset integrator and derivative memory."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._smoothed_steer = 0.0

    @property
    def integral(self) -> float:
        """Current integrator state (read-only, for debug/telemetry)."""
        return self._integral

    def compute(self, error: float, speed_kph: float = 0.0) -> float:
        """Compute the normalised steering output for the given error.

        Parameters
        ----------
        error:
            Current signed pixel offset from lane centre (positive = steer right).
        speed_kph:
            Current vehicle speed in km/h, used to look up the adaptive gain
            schedule.  Pass ``0.0`` (default) to use the static PID config.

        Returns
        -------
        float
            Normalised steering value in ``[-1.0, 1.0]`` after smoothing and
            rate-limiting.
        """
        pcfg = self.cfg.pid

        # Resolve gains — adaptive scheduler overrides static config when enabled
        kp, ki, kd, integral_max, gain_scale = self._gain_scheduler.get_gains(speed_kph)
        self.last_gain_scale = gain_scale

        # Deadzone with linear ramp to avoid abrupt derivative spikes.
        # Errors within the deadzone are scaled toward zero proportionally
        # so the transition is smooth rather than a hard step.
        abs_err = abs(error)
        if abs_err < pcfg.deadzone_px:
            # Scale from 0 at centre to 1 at the deadzone boundary
            scale = abs_err / pcfg.deadzone_px
            error = error * scale

        self._integral += error
        # Anti-windup using speed-adjusted integral clamp
        self._integral = max(
            -integral_max, min(integral_max, self._integral)
        )

        derivative = error - self._prev_error
        self._prev_error = error

        p_term = kp * error
        i_term = ki * self._integral
        d_term = kd * derivative

        # Store debug terms
        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term

        raw = p_term + i_term + d_term
        raw = max(-1.0, min(1.0, raw))
        self.last_raw = raw

        # Exponential smoothing on the raw output
        alpha = pcfg.steer_smoothing
        smoothed = alpha * self._smoothed_steer + (1.0 - alpha) * raw

        # Rate limiting — cap how much the steering can change in one frame
        delta = smoothed - self._smoothed_steer
        delta = max(-pcfg.max_steer_rate, min(pcfg.max_steer_rate, delta))
        self._smoothed_steer += delta

        return max(-1.0, min(1.0, self._smoothed_steer))


class SpeedController:
    """Smooth rule-based throttle/brake controller.

    Improves on the previous binary on/off logic with exponential smoothing,
    acceleration/deceleration ramping, adaptive speed scaling based on steering
    intensity, a coasting mode for gentle curve approach, and emergency braking
    for extreme steering errors.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._smoothed_throttle: float = 0.0
        self._smoothed_brake: float = 0.0

        # Debug / telemetry — readable after each call to compute()
        self.last_raw_throttle: float = 0.0
        self.last_raw_brake: float = 0.0
        self.in_coasting: bool = False
        self.in_emergency: bool = False

    def compute(self, steering_error: float, speed_kph: float = 0.0) -> tuple[float, float]:
        """Return ``(throttle, brake)`` based on the current steering error.

        Applies exponential smoothing and ramp-rate limiting to prevent sudden
        input changes, and triggers emergency braking when the error is extreme.
        A coasting mode gently reduces throttle when the steering error is
        approaching a turn threshold, smoothing the speed transition.

        At higher speeds the coasting zone is triggered earlier (lower steering
        error threshold), acceleration ramps faster, and the throttle is reduced
        before the brake is applied to simulate engine braking.

        Parameters
        ----------
        steering_error:
            Absolute steering error in pixels.
        speed_kph:
            Current vehicle speed in km/h.  Used to scale the coasting
            threshold and ramp rate dynamically.

        Returns
        -------
        tuple[float, float]
            ``(throttle, brake)`` both in ``[0.0, 1.0]``.
        """
        scfg = self.cfg.speed
        abs_error = abs(steering_error)

        self.in_coasting = False
        self.in_emergency = False

        # Speed-aware coasting threshold: trigger coasting earlier at high speed
        # so the truck starts slowing down sooner before a curve.
        # Factor scales from 1.0 at 0 km/h down to 0.5 at 200 km/h.
        coast_factor = max(0.5, 1.0 - speed_kph / 200.0)
        effective_coast_threshold = scfg.coasting_threshold * coast_factor

        # Speed-aware ramp rate: allow faster throttle build-up at higher speed
        # so the truck accelerates smoothly back up after a curve.
        # Factor scales from 1.0 at 0 km/h up to 2.0 at 100 km/h, then clamped.
        ramp_factor = min(2.0, 1.0 + speed_kph / 100.0)
        effective_ramp = min(
            scfg.speed_ramp_rate * ramp_factor,
            scfg.speed_ramp_rate * 2.0,
        )

        # --- Determine raw target values ---
        if abs_error > scfg.emergency_brake_threshold:
            # Emergency braking: hard stop, no throttle
            self.in_emergency = True
            target_throttle = 0.0
            target_brake = scfg.emergency_brake_value
        elif abs_error > scfg.turn_error_threshold:
            # Scale throttle/brake proportionally between turn and emergency thresholds.
            # If the two thresholds are equal or inverted, treat as full emergency.
            turn_range = scfg.emergency_brake_threshold - scfg.turn_error_threshold
            if turn_range <= 0:
                severity = 1.0
            else:
                severity = (abs_error - scfg.turn_error_threshold) / turn_range
                severity = max(0.0, min(1.0, severity))
            target_throttle = scfg.turn_throttle * (1.0 - severity)
            # Always interpolate toward emergency value, clamping so brake never
            # decreases even if emergency_brake_value < turn_brake.
            max_brake = max(scfg.turn_brake, scfg.emergency_brake_value)
            target_brake = scfg.turn_brake + severity * (max_brake - scfg.turn_brake)
        elif abs_error > effective_coast_threshold:
            # Coasting zone: gently reduce throttle before entering a turn
            self.in_coasting = True
            target_throttle = scfg.coasting_throttle
            target_brake = 0.0
        else:
            target_throttle = scfg.cruise_throttle
            target_brake = 0.0

        # Engine braking simulation: when brake target > 0 and we are still
        # carrying significant throttle, first shed throttle before braking.
        # This produces a smoother deceleration feel at higher speeds.
        if target_brake > 0.0 and self._smoothed_throttle > 0.1:
            target_throttle = 0.0

        # Store raw targets for debug
        self.last_raw_throttle = target_throttle
        self.last_raw_brake = target_brake

        # --- Ramp-rate limiting (cap per-frame change) ---
        throttle_delta = target_throttle - self._smoothed_throttle
        throttle_delta = max(-effective_ramp, min(effective_ramp, throttle_delta))
        ramped_throttle = self._smoothed_throttle + throttle_delta

        brake_delta = target_brake - self._smoothed_brake
        brake_delta = max(-scfg.brake_ramp_rate, min(scfg.brake_ramp_rate, brake_delta))
        ramped_brake = self._smoothed_brake + brake_delta

        # --- Exponential smoothing ---
        alpha_t = scfg.throttle_smoothing
        alpha_b = scfg.brake_smoothing
        self._smoothed_throttle = alpha_t * self._smoothed_throttle + (1.0 - alpha_t) * ramped_throttle
        self._smoothed_brake = alpha_b * self._smoothed_brake + (1.0 - alpha_b) * ramped_brake

        throttle = max(0.0, min(1.0, self._smoothed_throttle))
        brake = max(0.0, min(1.0, self._smoothed_brake))
        return throttle, brake
