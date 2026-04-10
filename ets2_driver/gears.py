"""Automatic gear-shifting module.

ETS2 accepts keyboard input for gear control.  The simplest approach is to
bind *Shift* to gear-up and *Ctrl* to gear-down (configurable in
:class:`~ets2_driver.config.GearConfig`) and then call :meth:`GearShifter.update`
from the control loop.

If you enable automatic transmission in ETS2's settings
(*Options → Gameplay → Transmission = Automatic*) you can set
``auto_transmission = true`` in the environment and this module becomes a
no-op.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .config import ETS2Config

logger = logging.getLogger(__name__)


class GearShifter:
    """Manages gear-shifting via keyboard key presses.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.

    Notes
    -----
    A minimum interval between gear changes
    (:attr:`_shift_cooldown_s`) prevents rapid hunting between gears at the
    speed thresholds.
    """

    _shift_cooldown_s: float = 1.5  # seconds between consecutive shifts

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._last_shift: float = 0.0
        self._keyboard_available: bool = False
        self._stuck_counter: int = 0
        self._last_reverse_time: float = 0.0

        try:
            import keyboard  # type: ignore  # noqa: F401
            self._keyboard_available = True
            if cfg.gear.auto_transmission:
                logger.info(
                    "GearShifter: ETS2 automatic transmission enabled — "
                    "auto-shifting disabled (reverse key still active: %s).",
                    cfg.gear.reverse_key,
                )
            else:
                logger.info("GearShifter ready (up=%s, down=%s, reverse=%s).",
                            cfg.gear.gear_up_key, cfg.gear.gear_down_key,
                            cfg.gear.reverse_key)
        except Exception as exc:
            logger.warning("keyboard module not available (%s). "
                           "Gear shifting disabled.", exc)

    # ------------------------------------------------------------------

    def gear_up(self) -> None:
        """Send a single gear-up key press."""
        if not self._keyboard_available:
            logger.debug("gear_up (no-op)")
            return
        import keyboard  # type: ignore
        keyboard.press_and_release(self.cfg.gear.gear_up_key)
        logger.debug("Shifted UP")

    def gear_down(self) -> None:
        """Send a single gear-down key press."""
        if not self._keyboard_available:
            logger.debug("gear_down (no-op)")
            return
        import keyboard  # type: ignore
        keyboard.press_and_release(self.cfg.gear.gear_down_key)
        logger.debug("Shifted DOWN")

    def reverse(self) -> None:
        """Engage reverse gear via key press.

        Can be called manually or programmatically.  Subject to the same
        keyboard-availability guard as :meth:`gear_up` and :meth:`gear_down`.
        """
        if not self._keyboard_available:
            logger.debug("reverse (no-op)")
            return
        import keyboard  # type: ignore
        keyboard.press_and_release(self.cfg.gear.reverse_key)
        logger.debug("Reverse gear engaged")

    def update(self, estimated_speed: Optional[float], throttle: float = 0.0, brake: float = 0.0) -> None:
        """Automatically shift up or down based on the estimated truck speed.

        Also detects when the truck is stuck (throttle applied but speed near
        zero) and automatically engages reverse to get it moving.

        Parameters
        ----------
        estimated_speed:
            Estimated speed in km/h.  Pass ``None`` if speed is unavailable
            (e.g. speed tracking disabled), in which case all logic is skipped.
            Pass ``0.0`` when speed tracking is active and the truck is
            confirmed stationary.
        throttle:
            Current throttle command in ``[0.0, 1.0]``.  Used to detect
            when the truck is trying to move but is stuck.
        brake:
            Current brake command in ``[0.0, 1.0]``.  Stuck detection is
            suppressed while braking.
        """
        if self.cfg.gear.auto_transmission:
            return

        if estimated_speed is None:
            return

        now = time.monotonic()
        gcfg = self.cfg.gear

        # --- Stuck / reverse detection ---
        # Only active outside the post-reverse cooldown window.
        total_cooldown = gcfg.reverse_hold_duration_s + gcfg.stuck_detection_cooldown_s
        if now - self._last_reverse_time >= total_cooldown:
            if (
                estimated_speed < gcfg.stuck_speed_threshold_kph
                and throttle > gcfg.stuck_throttle_threshold
                and brake < gcfg.stuck_brake_threshold
            ):
                self._stuck_counter += 1
                if self._stuck_counter >= gcfg.stuck_frames_before_reverse:
                    logger.info(
                        "Truck stuck (throttle=%.2f, speed=%.1f km/h) — engaging reverse.",
                        throttle,
                        estimated_speed,
                    )
                    self.reverse()
                    self._last_reverse_time = now
                    self._stuck_counter = 0
            else:
                # Reset counter whenever the stuck condition is not met
                self._stuck_counter = 0

        # --- Normal gear shifting ---
        if now - self._last_shift < self._shift_cooldown_s:
            return  # still in cooldown

        if estimated_speed > gcfg.gear_up_speed:
            self.gear_up()
            self._last_shift = now
        elif 0.0 < estimated_speed < gcfg.gear_down_speed:
            self.gear_down()
            self._last_shift = now
