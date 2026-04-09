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

        if cfg.gear.auto_transmission:
            logger.info("GearShifter: ETS2 automatic transmission enabled — "
                        "keyboard shifting is disabled.")
            return

        try:
            import keyboard  # type: ignore  # noqa: F401
            self._keyboard_available = True
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

    def update(self, estimated_speed: Optional[float]) -> None:
        """Automatically shift up or down based on the estimated truck speed.

        Parameters
        ----------
        estimated_speed:
            Estimated speed in km/h.  Pass ``None`` if speed is unavailable
            (e.g. no OCR), in which case shifting is skipped.
        """
        if self.cfg.gear.auto_transmission:
            return

        if estimated_speed is None:
            return

        now = time.monotonic()
        if now - self._last_shift < self._shift_cooldown_s:
            return  # still in cooldown

        gcfg = self.cfg.gear
        if estimated_speed > gcfg.gear_up_speed:
            self.gear_up()
            self._last_shift = now
        elif estimated_speed < gcfg.gear_down_speed:
            self.gear_down()
            self._last_shift = now
