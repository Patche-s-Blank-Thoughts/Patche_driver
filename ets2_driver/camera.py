"""Camera angle manager for ETS2 — keyboard-controlled camera switching.

ETS2 uses number keys to cycle through camera views.  This module maps those
keys to human-readable camera names and provides a simple API for switching
views programmatically or in response to keyboard events captured in the debug
window.

Default key bindings (mirror the ETS2 default keybindings):
    1 — Front camera (default driving view)
    2 — Left-mirror camera
    3 — Right-mirror camera
    4 — Top-down / bird's-eye view (useful for parking)
"""

from __future__ import annotations

import logging
from typing import Dict

from .config import ETS2Config

logger = logging.getLogger(__name__)

# Map camera-ID → key string sent to ETS2
CAMERA_KEYS: Dict[int, str] = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
}

# Human-readable names for logging / UI
CAMERA_NAMES: Dict[int, str] = {
    1: "Front",
    2: "Left Mirror",
    3: "Right Mirror",
    4: "Top-down",
}


class CameraManager:
    """Manage ETS2 camera angles via keyboard shortcuts.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.

    Notes
    -----
    Camera switching is implemented by pressing the corresponding number key.
    If the ``keyboard`` package is unavailable the camera ID is still tracked
    internally (so other code can query :attr:`current_camera`) but no key is
    sent to the game.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._current_camera: int = 1
        self._keyboard_available: bool = False

        try:
            import keyboard  # type: ignore
            # Verify the required API is present (guards against partial stubs)
            if not callable(getattr(keyboard, "press_and_release", None)):
                raise ImportError("keyboard.press_and_release not available")
            self._keyboard_available = True
            logger.info("CameraManager ready (front/left/right/top-down on keys 1–4).")
        except Exception as exc:
            logger.warning(
                "keyboard module not available (%s) — camera switching disabled.",
                exc,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_camera(self) -> int:
        """Currently active camera index (1–4, read-only)."""
        return self._current_camera

    @property
    def current_camera_name(self) -> str:
        """Human-readable name of the active camera view."""
        return CAMERA_NAMES.get(self._current_camera, f"Camera {self._current_camera}")

    # ------------------------------------------------------------------
    # Camera switching
    # ------------------------------------------------------------------

    def switch_to(self, camera_id: int) -> None:
        """Switch to the specified camera view.

        Parameters
        ----------
        camera_id:
            Camera index: 1=front, 2=left mirror, 3=right mirror, 4=top-down.
            Unknown IDs are logged and ignored.
        """
        if camera_id not in CAMERA_KEYS:
            logger.warning(
                "Unknown camera id %d — valid ids are %s.",
                camera_id, sorted(CAMERA_KEYS),
            )
            return

        if camera_id == self._current_camera:
            return

        self._current_camera = camera_id

        if not self._keyboard_available:
            logger.debug(
                "Camera switch to %s requested (no-op — keyboard unavailable).",
                CAMERA_NAMES.get(camera_id),
            )
            return

        import keyboard  # type: ignore
        keyboard.press_and_release(CAMERA_KEYS[camera_id])
        logger.info(
            "Camera switched to %s (key=%s).",
            CAMERA_NAMES.get(camera_id), CAMERA_KEYS[camera_id],
        )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def front(self) -> None:
        """Switch to the front / default driving camera."""
        self.switch_to(1)

    def left_mirror(self) -> None:
        """Switch to the left-mirror camera."""
        self.switch_to(2)

    def right_mirror(self) -> None:
        """Switch to the right-mirror camera."""
        self.switch_to(3)

    def top_down(self) -> None:
        """Switch to the top-down / bird's-eye view (useful for parking)."""
        self.switch_to(4)

    # ------------------------------------------------------------------
    # OpenCV key-event handler
    # ------------------------------------------------------------------

    def handle_key(self, key_code: int) -> bool:
        """Handle an OpenCV ``waitKey`` key code for camera switching.

        Call this with the return value of ``cv2.waitKey()`` so the debug
        window's number keys switch cameras in addition to the ``q`` quit key.

        Parameters
        ----------
        key_code:
            Raw key code from ``cv2.waitKey() & 0xFF``.

        Returns
        -------
        bool
            ``True`` if the key was recognised and a camera switch was
            performed; ``False`` otherwise.
        """
        # ord("1") == 49, ord("4") == 52
        for cam_id, cam_key in CAMERA_KEYS.items():
            if key_code == ord(cam_key):
                self.switch_to(cam_id)
                return True
        return False
