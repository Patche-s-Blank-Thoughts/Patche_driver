"""Real-time web dashboard for monitoring the ETS2 AI driver.

Architecture
------------
:class:`DashboardServer` wraps a Flask + Flask-SocketIO application and runs
it in a **daemon thread** alongside the main driver loop.  The driver calls
:meth:`DashboardServer.push_telemetry` once per tick (or every N ticks to
respect :attr:`~ets2_driver.config.DashboardConfig.update_hz`); the server
broadcasts the payload to all connected browser clients via Socket.IO.

The browser client (``static/index.html``) connects automatically on load and
renders:

* Live camera feed (JPEG-encoded, base64 embedded).
* Steering, throttle, and brake gauges.
* Speed-limit sign indicator.
* Decision reasoning log with LLM advisory.
* Real-time telemetry metrics (FPS, lane error, GPS offset).
* Mini-map snapshot (cropped GPS sub-image).

Usage::

    server = DashboardServer(cfg)
    server.start()   # non-blocking – starts daemon thread
    # …later in the driver loop:
    server.push_telemetry(state)
    # …on shutdown:
    server.stop()
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import ETS2Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry state
# ---------------------------------------------------------------------------


@dataclass
class TelemetryState:
    """Snapshot of driver state for one dashboard update.

    All fields have sensible defaults so callers can fill only the fields they
    know about and leave the rest at ``None`` / zero.

    Attributes
    ----------
    frame:
        Current vision frame (BGR numpy array) or ``None``.
    gps_frame:
        Cropped GPS mini-map image (BGR) or ``None``.
    steering:
        Normalised steering output in ``[-1.0, 1.0]``.
    throttle:
        Normalised throttle in ``[0.0, 1.0]``.
    brake:
        Normalised brake in ``[0.0, 1.0]``.
    lane_error:
        Signed pixel offset of the detected lane centre.
    gps_error:
        Signed pixel offset from the GPS route line.
    combined_error:
        Blended steering error.
    speed_limit:
        Detected speed-limit value in km/h, or ``None``.
    speed_limit_confidence:
        Confidence of the speed-limit detection.
    avoidance_action:
        Latest obstacle-avoidance action token.
    llm_action:
        Latest LLM advisory action token.
    decision_reasons:
        Human-readable list of decision factors for the reasoning log.
    obstacles:
        List of ``[x1, y1, x2, y2]`` bounding boxes.
    fps:
        Measured loop FPS.
    timestamp:
        Unix timestamp of this state snapshot.
    debug_summary:
        Per-frame debug dict from :class:`~ets2_driver.debug_state.DebugState`
        (PID internals, raw vs smoothed values, coasting/emergency flags, etc.).
    obstacle_sides:
        Per-obstacle zone labels (``"left"``/``"center"``/``"right"``).
    lane_confidence:
        Fraction of expected lane pixels found in the ROI [0, 1].
    lane_target:
        Which lane the bot is currently targeting.
    """

    frame: Optional[np.ndarray] = None
    gps_frame: Optional[np.ndarray] = None
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    lane_error: float = 0.0
    gps_error: float = 0.0
    combined_error: float = 0.0
    speed_limit: Optional[int] = None
    speed_limit_confidence: float = 0.0
    avoidance_action: str = "none"
    llm_action: str = "CONTINUE"
    decision_reasons: List[str] = field(default_factory=list)
    obstacles: List[List[float]] = field(default_factory=list)
    fps: float = 0.0
    timestamp: float = field(default_factory=time.time)
    debug_summary: Dict[str, Any] = field(default_factory=dict)
    obstacle_sides: List[str] = field(default_factory=list)
    lane_confidence: float = 0.0
    lane_target: str = "center"


# ---------------------------------------------------------------------------
# Dashboard server
# ---------------------------------------------------------------------------


class DashboardServer:
    """Flask + Socket.IO dashboard server.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._state: TelemetryState = TelemetryState()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_push: float = 0.0
        self._push_interval: float = (
            1.0 / max(1.0, cfg.dashboard.update_hz)
        )

        # Deferred import so that flask is only required when dashboard is enabled
        self._app = None
        self._socketio = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the dashboard server in a background daemon thread."""
        if not self.cfg.dashboard.enabled:
            logger.info("Dashboard disabled (ETS2_DASH_ENABLED=false).")
            return

        self._setup_flask()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            name="DashboardServer",
            daemon=True,
        )
        self._thread.start()
        dcfg = self.cfg.dashboard
        logger.info(
            "Dashboard server started → http://%s:%d",
            dcfg.host if dcfg.host != "0.0.0.0" else "localhost",
            dcfg.port,
        )

    def stop(self) -> None:
        """Signal the dashboard server to stop."""
        self._running = False
        logger.info("Dashboard server stopped.")

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def push_telemetry(self, state: TelemetryState) -> None:
        """Push a new telemetry snapshot to connected clients.

        Rate-limited to :attr:`~ets2_driver.config.DashboardConfig.update_hz`.
        Safe to call from any thread.

        Parameters
        ----------
        state:
            Current driver state snapshot.
        """
        if not self.cfg.dashboard.enabled or self._socketio is None:
            return

        now = time.monotonic()
        if now - self._last_push < self._push_interval:
            return
        self._last_push = now

        with self._lock:
            self._state = state

        try:
            payload = self._encode_state(state)
            self._socketio.emit("telemetry", payload, namespace="/")
        except Exception as exc:
            logger.debug("Dashboard emit failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal — Flask setup
    # ------------------------------------------------------------------

    def _setup_flask(self) -> None:
        """Create the Flask app and register routes."""
        from flask import Flask, send_from_directory  # type: ignore
        from flask_socketio import SocketIO  # type: ignore

        static_dir = os.path.join(os.path.dirname(__file__), "static")
        app = Flask(__name__, static_folder=static_dir)
        app.config["SECRET_KEY"] = "ets2-ai-driver-dashboard"

        socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode="threading",
            logger=False,
            engineio_logger=False,
        )

        @app.route("/")
        def index():  # type: ignore[misc]
            return send_from_directory(static_dir, "index.html")

        @app.route("/api/state")
        def api_state():  # type: ignore[misc]
            from flask import jsonify  # type: ignore
            with self._lock:
                state = self._state
            return jsonify(self._encode_state(state))

        @socketio.on("connect")
        def on_connect():  # type: ignore[misc]
            logger.debug("Dashboard client connected.")

        @socketio.on("disconnect")
        def on_disconnect():  # type: ignore[misc]
            logger.debug("Dashboard client disconnected.")

        self._app = app
        self._socketio = socketio

    def _run_server(self) -> None:
        """Entry point for the server daemon thread."""
        dcfg = self.cfg.dashboard
        try:
            self._socketio.run(
                self._app,
                host=dcfg.host,
                port=dcfg.port,
                # allow_unsafe_werkzeug is required when running Flask's
                # development server (Werkzeug) inside a non-main thread.
                # This is intentional — the dashboard is an internal monitoring
                # tool, not a public-facing service.  For a production
                # deployment, replace with a proper WSGI server (e.g. gunicorn).
                allow_unsafe_werkzeug=True,
                log_output=False,
            )
        except Exception as exc:
            logger.error("Dashboard server crashed: %s", exc)

    # ------------------------------------------------------------------
    # Payload encoding
    # ------------------------------------------------------------------

    def _encode_state(self, state: TelemetryState) -> Dict[str, Any]:
        """Convert a :class:`TelemetryState` into a JSON-serialisable dict."""
        dcfg = self.cfg.dashboard
        payload: Dict[str, Any] = {
            "steering": round(state.steering, 4),
            "throttle": round(state.throttle, 4),
            "brake": round(state.brake, 4),
            "lane_error": round(state.lane_error, 2),
            "gps_error": round(state.gps_error, 2),
            "combined_error": round(state.combined_error, 2),
            "speed_limit": state.speed_limit,
            "speed_limit_confidence": round(state.speed_limit_confidence, 3),
            "avoidance_action": state.avoidance_action,
            "llm_action": state.llm_action,
            "decision_reasons": state.decision_reasons,
            "obstacle_count": len(state.obstacles),
            "fps": round(state.fps, 1),
            "timestamp": state.timestamp,
            "frame_b64": None,
            "gps_b64": None,
            # Enhanced debug fields
            "debug_summary": state.debug_summary,
            "obstacle_sides": state.obstacle_sides,
            "lane_confidence": round(state.lane_confidence, 3),
            "lane_target": state.lane_target,
        }

        if state.frame is not None:
            payload["frame_b64"] = self._encode_frame(
                state.frame, dcfg.frame_quality
            )

        if state.gps_frame is not None:
            # GPS mini-map shown at full quality (it's tiny)
            payload["gps_b64"] = self._encode_frame(state.gps_frame, 90)

        return payload

    @staticmethod
    def _encode_frame(frame: np.ndarray, quality: int) -> Optional[str]:
        """JPEG-encode *frame* and return a base64 data-URI string."""
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                return None
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception as exc:
            logger.debug("Frame encode failed: %s", exc)
            return None
