"""ETS2 AI Driver — production-ready autonomous driving bot for Euro Truck Simulator 2.

Modules
-------
config      Centralised tuning parameters and constants.
vision      Screen capture, lane detection, GPS mini-map reading.
detection   YOLO-based obstacle / vehicle detection.
speed_limit OpenCV-based speed-limit sign detection with optional OCR.
controller  vJoy axis control (steering, throttle, brake).
gears       Automatic gear-shifting via keyboard automation.
llm_planner Optional LLM high-level planner (1-2 Hz).
dashboard   Real-time Flask + Socket.IO web monitoring dashboard.
driver      Top-level ETS2Driver class that wires all modules together.
"""

from .dashboard import DashboardServer, TelemetryState
from .driver import ETS2Driver
from .speed_limit import SpeedLimitDetector, SpeedLimitResult

__all__ = [
    "ETS2Driver",
    "SpeedLimitDetector",
    "SpeedLimitResult",
    "DashboardServer",
    "TelemetryState",
]
