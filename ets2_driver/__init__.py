"""ETS2 AI Driver — production-ready autonomous driving bot for Euro Truck Simulator 2.

Modules
-------
config          Centralised tuning parameters and constants.
vision          Screen capture, lane detection, GPS mini-map reading.
detection       YOLO-based obstacle / vehicle detection.
speed_limit     OpenCV-based speed-limit sign detection with optional OCR.
controller      vJoy axis control (steering, throttle, brake).
gears           Automatic gear-shifting via keyboard automation.
camera          Camera-angle manager (keyboard shortcuts 1–4).
llm_planner     Optional LLM high-level planner (1-2 Hz).
parking_planner Parking-lot detection and road-escape planner.
dashboard       Real-time Flask + Socket.IO web monitoring dashboard.
debug_state     Per-frame diagnostic state with rolling 60-frame history.
driver          Top-level ETS2Driver class that wires all modules together.
"""

from .camera import CameraManager
from .dashboard import DashboardServer, TelemetryState
from .debug_state import DebugState, FrameDebug
from .driver import ETS2Driver
from .parking_planner import ParkingLotPlanner
from .speed_limit import SpeedLimitDetector, SpeedLimitResult

__all__ = [
    "CameraManager",
    "ETS2Driver",
    "ParkingLotPlanner",
    "SpeedLimitDetector",
    "SpeedLimitResult",
    "DashboardServer",
    "TelemetryState",
    "DebugState",
    "FrameDebug",
]
