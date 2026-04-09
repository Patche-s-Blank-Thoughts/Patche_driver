"""ETS2 AI Driver — production-ready autonomous driving bot for Euro Truck Simulator 2.

Modules
-------
config      Centralised tuning parameters and constants.
vision      Screen capture, lane detection, GPS mini-map reading.
detection   YOLO-based obstacle / vehicle detection.
controller  vJoy axis control (steering, throttle, brake).
gears       Automatic gear-shifting via keyboard automation.
llm_planner Optional LLM high-level planner (1-2 Hz).
driver      Top-level ETS2Driver class that wires all modules together.
"""

from .driver import ETS2Driver

__all__ = ["ETS2Driver"]
