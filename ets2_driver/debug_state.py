"""Per-frame diagnostic state for the ETS2 AI driving bot.

The :class:`DebugState` maintains a rolling window of the last 60 frames so
that steering, throttle, brake, PID internals, lane confidence, obstacle
positions, and control decisions can be inspected post-run or streamed to the
dashboard.

Usage::

    ds = DebugState()

    # At the start of each tick:
    dbg = ds.new_frame()
    dbg.lane_error = 42.0
    dbg.pid_p = kp * error
    # … fill other fields …
    ds.commit()

    # Export for offline analysis:
    with open("debug_log.json", "w") as fh:
        fh.write(ds.export_json())
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class FrameDebug:
    """Diagnostic snapshot for a single driving frame.

    Attributes
    ----------
    frame_id:
        Monotonically increasing frame counter.
    timestamp:
        Unix timestamp at capture time.
    raw_steer:
        PID output *before* output smoothing / rate-limiting.
    steer:
        Final steering value sent to vJoy.
    raw_throttle:
        Target throttle *before* exponential smoothing.
    throttle:
        Final throttle value sent to vJoy.
    raw_brake:
        Target brake *before* exponential smoothing.
    brake:
        Final brake value sent to vJoy.
    pid_p / pid_i / pid_d:
        Individual PID term contributions to ``raw_steer``.
    pid_integral:
        Current integrator state (for windup monitoring).
    lane_error:
        Pixel offset from the detected lane centre to the screen centre.
    lane_confidence:
        Fraction of expected lane-pixels found in the ROI [0, 1].
    lane_candidates:
        Up to three candidate lane-centre x-positions [left, centre, right].
    lane_target:
        Which lane the bot is currently targeting (``"left"``/``"center"``/
        ``"right"``).
    gps_error:
        Signed pixel offset from the GPS route centre.
    gps_blend_weight:
        GPS weight used for blending in this frame.
    combined_error:
        Blended steering error fed to the PID.
    obstacle_count:
        Number of obstacles detected in this frame.
    obstacle_sides:
        Per-obstacle zone labels (``"left"``/``"center"``/``"right"``).
    obstacle_distances:
        Per-obstacle normalised bounding-box width (proxy for distance).
    avoidance_action:
        Avoidance action token (e.g. ``"swerve_left"``).
    llm_action:
        High-level LLM advisory token.
    speed_limit:
        Detected speed-limit value in km/h, or ``None``.
    in_coasting:
        ``True`` when coasting logic reduced throttle before a curve.
    in_emergency:
        ``True`` when the emergency-brake threshold was exceeded.
    """

    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)

    # Raw (pre-smoothing) vs. final output control values
    raw_steer: float = 0.0
    steer: float = 0.0
    raw_throttle: float = 0.0
    throttle: float = 0.0
    raw_brake: float = 0.0
    brake: float = 0.0

    # PID internals
    pid_p: float = 0.0
    pid_i: float = 0.0
    pid_d: float = 0.0
    pid_integral: float = 0.0

    # Lane detection
    lane_error: float = 0.0
    lane_confidence: float = 0.0
    lane_candidates: List[int] = field(default_factory=list)
    lane_target: str = "center"

    # GPS blending
    gps_error: float = 0.0
    gps_blend_weight: float = 0.0
    combined_error: float = 0.0

    # Obstacles
    obstacle_count: int = 0
    obstacle_sides: List[str] = field(default_factory=list)
    obstacle_distances: List[float] = field(default_factory=list)
    avoidance_action: str = "none"

    # High-level state
    llm_action: str = "CONTINUE"
    speed_limit: Optional[int] = None
    in_coasting: bool = False
    in_emergency: bool = False


class DebugState:
    """Rolling per-frame diagnostic store with a configurable history window.

    Parameters
    ----------
    history_len:
        Number of frames to retain (default 60).
    """

    HISTORY_LEN: int = 60

    def __init__(self, history_len: int = HISTORY_LEN) -> None:
        self._frame_id: int = 0
        self._history: Deque[FrameDebug] = collections.deque(maxlen=history_len)
        self.current: FrameDebug = FrameDebug()

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def new_frame(self) -> FrameDebug:
        """Initialise a new :class:`FrameDebug` and return it for filling.

        Call this at the very start of each driving tick, fill the returned
        object throughout the tick, then call :meth:`commit` at the end.
        """
        self._frame_id += 1
        self.current = FrameDebug(frame_id=self._frame_id)
        return self.current

    def commit(self) -> None:
        """Append the current frame snapshot to the rolling history."""
        self._history.append(self.current)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[FrameDebug]:
        """Return a list snapshot of the rolling history (oldest first)."""
        return list(self._history)

    def export_json(self) -> str:
        """Serialise the full rolling history to a JSON string.

        The returned string can be saved to disk for post-run analysis::

            with open("debug_log.json", "w") as fh:
                fh.write(debug_state.export_json())
        """
        data = [asdict(f) for f in self._history]
        return json.dumps(data, indent=2)

    def summary(self) -> Dict[str, Any]:
        """Return a dashboard-friendly dict for the current frame.

        Values are rounded to keep the payload compact.
        """
        f = self.current
        return {
            "frame_id": f.frame_id,
            "raw_steer": round(f.raw_steer, 4),
            "raw_throttle": round(f.raw_throttle, 4),
            "raw_brake": round(f.raw_brake, 4),
            "pid_p": round(f.pid_p, 4),
            "pid_i": round(f.pid_i, 4),
            "pid_d": round(f.pid_d, 4),
            "pid_integral": round(f.pid_integral, 4),
            "lane_confidence": round(f.lane_confidence, 3),
            "lane_candidates": f.lane_candidates,
            "lane_target": f.lane_target,
            "gps_blend_weight": round(f.gps_blend_weight, 3),
            "obstacle_sides": f.obstacle_sides,
            "obstacle_distances": [round(d, 3) for d in f.obstacle_distances],
            "in_coasting": f.in_coasting,
            "in_emergency": f.in_emergency,
        }
