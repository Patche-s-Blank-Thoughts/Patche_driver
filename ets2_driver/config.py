"""Centralised configuration and tuning parameters for the ETS2 AI Driver.

All values can be overridden via environment variables with the same name
(converted to uppercase), e.g.::

    ETS2_KP=0.005 python ets2_main.py

Typical tuning workflow:
1. Capture a few frames and adjust MONITOR / GPS_REGION so they match your
   in-game layout (resolution, UI scale, route-advisor position).
2. Increase KP / KI / KD until the truck tracks the lane without oscillating.
3. Widen OBSTACLE_BRAKE_WIDTH_PX or lower BRAKE_HARD if braking is too
   aggressive.
4. Adjust SPEED_CRUISE / SPEED_TURN to taste.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Screen / capture
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    """Screen-capture region for the main ETS2 viewport."""

    top: int = int(os.getenv("ETS2_CAP_TOP", "0"))
    left: int = int(os.getenv("ETS2_CAP_LEFT", "0"))
    width: int = int(os.getenv("ETS2_CAP_WIDTH", "1280"))
    height: int = int(os.getenv("ETS2_CAP_HEIGHT", "720"))

    @property
    def as_dict(self) -> Dict[str, int]:
        return {"top": self.top, "left": self.left,
                "width": self.width, "height": self.height}


# ---------------------------------------------------------------------------
# Vision / lane detection
# ---------------------------------------------------------------------------

@dataclass
class LaneConfig:
    """HSV thresholds and ROI for white-line lane detection."""

    # Fraction of frame height that forms the ROI bottom boundary (1.0 = full)
    roi_top_fraction: float = float(os.getenv("ETS2_ROI_TOP", "0.625"))

    # HSV bounds for white lane markings
    lower_white: Tuple[int, int, int] = (0, 0, 180)
    upper_white: Tuple[int, int, int] = (180, 60, 255)


# ---------------------------------------------------------------------------
# GPS mini-map
# ---------------------------------------------------------------------------

@dataclass
class GpsConfig:
    """Crop region and HSV thresholds for the route-advisor mini-map."""

    # Pixel offsets from the bottom-right corner (default ETS2 1280×720 layout)
    top: int = int(os.getenv("ETS2_GPS_TOP", "520"))
    left: int = int(os.getenv("ETS2_GPS_LEFT", "1050"))
    bottom: int = int(os.getenv("ETS2_GPS_BOTTOM", "720"))
    right: int = int(os.getenv("ETS2_GPS_RIGHT", "1280"))

    # HSV bounds for the red route line drawn on the mini-map
    lower_red1: Tuple[int, int, int] = (0, 120, 100)
    upper_red1: Tuple[int, int, int] = (10, 255, 255)
    lower_red2: Tuple[int, int, int] = (170, 120, 100)
    upper_red2: Tuple[int, int, int] = (180, 255, 255)

    # Weight applied when blending GPS guidance with lane-follow error
    blend_weight: float = float(os.getenv("ETS2_GPS_WEIGHT", "0.3"))


# ---------------------------------------------------------------------------
# PID steering
# ---------------------------------------------------------------------------

@dataclass
class PidConfig:
    """Proportional-Integral-Derivative gains for the steering controller."""

    kp: float = float(os.getenv("ETS2_KP", "0.004"))
    ki: float = float(os.getenv("ETS2_KI", "0.0001"))
    kd: float = float(os.getenv("ETS2_KD", "0.002"))

    # Maximum allowed integral accumulation (anti-windup)
    integral_max: float = float(os.getenv("ETS2_INT_MAX", "200.0"))


# ---------------------------------------------------------------------------
# Speed control
# ---------------------------------------------------------------------------

@dataclass
class SpeedConfig:
    """Throttle and brake levels for different driving states."""

    # Normalised throttle [0, 1] for straight / cruising
    cruise_throttle: float = float(os.getenv("ETS2_CRUISE_THR", "0.7"))

    # Normalised throttle during sharp turns
    turn_throttle: float = float(os.getenv("ETS2_TURN_THR", "0.3"))

    # Light brake applied during turns
    turn_brake: float = float(os.getenv("ETS2_TURN_BRK", "0.2"))

    # Hard brake applied when an obstacle is dangerously close
    hard_brake: float = float(os.getenv("ETS2_HARD_BRK", "0.7"))

    # Steering-error threshold (pixels) above which "turn" logic activates
    turn_error_threshold: int = int(os.getenv("ETS2_TURN_ERR", "120"))

    # Exponential smoothing factor for throttle output [0, 1]; higher = smoother
    throttle_smoothing: float = float(os.getenv("ETS2_THR_SMOOTH", "0.8"))

    # Exponential smoothing factor for brake output [0, 1]; higher = smoother
    brake_smoothing: float = float(os.getenv("ETS2_BRK_SMOOTH", "0.85"))

    # Maximum throttle increase per frame (acceleration ramp rate)
    speed_ramp_rate: float = float(os.getenv("ETS2_SPD_RAMP", "0.05"))

    # Maximum brake increase per frame (brake ramp rate)
    brake_ramp_rate: float = float(os.getenv("ETS2_BRK_RAMP", "0.08"))

    # Steering-error magnitude (px) that triggers emergency braking
    emergency_brake_threshold: int = int(os.getenv("ETS2_EMRG_THR", "150"))

    # Brake pressure applied during emergency braking
    emergency_brake_value: float = float(os.getenv("ETS2_EMRG_BRK", "0.95"))


# ---------------------------------------------------------------------------
# Obstacle / YOLO detection
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """YOLO model settings and obstacle response thresholds."""

    # Path to YOLOv8 weights; 'yolov8n.pt' downloads automatically on first run
    model_path: str = os.getenv("ETS2_YOLO_MODEL", "yolov8n.pt")

    # COCO class IDs considered as road obstacles
    # 2=car, 3=motorcycle, 5=bus, 7=truck
    obstacle_classes: Tuple[int, ...] = (2, 3, 5, 7)

    # Minimum bounding-box width (px) to trigger emergency braking
    brake_width_px: int = int(os.getenv("ETS2_BRAKE_WIDTH", "200"))

    # Steering offset applied when swerving around an obstacle [0, 1]
    swerve_amount: float = float(os.getenv("ETS2_SWERVE", "0.4"))

    # Detection confidence threshold
    conf_threshold: float = float(os.getenv("ETS2_CONF", "0.4"))

    # Run YOLO every N vision frames to reduce CPU/GPU load
    inference_every_n_frames: int = int(os.getenv("ETS2_YOLO_SKIP", "2"))


# ---------------------------------------------------------------------------
# Gear shifting
# ---------------------------------------------------------------------------

@dataclass
class GearConfig:
    """Keyboard keys used for manual gear-shifting and speed thresholds."""

    gear_up_key: str = os.getenv("ETS2_GEAR_UP", "shift")
    gear_down_key: str = os.getenv("ETS2_GEAR_DOWN", "ctrl")
    reverse_key: str = os.getenv("ETS2_REVERSE_KEY", "r")

    # Speed (km/h, estimated) thresholds – only used when auto_transmission=False
    gear_up_speed: int = int(os.getenv("ETS2_GEAR_UP_SPD", "20"))
    gear_down_speed: int = int(os.getenv("ETS2_GEAR_DOWN_SPD", "10"))

    # Set True if you have automatic transmission enabled in ETS2 settings
    auto_transmission: bool = os.getenv("ETS2_AUTO_TRANS", "true").lower() == "true"


# ---------------------------------------------------------------------------
# LLM planner
# ---------------------------------------------------------------------------

@dataclass
class LlmConfig:
    """Settings for the optional LLM high-level planner."""

    enabled: bool = os.getenv("ETS2_LLM_ENABLED", "false").lower() == "true"

    # How often (seconds) to call the LLM
    call_interval: float = float(os.getenv("ETS2_LLM_INTERVAL", "1.0"))

    # Passed straight to the Agent in index.py / agent/agent.py
    model_name: str = os.getenv("ETS2_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # Maximum tokens per LLM response
    max_new_tokens: int = int(os.getenv("ETS2_LLM_TOKENS", "128"))


# ---------------------------------------------------------------------------
# Loop / timing
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    """Target frame-rate and thread configuration for the main loops."""

    # Desired frames per second for the vision + control loop
    fps: int = int(os.getenv("ETS2_FPS", "30"))

    # Whether to show a live OpenCV debug window
    show_debug: bool = os.getenv("ETS2_DEBUG", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class ETS2Config:
    """Top-level configuration object that bundles all sub-configs."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    pid: PidConfig = field(default_factory=PidConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    gear: GearConfig = field(default_factory=GearConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
