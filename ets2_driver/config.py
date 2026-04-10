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
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Screen / capture
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    """Screen-capture region for the main ETS2 viewport."""

    top: int = int(os.getenv("ETS2_CAP_TOP", "0"))
    left: int = int(os.getenv("ETS2_CAP_LEFT", "0"))
    width: int = int(os.getenv("ETS2_CAP_WIDTH", "2560"))
    height: int = int(os.getenv("ETS2_CAP_HEIGHT", "1080"))

    @property
    def as_dict(self) -> Dict[str, int]:
        return {"top": self.top, "left": self.left,
                "width": self.width, "height": self.height}


# ---------------------------------------------------------------------------
# Vision / lane detection
# ---------------------------------------------------------------------------

@dataclass
class LaneConfig:
    """HSV thresholds and ROI for white-line lane detection.

    Notes
    -----
    ``roi_top_fraction`` is expressed as a fraction of the captured frame
    height (must be in [0.0, 1.0]).  For a 2560×1080 capture the default
    value of 0.625 crops from pixel row 675 downward, keeping the lower
    portion of the road visible.
    """

    # Fraction of frame height that forms the ROI bottom boundary (1.0 = full)
    roi_top_fraction: float = float(os.getenv("ETS2_ROI_TOP", "0.625"))

    # HSV bounds for white lane markings
    lower_white: Tuple[int, int, int] = (0, 0, 180)
    upper_white: Tuple[int, int, int] = (180, 60, 255)

    def __post_init__(self) -> None:
        if not (0.0 <= self.roi_top_fraction <= 1.0):
            raise ValueError(
                f"roi_top_fraction (ETS2_ROI_TOP) must be in [0.0, 1.0], "
                f"got {self.roi_top_fraction!r}"
            )

    def roi_top_px(self, frame_height: int) -> int:
        """Return the ROI top boundary in pixels for the given frame height."""
        return int(self.roi_top_fraction * frame_height)


# ---------------------------------------------------------------------------
# GPS mini-map
# ---------------------------------------------------------------------------

@dataclass
class GpsConfig:
    """Crop region and HSV thresholds for the route-advisor mini-map.

    Defaults are tuned for the ETS2 2560×1080 layout.  Adjust if your
    UI scale or resolution differs.
    """

    # Pixel crop region within the captured frame (default ETS2 2560×1080 layout)
    top: int = int(os.getenv("ETS2_GPS_TOP", "0"))
    left: int = int(os.getenv("ETS2_GPS_LEFT", "0"))
    bottom: int = int(os.getenv("ETS2_GPS_BOTTOM", "1080"))
    right: int = int(os.getenv("ETS2_GPS_RIGHT", "2560"))
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

    # Errors below this threshold (pixels) produce zero steering output
    # (deadzone to prevent micro-corrections when nearly centred).
    deadzone_px: float = float(os.getenv("ETS2_PID_DEADZONE", "5.0"))

    # Exponential smoothing applied to the raw PID output before the rate
    # limiter.  Range [0, 1]; higher values are smoother but more sluggish.
    steer_smoothing: float = float(os.getenv("ETS2_STEER_SMOOTH", "0.7"))

    # Maximum steering change allowed per frame (rate limiter).
    # Prevents violent wheel snaps.  Range [0, 1].
    max_steer_rate: float = float(os.getenv("ETS2_STEER_RATE", "0.12"))


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

    # Coasting: reduce throttle gently when steering error is in this range
    # (between straight cruising and a full turn) to smooth the approach.
    coasting_threshold: int = int(os.getenv("ETS2_COAST_THR", "60"))
    coasting_throttle: float = float(os.getenv("ETS2_COAST_VAL", "0.55"))


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

    # Width of the "centre zone" as a fraction of the frame width.
    # Obstacles whose horizontal centre falls within ±(fraction/2) of the
    # screen centre are considered directly ahead.
    center_zone_fraction: float = float(os.getenv("ETS2_CENTER_ZONE", "0.33"))


# ---------------------------------------------------------------------------
# Speed-limit sign detection
# ---------------------------------------------------------------------------

@dataclass
class SpeedLimitConfig:
    """Configuration for the OpenCV speed-limit sign detector.

    The ROI defaults to the upper-left quadrant of a 1280×720 frame, which is
    where ETS2 renders road signs.  Adjust if your UI scale or resolution
    differs.
    """

    # Region-of-interest within the captured frame (pixels)
    roi_top: int    = int(os.getenv("ETS2_SL_ROI_TOP",    "0"))
    roi_bottom: int = int(os.getenv("ETS2_SL_ROI_BOTTOM", "360"))
    roi_left: int   = int(os.getenv("ETS2_SL_ROI_LEFT",   "0"))
    roi_right: int  = int(os.getenv("ETS2_SL_ROI_RIGHT",  "640"))

    # Hough circle transform parameters
    min_radius_px: int  = int(os.getenv("ETS2_SL_MIN_R",    "10"))
    max_radius_px: int  = int(os.getenv("ETS2_SL_MAX_R",    "60"))
    min_dist_px: int    = int(os.getenv("ETS2_SL_MIN_DIST", "50"))
    hough_param1: float = float(os.getenv("ETS2_SL_P1",     "100"))
    hough_param2: float = float(os.getenv("ETS2_SL_P2",     "25"))

    # Minimum confidence to accept a detection [0, 1]
    confidence_threshold: float = float(os.getenv("ETS2_SL_CONF", "0.40"))

    # Run detection every N frames (1 = every frame, 3 = every 3rd frame, …)
    detection_every_n_frames: int = int(os.getenv("ETS2_SL_SKIP", "3"))

    # Reference speed (km/h) used to scale the throttle cap.
    # At this speed, cruise_throttle is applied without reduction.
    # ETS2's unrestricted motorway speed is 130 km/h (default European limit).
    max_reference_speed_kph: int = int(os.getenv("ETS2_SL_MAX_SPEED", "130"))

    # Seconds to keep showing a detected speed limit after the sign leaves
    # frame.  Prevents dashboard flickering when the sign briefly disappears.
    # Set to 0 to disable persistence (always show live result only).
    decay_s: float = float(os.getenv("ETS2_SL_DECAY", "4.0"))


# ---------------------------------------------------------------------------
# Web dashboard
# ---------------------------------------------------------------------------

@dataclass
class DashboardConfig:
    """Configuration for the real-time web monitoring dashboard."""

    # Set to false to disable the dashboard entirely
    enabled: bool = os.getenv("ETS2_DASH_ENABLED", "true").lower() == "true"

    host: str = os.getenv("ETS2_DASH_HOST", "0.0.0.0")
    port: int = int(os.getenv("ETS2_DASH_PORT", "5000"))

    # Maximum Socket.IO push rate (Hz)
    update_hz: float = float(os.getenv("ETS2_DASH_HZ", "10"))

    # JPEG quality for the vision frame preview [0–100]
    frame_quality: int = int(os.getenv("ETS2_DASH_QUALITY", "60"))

    # Number of decision-log entries shown in the dashboard
    max_log_entries: int = int(os.getenv("ETS2_DASH_LOG", "20"))


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
# Speed tracking (HUD OCR)
# ---------------------------------------------------------------------------

@dataclass
class SpeedTrackingConfig:
    """Configuration for HUD OCR-based speed capture and velocity estimation.

    The OCR region should cover the in-game speedometer display (the large
    km/h number on the HUD).  Defaults are tuned for the ETS2 2560×1080
    layout with the speedometer in the bottom-right corner.

    Set ``ETS2_SPD_TRACK=true`` to enable (disabled by default to avoid
    unnecessary CPU cost when pytesseract is not available).
    """

    # Set to true to enable per-frame speed OCR
    enabled: bool = os.getenv("ETS2_SPD_TRACK", "false").lower() == "true"

    # Pixel coordinates of the speedometer crop within the captured frame
    # (defaults calibrated for 2560×1080; scale proportionally for other resolutions)
    roi_top:    int = int(os.getenv("ETS2_SPD_ROI_TOP",    "900"))
    roi_bottom: int = int(os.getenv("ETS2_SPD_ROI_BOTTOM", "1000"))
    roi_left:   int = int(os.getenv("ETS2_SPD_ROI_LEFT",   "2200"))
    roi_right:  int = int(os.getenv("ETS2_SPD_ROI_RIGHT",  "2560"))

    # Exponential smoothing weight applied to new OCR readings.
    # Range [0, 1]; higher = more smoothing (slower to respond to changes).
    smoothing_alpha: float = float(os.getenv("ETS2_SPD_ALPHA", "0.7"))

    # Number of frames to average when computing the velocity trend
    velocity_window: int = int(os.getenv("ETS2_SPD_VEL_WIN", "5"))

    # Maximum plausible speed (km/h); OCR readings above this are discarded
    max_speed_kph: float = float(os.getenv("ETS2_SPD_MAX", "200.0"))

    # Run OCR every N frames to limit CPU overhead (1 = every frame)
    ocr_every_n_frames: int = int(os.getenv("ETS2_SPD_SKIP", "3"))


# ---------------------------------------------------------------------------
# Adaptive PID gain scheduling
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveGainConfig:
    """Speed-dependent PID gain scheduling for the steering controller.

    Three anchor speed points define the gain profile.  Gains are linearly
    interpolated between them so there are no abrupt step changes.

    Low speed  (0 → speed_mid km/h): higher P-gain for precision; low D.
    Mid speed  (speed_mid km/h):     balanced defaults (original tuning).
    High speed (speed_high+ km/h):   lower P-gain; higher D for stability.

    All values can be overridden via the environment variables listed below.
    """

    # Enable/disable adaptive gain scheduling
    enabled: bool = os.getenv("ETS2_ADAPT_GAIN", "true").lower() == "true"

    # Speed range anchor points (km/h)
    speed_mid:  float = float(os.getenv("ETS2_SPEED_MID",  "50.0"))
    speed_high: float = float(os.getenv("ETS2_SPEED_HIGH", "80.0"))

    # Proportional gain at each anchor
    kp_low:  float = float(os.getenv("ETS2_PID_P_LOW",  "0.006"))
    kp_mid:  float = float(os.getenv("ETS2_PID_P_MID",  "0.004"))
    kp_high: float = float(os.getenv("ETS2_PID_P_HIGH", "0.002"))

    # Integral gain at each anchor
    ki_low:  float = float(os.getenv("ETS2_PID_I_LOW",  "0.0002"))
    ki_mid:  float = float(os.getenv("ETS2_PID_I_MID",  "0.0001"))
    ki_high: float = float(os.getenv("ETS2_PID_I_HIGH", "0.00005"))

    # Derivative gain at each anchor (higher at speed for stability)
    kd_low:  float = float(os.getenv("ETS2_PID_D_LOW",  "0.001"))
    kd_mid:  float = float(os.getenv("ETS2_PID_D_MID",  "0.002"))
    kd_high: float = float(os.getenv("ETS2_PID_D_HIGH", "0.004"))

    # Anti-windup integral clamp at each anchor (tighter at high speed)
    integral_max_low:  float = float(os.getenv("ETS2_INT_MAX_LOW",  "300.0"))
    integral_max_mid:  float = float(os.getenv("ETS2_INT_MAX_MID",  "200.0"))
    integral_max_high: float = float(os.getenv("ETS2_INT_MAX_HIGH", "100.0"))


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
    speed_limit: SpeedLimitConfig = field(default_factory=SpeedLimitConfig)
    gear: GearConfig = field(default_factory=GearConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    speed_tracking: SpeedTrackingConfig = field(default_factory=SpeedTrackingConfig)
    adaptive_gain: AdaptiveGainConfig = field(default_factory=AdaptiveGainConfig)
