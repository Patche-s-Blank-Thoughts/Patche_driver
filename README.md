# Patche_driver

## ETS2 AI Driving Bot

An advanced autonomous driving bot for **Euro Truck Simulator 2** with
computer-vision perception, speed-limit sign detection, and a real-time
web-based monitoring dashboard.

---

### Features

| Module | Description |
|---|---|
| **Lane / GPS detection** | White-line lane detection + red-route GPS mini-map reading blended via PID steering |
| **Speed-limit signs** | HoughCircles + optional pytesseract OCR to detect and respect speed-limit signs |
| **Obstacle avoidance** | YOLOv8 vehicle/truck/bus detection with swerve-and-brake avoidance logic |
| **Web Dashboard** | Flask + Socket.IO real-time dashboard showing live camera feed, control gauges, decision log, GPS mini-map, and telemetry |
| **LLM Planner** | Optional Qwen 2.5 high-level advisory (CONTINUE / BRAKE / OVERTAKE / STOP …) |
| **Gear Shifter** | Automatic transmission support or keyboard-based gear shifting |

---

### Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   For speed-limit OCR, also install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
   on your system (`sudo apt install tesseract-ocr` / Windows installer).

2. **Configure vJoy** (Windows) — install and configure a vJoy virtual joystick device.

3. **Launch ETS2** in windowed or borderless mode at **2560 × 1080**.

4. **Run the bot**

   ```bash
   python ets2_main.py
   ```

5. **Open the dashboard** in your browser:

   ```
   http://localhost:5000
   ```

6. Switch focus to the ETS2 window.  The bot will start driving within a second or two.

Press **Ctrl-C** (or **Q** in the debug window) to stop.

---

### Dashboard

The dashboard runs automatically on port **5000**.  It shows:

- **Live camera feed** — exactly what the AI sees, with obstacle badges.
- **Steering / Throttle / Brake gauges** — animated arc gauges updated in real-time.
- **Speed-limit sign** — detected sign value and confidence score.
- **GPS mini-map** — the route-advisor crop used for GPS guidance.
- **AI Planner** — current LLM advisory action pill.
- **Thought Process log** — human-readable decision reasoning per frame.
- **Telemetry table** — raw numeric values (lane error, GPS offset, etc.).

To disable the dashboard:

```bash
python ets2_main.py --no-dashboard
# or
ETS2_DASH_ENABLED=false python ets2_main.py
```

To change the port:

```bash
python ets2_main.py --dashboard-port 8080
```

---

### CLI Flags

```
python ets2_main.py [--debug] [--fps N] [--llm] [--no-dashboard] [--dashboard-port PORT]
```

| Flag | Description |
|---|---|
| `--debug` | Show OpenCV debug overlay window |
| `--fps N` | Target loop FPS (default 30) |
| `--llm` | Enable LLM high-level planner |
| `--no-dashboard` | Disable the web dashboard |
| `--dashboard-port PORT` | Dashboard HTTP port (default 5000) |

---

### Environment Variables

All parameters can be overridden with environment variables.  See
`ets2_driver/config.py` for the full list.  Key ones:

| Variable | Default | Description |
|---|---|---|
| `ETS2_DASH_ENABLED` | `true` | Enable/disable dashboard |
| `ETS2_DASH_PORT` | `5000` | Dashboard port |
| `ETS2_DASH_HZ` | `10` | Dashboard update rate (Hz) |
| `ETS2_DASH_QUALITY` | `60` | JPEG quality for frame preview |
| `ETS2_SL_CONF` | `0.40` | Speed-limit detection confidence threshold |
| `ETS2_SL_SKIP` | `3` | Run speed-limit detection every N frames |
| `ETS2_KP` / `ETS2_KI` / `ETS2_KD` | `0.004 / 0.0001 / 0.002` | PID gains |
| `ETS2_DEBUG` | `false` | Show OpenCV debug window |
| `ETS2_LLM_ENABLED` | `false` | Enable LLM planner |

---

### Architecture

```
ets2_main.py
└── ETS2Driver (driver.py)
    ├── VisionSystem       — screen capture, lane detection, GPS reading
    ├── SpeedLimitDetector — HoughCircles + OCR speed-limit sign reader
    ├── ObstacleDetector   — YOLOv8 vehicle/obstacle detection
    ├── PIDSteering        — PID steering controller
    ├── SpeedController    — rule-based throttle/brake with smoothing
    ├── VJoyController     — virtual joystick axis output
    ├── GearShifter        — automatic gear shifting
    ├── LLMPlanner         — optional Qwen 2.5 high-level advisory
    └── DashboardServer    — Flask + Socket.IO real-time web dashboard
```
