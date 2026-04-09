#!/usr/bin/env python3
"""ets2_main.py — entry point for the ETS2 AI Driving Bot.

Quick start
-----------
1. Install dependencies::

       pip install -r requirements.txt

2. Make sure vJoy is installed and configured (see README).

3. Launch ETS2 in windowed or borderless mode at 1280×720.

4. Run the bot::

       python ets2_main.py

5. Switch focus to the ETS2 window.  The bot will start driving within
   a second or two.

Press  Ctrl-C  (or  Q  in the debug window) to stop.

Tuning
------
All parameters are configurable via environment variables.  For example::

    ETS2_KP=0.005 ETS2_DEBUG=true python ets2_main.py

See ``ets2_driver/config.py`` for the full list of variables.
"""

import logging
import sys

# ---------------------------------------------------------------------------
# Configure logging before importing anything else
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("ets2_main")

# ---------------------------------------------------------------------------
# Import and run
# ---------------------------------------------------------------------------

from ets2_driver import ETS2Driver
from ets2_driver.config import ETS2Config


def main() -> None:
    """Parse any CLI flags, create the driver, and start the loop."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ETS2 AI Driving Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show OpenCV debug window (overrides ETS2_DEBUG env var)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Target frames per second (overrides ETS2_FPS env var)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM high-level planner (overrides ETS2_LLM_ENABLED env var)",
    )
    args = parser.parse_args()

    cfg = ETS2Config()

    if args.debug:
        cfg.loop.show_debug = True
    if args.fps is not None:
        cfg.loop.fps = args.fps
    if args.llm:
        cfg.llm.enabled = True

    logger.info("Starting ETS2 AI Driver  (FPS=%d, debug=%s, LLM=%s)",
                cfg.loop.fps, cfg.loop.show_debug, cfg.llm.enabled)

    driver = ETS2Driver(cfg)

    try:
        driver.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        driver.stop()
        logger.info("Bye!")


if __name__ == "__main__":
    main()
