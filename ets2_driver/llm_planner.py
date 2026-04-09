"""Optional LLM high-level planner (runs at 1–2 Hz).

The planner sends a short situational summary to the local Qwen 2.5 model
(reusing the existing :class:`~agent.Agent` infrastructure in this repo) and
parses a high-level *action* token from the response.

Supported action tokens (case-insensitive):
    ``CONTINUE``    — do nothing, keep current behaviour
    ``BRAKE``       — apply moderate braking
    ``OVERTAKE``    — signal intent to overtake the vehicle ahead
    ``EXIT_RIGHT``  — prepare to take the next motorway exit on the right
    ``EXIT_LEFT``   — prepare to take the next motorway exit on the left
    ``STOP``        — bring the truck to a safe halt

The planner is intentionally *advisory*: the fast control loop in
:class:`~ets2_driver.driver.ETS2Driver` can ignore or override any action.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional

from .config import ETS2Config

logger = logging.getLogger(__name__)

# Recognised high-level action tokens
KNOWN_ACTIONS = frozenset(
    {"CONTINUE", "BRAKE", "OVERTAKE", "EXIT_RIGHT", "EXIT_LEFT", "STOP"}
)


class LLMPlanner:
    """Wraps the repo's Agent class to provide ETS2 driving decisions.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.

    Notes
    -----
    The agent is loaded lazily on the first call to :meth:`query` so that
    importing the module does not trigger a heavy model load.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg
        self._agent = None
        self._last_action: str = "CONTINUE"
        self._last_call_time: float = 0.0

    def _ensure_agent(self) -> None:
        """Lazily initialise the Qwen agent (downloads weights on first run)."""
        if self._agent is not None:
            return
        try:
            # Override the model name from the LLM config
            import config as repo_config  # type: ignore
            original = repo_config.MODEL_NAME
            repo_config.MODEL_NAME = self.cfg.llm.model_name
            from agent import Agent  # type: ignore
            self._agent = Agent()
            repo_config.MODEL_NAME = original
            logger.info("LLMPlanner agent loaded (%s).", self.cfg.llm.model_name)
        except Exception as exc:
            logger.warning("LLMPlanner could not load agent: %s. "
                           "High-level planning is disabled.", exc)

    @staticmethod
    def _build_prompt(
        lane_error: float,
        obstacle_action: str,
        gps_error: float,
        speed: Optional[float],
    ) -> str:
        """Build a concise situational prompt for the LLM.

        Parameters
        ----------
        lane_error:
            Current signed lane-following error in pixels.
        obstacle_action:
            Latest avoidance recommendation from the detector.
        gps_error:
            Current GPS lateral offset in pixels.
        speed:
            Estimated truck speed in km/h, or ``None`` if unknown.
        """
        speed_str = f"{speed:.0f} km/h" if speed is not None else "unknown"
        drift = "left" if lane_error < 0 else "right" if lane_error > 0 else "centre"
        lines = [
            f"Lane drift: {drift} ({abs(lane_error):.0f}px).",
            f"Obstacle action suggested: {obstacle_action}.",
            f"GPS lateral offset: {gps_error:.0f}px.",
            f"Estimated speed: {speed_str}.",
            "What is the single best high-level action? "
            "Reply with exactly one of: "
            + ", ".join(sorted(KNOWN_ACTIONS)) + ".",
        ]
        return " ".join(lines)

    @staticmethod
    def _parse_action(text: str) -> str:
        """Extract the first recognised action token from LLM output."""
        upper = text.upper()
        for action in KNOWN_ACTIONS:
            if action in upper:
                return action
        return "CONTINUE"

    def should_query(self) -> bool:
        """Return True if enough time has passed since the last LLM call."""
        return (
            self.cfg.llm.enabled
            and time.monotonic() - self._last_call_time >= self.cfg.llm.call_interval
        )

    def query(
        self,
        lane_error: float,
        obstacle_action: str,
        gps_error: float,
        speed: Optional[float] = None,
    ) -> str:
        """Query the LLM for a high-level driving action.

        This call is *synchronous* but internally uses asyncio to drive the
        async :meth:`~agent.Agent.chat` method.  Keep call frequency low
        (≤ 2 Hz) to avoid blocking the control loop.

        Parameters
        ----------
        lane_error:
            Current signed lane-following error in pixels.
        obstacle_action:
            Latest avoidance recommendation from the obstacle detector.
        gps_error:
            GPS lateral offset in pixels.
        speed:
            Estimated truck speed in km/h (pass ``None`` if unavailable).

        Returns
        -------
        str
            One of the :data:`KNOWN_ACTIONS` tokens (defaults to
            ``"CONTINUE"`` on failure).
        """
        if not self.cfg.llm.enabled:
            return self._last_action

        self._ensure_agent()
        if self._agent is None:
            return self._last_action

        prompt = self._build_prompt(lane_error, obstacle_action, gps_error, speed)
        logger.debug("LLM prompt: %s", prompt)

        try:
            from models.chat import ChatMessage  # type: ignore
            messages = [ChatMessage(role="user", content=prompt)]
            # Run the async chat method synchronously using asyncio.run()
            response = asyncio.run(self._agent.chat(messages))

            action = self._parse_action(response.message)
            logger.info("LLM action: %s (raw: %.60s…)", action, response.message)
        except Exception as exc:
            logger.error("LLM query failed: %s", exc)
            action = self._last_action

        self._last_action = action
        self._last_call_time = time.monotonic()
        return action
