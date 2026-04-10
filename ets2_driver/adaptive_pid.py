"""Speed-dependent PID gain scheduler for the ETS2 AI driver.

Gain scheduling improves steering stability across speed regimes:

* **Low speed (0 → speed_mid km/h)**: Higher P-gain for precise, tight
  lane-keeping; lower D-gain to avoid oscillation at near-zero speed.
* **Mid speed (speed_mid km/h)**: Balanced gains matching the original
  static defaults — a stable cruise-control profile.
* **High speed (speed_high+ km/h)**: Lower P-gain to prevent oversteer
  on the motorway; higher D-gain to damp steering oscillation at speed.

Between anchor points gains are **linearly interpolated**, so transitions
are smooth rather than step functions.  The ``gain_scale`` output (0.0 =
low-speed profile, 1.0 = high-speed profile) is useful for dashboard display.

Usage::

    scheduler = AdaptivePIDGains(cfg)
    kp, ki, kd, integral_max, gain_scale = scheduler.get_gains(speed_kph)
"""

from __future__ import annotations

from .config import ETS2Config


class AdaptivePIDGains:
    """Interpolated PID gain scheduler driven by vehicle speed.

    Parameters
    ----------
    cfg:
        Top-level :class:`~ets2_driver.config.ETS2Config` instance.
    """

    def __init__(self, cfg: ETS2Config) -> None:
        self.cfg = cfg

    def get_gains(
        self, speed_kph: float
    ) -> tuple[float, float, float, float, float]:
        """Return ``(kp, ki, kd, integral_max, gain_scale)`` for *speed_kph*.

        ``gain_scale`` is a normalised value in ``[0.0, 1.0]`` that represents
        where along the low→high gain spectrum the current speed sits (0.0 =
        low-speed profile, 1.0 = high-speed profile).  It is exposed for
        dashboard/debug display so the operator can see which gain curve is
        active.

        If :attr:`~ets2_driver.config.AdaptiveGainConfig.enabled` is ``False``
        the base :class:`~ets2_driver.config.PidConfig` values are returned
        unchanged and ``gain_scale`` is ``0.5`` (neutral).

        Parameters
        ----------
        speed_kph:
            Current vehicle speed in km/h (from
            :class:`~ets2_driver.speed_tracker.SpeedTracker`).

        Returns
        -------
        tuple[float, float, float, float, float]
            ``(kp, ki, kd, integral_max, gain_scale)``
        """
        agcfg = self.cfg.adaptive_gain
        pcfg  = self.cfg.pid

        if not agcfg.enabled:
            return pcfg.kp, pcfg.ki, pcfg.kd, pcfg.integral_max, 0.5

        speed = max(0.0, speed_kph)

        # Three anchor speeds: 0 km/h, speed_mid, speed_high.
        # Enforce a minimum gap of 1 km/h between anchors to avoid division
        # by zero in the interpolation.
        s0 = 0.0
        s1 = max(s0 + 1.0, agcfg.speed_mid)
        s2 = max(s1 + 1.0, agcfg.speed_high)

        # Compute the normalised position t in [0, 1].
        # t=0 → low-speed anchor, t=0.5 → mid-speed anchor, t=1 → high-speed.
        if speed <= s0:
            t = 0.0
        elif speed >= s2:
            t = 1.0
        elif speed <= s1:
            # Lower half: between anchor 0 and anchor 1
            t = (speed - s0) / (s1 - s0) * 0.5   # maps to [0.0, 0.5]
        else:
            # Upper half: between anchor 1 and anchor 2
            t = 0.5 + (speed - s1) / (s2 - s1) * 0.5  # maps to [0.5, 1.0]

        kp      = self._lerp3(agcfg.kp_low,        agcfg.kp_mid,        agcfg.kp_high,        t)
        ki      = self._lerp3(agcfg.ki_low,        agcfg.ki_mid,        agcfg.ki_high,        t)
        kd      = self._lerp3(agcfg.kd_low,        agcfg.kd_mid,        agcfg.kd_high,        t)
        int_max = self._lerp3(agcfg.integral_max_low, agcfg.integral_max_mid, agcfg.integral_max_high, t)

        return kp, ki, kd, int_max, t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lerp3(v_low: float, v_mid: float, v_high: float, t: float) -> float:
        """Interpolate between three anchor values given normalised *t* in [0, 1].

        The interval ``[0, 0.5]`` maps linearly from *v_low* to *v_mid*, and
        ``[0.5, 1.0]`` maps linearly from *v_mid* to *v_high*.
        """
        t = max(0.0, min(1.0, t))
        if t <= 0.5:
            s = t * 2.0  # normalise [0, 0.5] → [0, 1]
            return v_low + s * (v_mid - v_low)
        else:
            s = (t - 0.5) * 2.0  # normalise [0.5, 1.0] → [0, 1]
            return v_mid + s * (v_high - v_mid)
