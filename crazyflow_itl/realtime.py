import logging
import math
import time

from crazyflow.sim import Sim

logger = logging.getLogger(__name__)


class RealTimePlugin:
    """Pace the sim to the wall clock."""

    def __init__(self, freq: int = 100, warn_lag: float = 0.05, warn_interval: float = 5.0):
        """Create a real-time plugin.

        Args:
            freq: Sync rate in Hz.
            warn_lag: Lag threshold for warnings in seconds.
            warn_interval: Minimum time between warnings in seconds.
        """
        self.freq = freq
        self.warn_lag = warn_lag
        self.warn_interval = warn_interval
        self.offset = None  # Set on the first step after a reset
        self.last_warning = -math.inf

    def setup(self, sim: Sim) -> Sim:
        """Start pacing and return the sim."""
        self.reset()
        return sim

    def reset(self):
        """Restart pacing from the next step."""
        self.offset = None
        self.last_warning = -math.inf

    def step(self, sim: Sim) -> Sim:
        """Sleep until the wall clock reaches the sim time and return the sim."""
        t = int(sim.data.core.steps[0, 0]) / sim.freq
        now = time.perf_counter()
        if self.offset is None:
            self.offset = now - t
        lag = now - self.offset - t
        if lag < 0:
            time.sleep(-lag)
        elif lag > self.warn_lag and now - self.last_warning > self.warn_interval:
            self.last_warning = now
            logger.warning(f"Simulation lags real time by {lag:.3f} s")
        return sim

    def close(self):
        """Nothing to release."""
