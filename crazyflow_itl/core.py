import math
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Protocol

from crazyflow.sim import Sim


class Plugin(Protocol):
    """Define the plugin protocol."""

    freq: int

    def setup(self, sim: Sim) -> Sim:
        """Prepare the plugin and configure the simulation.

        Args:
            sim: The run-loop simulation.

        Returns:
            The simulation.
        """

    def step(self, sim: Sim) -> Sim:
        """Execute the plugin at its frequency.

        Args:
            sim: The run-loop simulation.

        Returns:
            The simulation.
        """

    def close(self):
        """Release the plugin resources."""


def run(sim: Sim, plugins: Sequence[Plugin], duration: float | None = None) -> Sim:
    """Run the sim with its plugins.

    Args:
        sim: Single world simulation.
        plugins: Plugins in execution order, with rates that divide the sim frequency.
        duration: Sim duration in seconds, or None to run forever.

    Returns:
        The final simulation.
    """
    assert sim.n_worlds == 1
    assert plugins
    assert all(sim.freq % plugin.freq == 0 for plugin in plugins)
    periods = [sim.freq // plugin.freq for plugin in plugins]
    chunk = math.gcd(*periods)
    steps = int(sim.data.core.steps[0, 0])
    with ExitStack() as stack:
        for plugin in plugins:
            sim = plugin.setup(sim)
            stack.callback(plugin.close)
        # Compile the step function before the loop to prevent lagging
        data = sim.data
        sim.step(chunk)
        sim.data = data
        while duration is None or steps / sim.freq < duration:
            for plugin, period in zip(plugins, periods):
                if steps % period == 0:
                    sim = plugin.step(sim)
            sim.step(chunk)
            steps += chunk
    return sim
