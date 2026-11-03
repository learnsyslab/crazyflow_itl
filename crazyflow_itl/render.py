from crazyflow.sim import Sim


class RenderPlugin:
    """Show the sim in the MuJoCo viewer."""

    def __init__(self, freq: int = 50):
        """Create a render plugin.

        Args:
            freq: Render rate in Hz.
        """
        self.freq = freq
        self.sim = None  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Store the sim for clean exits."""
        self.sim = sim
        return sim

    def step(self, sim: Sim) -> Sim:
        """Draw the current state in the viewer and return the sim."""
        sim.render()
        return sim

    def close(self):
        """Close the viewer."""
        self.sim.close()
