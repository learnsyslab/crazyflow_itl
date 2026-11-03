from crazyflow.sim import Sim
from rclpy.executors import SingleThreadedExecutor, TimeoutException
from rclpy.node import Node


class SpinPlugin:
    """Run the ready callbacks of a node inside the thread of the run loop."""

    def __init__(self, node: Node, freq: int = 100):
        """Create a spin plugin.

        Args:
            node: Node whose callbacks are served.
            freq: Spin rate in Hz.
        """
        self.node = node
        self.freq = freq
        self.executor = SingleThreadedExecutor()

    def setup(self, sim: Sim) -> Sim:
        """Add the node to the executor and return the sim."""
        self.executor.add_node(self.node)
        return sim

    def step(self, sim: Sim) -> Sim:
        """Run all ready callbacks without blocking and return the sim."""
        while True:
            try:
                handler, _, _ = self.executor.wait_for_ready_callbacks(timeout_sec=0)
            except TimeoutException:
                return sim
            handler()
            handler.result()

    def close(self):
        """Remove the node from the executor and shut the executor down."""
        self.executor.remove_node(self.node)
        self.executor.shutdown()
