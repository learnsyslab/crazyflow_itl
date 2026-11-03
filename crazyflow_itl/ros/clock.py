from crazyflow.sim import Sim
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

from crazyflow_itl.ros.utils import sim_time


class ClockPlugin:
    """Publish the sim time on /clock."""

    def __init__(self, node: Node, freq: int = 100):
        """Create a clock plugin.

        Args:
            node: ROS node.
            freq: Publish rate in Hz.
        """
        self.node = node
        self.freq = freq
        self.publisher = None  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Create the publisher and return the sim."""
        self.publisher = self.node.create_publisher(Clock, "/clock", 10)
        return sim

    def step(self, sim: Sim) -> Sim:
        """Publish the sim time and return the sim."""
        self.publisher.publish(Clock(clock=sim_time(sim)))
        return sim

    def close(self):
        """Destroy the publisher."""
        self.node.destroy_publisher(self.publisher)
