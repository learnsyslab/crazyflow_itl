import numpy as np
from crazyflow.sim import Sim
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from crazyflow_itl.ros.utils import drone_names, stamp


class TwistPlugin:
    """Publish the world frame twist of every drone."""

    def __init__(
        self,
        node: Node,
        names: list[str] | None = None,
        parent: str = "world",
        topic: str = "/crazyflow/{name}/twist",
        freq: int = 100,
    ):
        """Create a twist plugin.

        Args:
            node: ROS node.
            names: Drone names. Defaults to the drone model and index.
            parent: Frame of the twists.
            topic: Topic template with a `{name}` field.
            freq: Publish rate in Hz.
        """
        self.node = node
        self.names = names
        self.parent = parent
        self.topic = topic
        self.freq = freq
        self.publishers = []  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Create one publisher per drone and return the sim."""
        self.names = drone_names(sim) if self.names is None else self.names
        assert len(self.names) == sim.n_drones
        self.publishers = [
            self.node.create_publisher(TwistStamped, self.topic.format(name=name), 10)
            for name in self.names
        ]
        return sim

    def step(self, sim: Sim) -> Sim:
        """Publish one twist per drone and return the sim."""
        header_stamp = stamp(self.node, sim)
        states = sim.data.states
        world_ang_vel = Rotation.from_quat(np.asarray(states.quat[0])).apply(
            np.asarray(states.ang_vel[0])
        )
        for publisher, vel, ang_vel in zip(
            self.publishers, states.vel[0].tolist(), world_ang_vel.tolist()
        ):
            message = TwistStamped()
            message.header.stamp = header_stamp
            message.header.frame_id = self.parent
            message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = vel
            message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = ang_vel
            publisher.publish(message)
        return sim

    def close(self):
        """Destroy the publishers."""
        for publisher in self.publishers:
            self.node.destroy_publisher(publisher)
