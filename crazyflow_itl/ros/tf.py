from crazyflow.sim import Sim
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from crazyflow_itl.ros.utils import drone_names, stamp


class TFPlugin:
    """Broadcast the pose of every drone on /tf."""

    def __init__(
        self,
        node: Node,
        names: list[str] | None = None,
        parent: str = "world",
        freq: int = 100,
    ):
        """Create a TF plugin.

        Args:
            node: ROS node.
            names: Drone frame names. Defaults to the drone model and index.
            parent: Parent frame.
            freq: Publish rate in Hz.
        """
        self.node = node
        self.names = names
        self.parent = parent
        self.freq = freq
        self.publisher = None  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Create the publisher and return the sim."""
        self.names = drone_names(sim) if self.names is None else self.names
        assert len(self.names) == sim.n_drones
        self.publisher = self.node.create_publisher(TFMessage, "/tf", 100)
        return sim

    def step(self, sim: Sim) -> Sim:
        """Broadcast one transform per drone and return the sim."""
        header_stamp = stamp(self.node, sim)
        states = sim.data.states
        message = TFMessage()
        for name, pos, quat in zip(self.names, states.pos[0].tolist(), states.quat[0].tolist()):
            transform = TransformStamped()
            transform.header.stamp = header_stamp
            transform.header.frame_id = self.parent
            transform.child_frame_id = name
            translation, rotation = transform.transform.translation, transform.transform.rotation
            translation.x, translation.y, translation.z = pos
            rotation.x, rotation.y, rotation.z, rotation.w = quat
            message.transforms.append(transform)
        self.publisher.publish(message)
        return sim

    def close(self):
        """Destroy the publisher."""
        self.node.destroy_publisher(self.publisher)
