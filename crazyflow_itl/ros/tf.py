import numpy as np
from crazyflow.sim import Sim
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from crazyflow_itl.ros.utils import drone_names, stamp


class TFPlugin:
    """Broadcast the pose of every drone and every other mocap body on /tf."""

    def __init__(
        self,
        node: Node,
        drone_names: list[str] | None = None,
        mocap_names: list[str] | None = None,
        parent: str = "world",
        freq: int = 100,
    ):
        """Create a TF plugin.

        Args:
            node: ROS node.
            drone_names: Drone frame names. Defaults to the drone model and index.
            mocap_names: Mocap body frame names, in mocap order. Defaults to the body names.
            parent: Parent frame.
            freq: Publish rate in Hz.
        """
        self.node = node
        self.drone_names = drone_names
        self.mocap_names = mocap_names
        self.parent = parent
        self.freq = freq
        self.frame_names = []  # Set in setup
        self.mocap_ids = []  # Set in setup
        self.publisher = None  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Create the publisher and return the sim."""
        self.drone_names = drone_names(sim) if self.drone_names is None else self.drone_names
        assert len(self.drone_names) == sim.n_drones
        model = sim.mj_model
        drone_ids = np.asarray(sim.data.core.drone_mocap_ids).tolist()
        bodies = [model.body(i) for i in np.flatnonzero(model.body_mocapid >= 0)]
        bodies = [body for body in bodies if body.mocapid.item() not in drone_ids]
        if self.mocap_names is None:
            self.mocap_names = [body.name for body in bodies]
        assert len(self.mocap_names) == len(bodies)
        self.frame_names = self.drone_names + self.mocap_names  # Order of the poses in step
        self.mocap_ids = [body.mocapid.item() for body in bodies]
        self.publisher = self.node.create_publisher(TFMessage, "/tf", 100)
        return sim

    def step(self, sim: Sim) -> Sim:
        """Broadcast one transform per drone and body and return the sim."""
        header_stamp = stamp(self.node, sim)
        states = sim.data.states
        mocap_pos = np.asarray(sim.mjx_data.mocap_pos[0])[self.mocap_ids]
        mocap_quat = np.asarray(sim.mjx_data.mocap_quat[0])[self.mocap_ids]  # MuJoCo order
        pos = np.concatenate([states.pos[0], mocap_pos])
        quat = np.concatenate([states.quat[0], np.roll(mocap_quat, -1, axis=-1)])
        message = TFMessage()
        for name, p, q in zip(self.frame_names, pos.tolist(), quat.tolist()):
            transform = TransformStamped()
            transform.header.stamp = header_stamp
            transform.header.frame_id = self.parent
            transform.child_frame_id = name
            translation, rotation = transform.transform.translation, transform.transform.rotation
            translation.x, translation.y, translation.z = p
            rotation.x, rotation.y, rotation.z, rotation.w = q
            message.transforms.append(transform)
        self.publisher.publish(message)
        return sim

    def close(self):
        """Destroy the publisher."""
        self.node.destroy_publisher(self.publisher)
