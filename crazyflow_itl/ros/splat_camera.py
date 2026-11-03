from pathlib import Path

import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import build_render_splat_fn
from crazyflow.sim.splat import attach_splats
from rclpy.node import Node
from sensor_msgs.msg import Image

from crazyflow_itl.ros.utils import drone_names, stamp


class SplatCameraPlugin:
    """Publish a splat rendered onboard camera image of every drone."""

    def __init__(
        self,
        node: Node,
        scene: Path,
        names: list[str] | None = None,
        resolution: tuple[int, int] = (640, 480),
        topic: str = "/crazyflow/{name}/image",
        freq: int = 50,
    ):
        """Create a splat camera plugin.

        Args:
            node: ROS node.
            scene: Splat file of the static scene in the world frame.
            names: Drone names. Defaults to the drone model and index.
            resolution: Image (width, height) in pixels.
            topic: Topic template with a `{name}` field.
            freq: Publish rate in Hz.
        """
        self.node = node
        self.names = names
        self.scene = scene
        self.resolution = resolution
        self.topic = topic
        self.freq = freq
        self.render = None  # Set in setup
        self.publishers = []  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Attach the scene, create one publisher per drone and return the sim."""
        self.names = drone_names(sim) if self.names is None else self.names
        assert len(self.names) == sim.n_drones
        assert sim.device.platform == "gpu", "Splat rendering requires a sim on the GPU"
        attach_splats(sim, self.scene)
        self.render = build_render_splat_fn(sim, resolution=self.resolution)
        self.publishers = [
            self.node.create_publisher(Image, self.topic.format(name=name), 10)
            for name in self.names
        ]
        return sim

    def step(self, sim: Sim) -> Sim:
        """Publish one image per drone and return the sim."""
        header_stamp = stamp(self.node, sim)
        images = (np.asarray(self.render(sim.data))[0] * 255).astype(np.uint8)
        for publisher, name, image in zip(self.publishers, self.names, images):
            message = Image()
            message.header.stamp = header_stamp
            message.header.frame_id = name
            message.height, message.width = image.shape[:2]
            message.encoding = "rgb8"
            message.step = image.shape[1] * 3
            message.data = image.tobytes()
            publisher.publish(message)
        return sim

    def close(self):
        """Destroy the publishers."""
        for publisher in self.publishers:
            self.node.destroy_publisher(publisher)
