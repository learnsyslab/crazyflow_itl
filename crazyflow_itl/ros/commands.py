from functools import partial

import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from crazyflow_itl.control import CONTROL_KEY, MODES, command_drones, insert_control_stage
from crazyflow_itl.ros.utils import drone_names
from crazyflow_itl.stop import insert_stop_stage, stop_drones


class CommandPlugin:
    """Forward ROS command messages to the controllers of the sim."""

    def __init__(
        self,
        node: Node,
        names: list[str] | None = None,
        topic: str = "/crazyflow/{name}/command/{mode}",
        freq: int = 100,
    ):
        """Create a command plugin.

        Args:
            node: ROS node.
            names: Drone names. Defaults to the drone model and index.
            topic: Topic template with `{name}` and `{mode}` fields.
            freq: Command rate in Hz.
        """
        self.node = node
        self.names = names
        self.topic = topic
        self.freq = freq
        self.modes = []  # Set in setup
        self.commands = None  # Set in setup
        self.stopped = None  # Set in setup
        self.subscriptions = []  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Create one subscription per drone and mode, switch all motors off and return the sim.

        The modes are state, attitude and body rate. A command switches its drone to the mode of
        its topic. State commands have 16 entries and the others 4. The motors of a drone stay
        off until its first command. The sim must use force and torque control with first
        principles dynamics.
        """
        self.names = drone_names(sim) if self.names is None else self.names
        assert len(self.names) == sim.n_drones
        sim = insert_control_stage(insert_stop_stage(sim))
        self.modes = [Control.state] * sim.n_drones
        self.commands = np.array(sim.data.plugins[CONTROL_KEY].command[0])
        self.stopped = np.ones(sim.n_drones, bool)
        self.subscriptions = [
            self.node.create_subscription(
                Float64MultiArray,
                self.topic.format(name=name, mode=mode),
                partial(self._receive, index, mode),
                10,
            )
            for index, name in enumerate(self.names)
            for mode in MODES
        ]
        return sim

    def step(self, sim: Sim) -> Sim:
        """Apply the latest commands to the sim and return the sim."""
        stop_drones(sim, self.stopped)
        command_drones(sim, self.modes, self.commands)
        return sim

    def close(self):
        """Destroy the subscriptions."""
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)

    def _receive(self, index: int, mode: Control, message: Float64MultiArray):
        """Store the mode and command of one drone."""
        length = 16 if mode == Control.state else 4
        assert len(message.data) == length, f"Command of length {len(message.data)}, not {length}"
        self.modes[index] = mode
        self.commands[index, :length] = message.data
        self.stopped[index] = False
