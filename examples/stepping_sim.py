"""Fly simulated drones from ROS clients that pause, play and step the sim.

The node serves the simulation_interfaces services get_simulator_features, get_simulation_state,
set_simulation_state and step_simulation and the action simulate_steps in its private namespace,
for example /crazyflow_itl/step_simulation.
"""

import sys

import fire
import rclpy
from crazyflow.sim import Sim
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args

from crazyflow_itl import run
from crazyflow_itl.ros import ClockPlugin, CommandPlugin, SteppingPlugin, TFPlugin


def main(n_drones: int = 1, duration: float | None = None, paused: bool = True):
    sim = Sim(n_drones=n_drones, control="force_torque")
    rclpy.init(args=sys.argv)
    node = rclpy.create_node(
        "crazyflow_itl", parameter_overrides=[Parameter("use_sim_time", value=True)]
    )
    try:
        # SteppingPlugin handles all rclpy callbacks
        plugins = [
            ClockPlugin(node),
            TFPlugin(node),
            SteppingPlugin(node, paused=paused),
            CommandPlugin(node),
        ]
        run(sim, plugins, duration)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    fire.Fire(main, command=remove_ros_args(sys.argv)[1:])
