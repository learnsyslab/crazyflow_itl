"""Fly simulated drones from ROS in real time.

Each drone broadcasts its pose on TF, publishes its world frame twist on its own topic and follows
the state, attitude or body rate commands on its own command topics.
"""

import sys

import fire
import rclpy
from crazyflow.sim import Sim
from rclpy.utilities import remove_ros_args

from crazyflow_itl import RealTimePlugin, RenderPlugin, run
from crazyflow_itl.ros import CommandPlugin, SpinPlugin, TFPlugin, TwistPlugin


def main(n_drones: int = 1, duration: float | None = None, render: bool = False):
    sim = Sim(n_drones=n_drones, control="force_torque")
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("crazyflow_itl")
    try:
        # Outputs publish the state first, the pacing plugin waits, then inputs act
        plugins = [
            TFPlugin(node),
            TwistPlugin(node),
            RealTimePlugin(),
            SpinPlugin(node),
            CommandPlugin(node),
        ]
        if render:
            plugins.insert(0, RenderPlugin())
        run(sim, plugins, duration)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    fire.Fire(main, command=remove_ros_args(sys.argv)[1:])
