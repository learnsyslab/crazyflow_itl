"""Fly simulated Crazyflies with cflib2 clients over UDP.

cflib2 clients reach each drone at the URI udp://127.0.0.1:19850 + drone_idx, with the drone index
added to the port. The second drone, e.g., listens on udp://127.0.0.1:19851.
"""

import sys

import fire
import rclpy
from crazyflow.sim import Sim
from rclpy.utilities import remove_ros_args

from crazyflow_itl import RealTimePlugin, RenderPlugin, run
from crazyflow_itl.crazyflie import CrazyfliePlugin
from crazyflow_itl.ros import TFPlugin


def main(n_drones: int = 2, duration: float | None = None, render: bool = False):
    sim = Sim(n_drones=n_drones, control="force_torque")
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("crazyflow_itl")
    try:
        # Outputs publish the state first, the pacing plugin waits, then inputs act
        plugins = [TFPlugin(node), RealTimePlugin(), CrazyfliePlugin()]
        if render:
            plugins.insert(0, RenderPlugin())
        run(sim, plugins, duration)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    fire.Fire(main, command=remove_ros_args(sys.argv)[1:])
