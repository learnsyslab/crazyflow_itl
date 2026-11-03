"""Fly the drones of the stepping sim through a takeoff, a square and a landing.

Since the sim only advances on step requests, clients can freely control the simulation pace.
"""

import sys
import time
from functools import partial
from threading import Thread

import fire
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.utilities import remove_ros_args
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import StepSimulation
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

# Takeoff to 0.5 m, a square with 0.5 m sides and the landing, relative to the start position
WAYPOINTS = np.array(
    [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0, 0, 0.5], [0, 0, 0]]
)


def receive(positions: dict[str, np.ndarray], message: TFMessage):
    """Store the position of every drone in a TF message."""
    for transform in message.transforms:
        translation = transform.transform.translation
        positions[transform.child_frame_id] = np.array(
            [translation.x, translation.y, translation.z]
        )


def main(leg_steps: int = 200):
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("stepping_client")
    positions = {}
    node.create_subscription(TFMessage, "/tf", partial(receive, positions), 10)
    stepper = node.create_client(StepSimulation, "/crazyflow_itl/step_simulation")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = Thread(target=executor.spin)
    thread.start()
    try:
        stepper.wait_for_service()
        # The paused sim publishes the poses only when it advances
        while not positions:
            stepper.call(StepSimulation.Request(steps=1))
            time.sleep(0.1)
        starts = dict(positions)
        publishers = {
            name: node.create_publisher(Float64MultiArray, f"/crazyflow/{name}/command/state", 10)
            for name in starts
        }
        # Each waypoint is a single command, so the sim must have discovered the publishers
        while any(publisher.get_subscription_count() == 0 for publisher in publishers.values()):
            time.sleep(0.01)
        for waypoint in WAYPOINTS[1:]:
            for name, publisher in publishers.items():
                position = starts[name] + waypoint
                # Position, velocity, acceleration, quaternion and body rates
                command = np.concatenate([position, np.zeros(6), [0, 0, 0, 1], np.zeros(3)])
                publisher.publish(Float64MultiArray(data=command.tolist()))
            result = stepper.call(StepSimulation.Request(steps=leg_steps)).result
            assert result.result == Result.RESULT_OK, result.error_message
    finally:
        executor.shutdown()
        thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    fire.Fire(main, command=remove_ros_args(sys.argv)[1:])
