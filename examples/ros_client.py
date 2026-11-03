"""Fly the drones of the ROS sim through a takeoff, a square and a landing.

The client listens to TF messages and computes its control commands as full state setpoints. It
also prints the drone positions at every waypoint to demonstrate the ROS integration.
"""

import sys
import time
from functools import partial
from itertools import pairwise

import fire
import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
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


def main(leg_duration: float = 2.0):
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("ros_client")
    positions = {}
    node.create_subscription(TFMessage, "/tf", partial(receive, positions), 10)
    try:
        while not positions:
            rclpy.spin_once(node)
        starts = dict(positions)
        publishers = {
            name: node.create_publisher(Float64MultiArray, f"/crazyflow/{name}/command/state", 10)
            for name in starts
        }
        for previous, waypoint in pairwise(WAYPOINTS):
            velocity = (waypoint - previous) / leg_duration
            leg_start = time.monotonic()
            while (elapsed := time.monotonic() - leg_start) < leg_duration:
                rclpy.spin_once(node)  # Waiting for the next poses paces the commands at TF rate
                for name, publisher in publishers.items():
                    position = starts[name] + previous + elapsed * velocity
                    # Position, velocity, acceleration, quaternion and body rates
                    command = np.concatenate(
                        [position, velocity, np.zeros(3), [0, 0, 0, 1], np.zeros(3)]
                    )
                    publisher.publish(Float64MultiArray(data=command.tolist()))
            for name, position in positions.items():
                print(f"{name} at {position.round(2)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    fire.Fire(main, command=remove_ros_args(sys.argv)[1:])
