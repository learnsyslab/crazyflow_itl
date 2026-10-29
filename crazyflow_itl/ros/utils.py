from builtin_interfaces.msg import Time as TimeMessage
from crazyflow.sim import Sim
from rclpy.node import Node
from rclpy.time import Time


def drone_names(sim: Sim) -> list[str]:
    """Return the default frame and topic names of the drones in the sim.

    Args:
        sim: The simulation.

    Returns:
        One name per drone, made of the drone model and index.
    """
    return [f"{sim.drone}_{i}" for i in range(sim.n_drones)]


def sim_time(sim: Sim) -> TimeMessage:
    """Return the sim time as a ROS time message.

    Args:
        sim: The simulation.

    Returns:
        The sim time.
    """
    return Time(nanoseconds=round(int(sim.data.core.steps[0, 0]) / sim.freq * 1e9)).to_msg()


def stamp(node: Node, sim: Sim) -> TimeMessage:
    """Return the message time for the current step.

    Args:
        node: ROS node.
        sim: The simulation.

    Returns:
        The sim time if the node uses it, otherwise the node clock time.
    """
    if node.get_parameter("use_sim_time").value:
        return sim_time(sim)
    return node.get_clock().now().to_msg()
