"""Start the ROS sim example from ros2 launch with a configurable number of drones."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    """Declare the number of drones and start the ROS sim example with it."""
    script = Path(__file__).parent / "ros_sim.py"
    return LaunchDescription(
        [
            DeclareLaunchArgument("n_drones", default_value="1"),
            ExecuteProcess(
                cmd=["python", str(script), "--n_drones", LaunchConfiguration("n_drones")],
                output="screen",
            ),
        ]
    )
