import importlib.util
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TwistStamped
from simulation_interfaces.msg import Result, SimulationState
from simulation_interfaces.srv import SetSimulationState
from tf2_msgs.msg import TFMessage

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
FRAMES = {"cf21B_500_0", "cf21B_500_1"}


def wait_for_message(message_type, topic, accept, timeout=30.0):
    """Spin a listener node until a message on the topic satisfies `accept`."""
    rclpy.init()
    try:
        node = rclpy.create_node("crazyflow_itl_examples_listener")
        received = []
        node.create_subscription(
            message_type, topic, lambda message: accept(message) and received.append(message), 10
        )
        deadline = time.monotonic() + timeout
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
    finally:
        rclpy.shutdown()
    return bool(received)


def stop(process):
    """Interrupt the process and return the exit code and stderr.

    Pixi and launch pass the interrupt on to the sim. The group is killed if it does not exit
    within the grace period.
    """
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        _, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        _, stderr = process.communicate()
    return process.returncode, stderr


@contextmanager
def spawn(*command):
    """Start a command in its own process group and stop it when the block exits."""
    process = subprocess.Popen(
        command, cwd=ROOT, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        yield process
    finally:
        if process.returncode is None:
            stop(process)


def has_frames(message):
    return FRAMES <= {transform.child_frame_id for transform in message.transforms}


def assert_interrupted(returncode, stderr):
    """Assert that the interrupt alone ended a sim."""
    assert returncode == -signal.SIGINT, stderr
    assert stderr.count("Traceback") == 1, stderr
    assert stderr.rstrip().endswith("KeyboardInterrupt"), stderr


def check_published(command, message_type, topic, accept):
    """Check that a command publishes an accepted message and return its exit code and stderr."""
    with spawn(*command) as process:
        received = wait_for_message(message_type, topic, accept)
        returncode, stderr = stop(process)
    assert received, f"nothing received, exit code {returncode}, stderr:\n{stderr}"
    return returncode, stderr


def request_quit(service):
    """Set the simulation state of the stepping service to quitting and return the result code."""
    rclpy.init()
    try:
        node = rclpy.create_node("crazyflow_itl_examples_client")
        client = node.create_client(SetSimulationState, service)
        assert client.wait_for_service(timeout_sec=60.0)
        quitting = SimulationState(state=SimulationState.STATE_QUITTING)
        future = client.call_async(SetSimulationState.Request(state=quitting))
        rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
        node.destroy_node()
    finally:
        rclpy.shutdown()
    return future.result().result.result


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("ros_sim.py", {}),
        ("crazyflie_sim.py", {}),
        ("stepping_sim.py", {"paused": False}),
    ],
)
@pytest.mark.timeout(60)
def test_example_main(name, arguments):
    spec = importlib.util.spec_from_file_location("example", EXAMPLES / name)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.main(duration=0.5, **arguments)


@pytest.mark.timeout(90)
def test_pixi_task_publishes_tf():
    command = ["pixi", "run", "sim", "--n_drones", "2"]
    assert_interrupted(*check_published(command, TFMessage, "/tf", has_frames))


@pytest.mark.timeout(90)
def test_launch_file_publishes_tf():
    command = ["ros2", "launch", str(EXAMPLES / "launch_sim.py"), "n_drones:=2"]
    # Launch reports the output of the sim on stdout and exits cleanly
    returncode, stderr = check_published(command, TFMessage, "/tf", has_frames)
    assert returncode == 0, stderr
    assert "Traceback" not in stderr, stderr


@pytest.mark.timeout(90)
def test_ros_args_remap_twist():
    command = ["pixi", "run", "sim", "--ros-args", "-r", "/crazyflow/cf21B_500_0/twist:=/remapped"]
    assert_interrupted(*check_published(command, TwistStamped, "/remapped", lambda message: True))


@pytest.mark.timeout(120)
def test_stepping_quits_on_request():
    with spawn("pixi", "run", "python", "examples/stepping_sim.py") as process:
        assert request_quit("/crazyflow_itl/set_simulation_state") == Result.RESULT_OK
        _, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    assert "Traceback" not in stderr, stderr


def fly(sim_command, client_command, ready):
    """Fly the drones of a sim with a client, check that both exit cleanly and return the poses.

    The client starts once the sim has published `ready` TF messages.
    """
    rclpy.init()
    try:
        node = rclpy.create_node("crazyflow_itl_examples_listener")
        messages = []
        node.create_subscription(TFMessage, "/tf", messages.append, 100)
        with spawn(*sim_command) as sim:
            while len(messages) < ready:
                rclpy.spin_once(node, timeout_sec=0.1)
            with spawn(*client_command) as client:
                while client.poll() is None:
                    rclpy.spin_once(node, timeout_sec=0.1)
            # Receive the poses that the sim published while the client exited
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            sim_code, sim_stderr = stop(sim)
        node.destroy_node()
    finally:
        rclpy.shutdown()
    _, client_stderr = client.communicate()
    assert client.returncode == 0, client_stderr
    assert_interrupted(sim_code, sim_stderr)
    return messages


# The real-time sims publish their second poses once the first chunk has compiled. A paused sim
# publishes nothing until a client steps it.
@pytest.mark.parametrize(
    ("sim_command", "client_command", "ready"),
    [
        (
            "pixi run sim --n_drones 2",
            "pixi run python examples/ros_client.py",
            2,
        ),
        (
            "pixi run python examples/crazyflie_sim.py --n_drones 2",
            "pixi run -e tests python examples/crazyflie_client.py --n_drones 2",
            2,
        ),
        (
            "pixi run python examples/stepping_sim.py --n_drones 2",
            "pixi run python examples/stepping_client.py",
            0,
        ),
    ],
    ids=["ros", "crazyflie", "stepping"],
)
@pytest.mark.timeout(120)
def test_client_flies_square(sim_command, client_command, ready):
    messages = fly(sim_command.split(), client_command.split(), ready)
    for frame in FRAMES:
        translations = [
            transform.transform.translation
            for message in messages
            for transform in message.transforms
            if transform.child_frame_id == frame
        ]
        path = np.array([[point.x, point.y, point.z] for point in translations])
        travel = path.max(axis=0) - path.min(axis=0)
        np.testing.assert_allclose(travel, [0.5, 0.5, 0.5], atol=0.2, err_msg=frame)
        assert path[-1, 2] < 0.2, f"{frame} did not land"
