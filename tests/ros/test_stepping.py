import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from crazyflow.sim import Sim
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rosgraph_msgs.msg import Clock
from simulation_interfaces.action import SimulateSteps
from simulation_interfaces.msg import Result, SimulationState, SimulatorFeatures
from simulation_interfaces.srv import (
    GetSimulationState,
    GetSimulatorFeatures,
    SetSimulationState,
    StepSimulation,
)
from std_msgs.msg import String

from crazyflow_itl import run
from crazyflow_itl.ros import ClockPlugin, SteppingPlugin

SERVICES = {
    GetSimulatorFeatures: "get_simulator_features",
    GetSimulationState: "get_simulation_state",
    SetSimulationState: "set_simulation_state",
    StepSimulation: "step_simulation",
}
PLAYING = SimulationState(state=SimulationState.STATE_PLAYING)
PAUSED = SimulationState(state=SimulationState.STATE_PAUSED)

TOPIC = "/crazyflow_itl_stepping_test"

Stepped = namedtuple(
    "Stepped", ["sim", "node", "executor", "clients", "running", "received", "clocks"]
)


class ClockDiscovery:
    """Wait in setup until the publisher of a clock plugin has a subscriber."""

    freq = 100
    timeout = 5.0

    def __init__(self, clock):
        self.clock = clock

    def setup(self, sim):
        deadline = time.monotonic() + self.timeout
        while self.clock.publisher.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "/clock subscription was not discovered"
            time.sleep(0.01)
        return sim

    def step(self, sim):
        return sim

    def close(self):
        pass


@pytest.fixture
def stepped(node):
    """Run a paused sim with a stepping plugin in a background thread with clients on `node`.

    The sim publishes /clock before the stepping plugin, as scripts place outputs first.
    """
    stepping_node = rclpy.create_node("crazyflow_itl_stepping")
    received, clocks = [], []
    stepping_node.create_subscription(String, TOPIC, received.append, 10)
    node.create_subscription(Clock, "/clock", clocks.append, 10)
    plugin = SteppingPlugin(stepping_node)
    sim = Sim()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    clients = {
        service: node.create_client(service, f"/crazyflow_itl_stepping/{name}")
        for service, name in SERVICES.items()
    }
    with ThreadPoolExecutor() as pool:
        clock = ClockPlugin(stepping_node)
        plugins = [clock, ClockDiscovery(clock), plugin]
        running = pool.submit(run, sim, plugins)
        assert all(client.wait_for_service(timeout_sec=5.0) for client in clients.values())
        yield Stepped(sim, node, executor, clients, running, received, clocks)
        plugin.state = SimulationState.STATE_QUITTING
        running.result(timeout=30.0)
    stepping_node.destroy_node()


def spin(stepped, future):
    """Spin the client executor until the future is done and return its result."""
    stepped.executor.spin_until_future_complete(future, timeout_sec=30.0)
    assert future.done()
    return future.result()


def call(stepped, service, **fields):
    """Call a stepping service with the given request fields and return the response."""
    return spin(stepped, stepped.clients[service].call_async(service.Request(**fields)))


def latest_clock(stepped):
    """Spin the client executor until no more /clock messages arrive and return the last time."""
    previous = None
    while previous != len(stepped.clocks):
        previous = len(stepped.clocks)
        stepped.executor.spin_once(timeout_sec=0.1)
    clock = stepped.clocks[-1].clock
    return clock.sec + clock.nanosec * 1e-9


def test_paused_sim_does_not_advance(stepped):
    time.sleep(0.3)
    assert int(stepped.sim.data.core.steps[0, 0]) == 0


def test_paused_plugin_serves_its_node(stepped):
    publisher = stepped.node.create_publisher(String, TOPIC, 10)
    deadline = time.monotonic() + 5.0
    while publisher.get_subscription_count() == 0:
        assert time.monotonic() < deadline, "subscription was not discovered"
        time.sleep(0.01)
    publisher.publish(String(data="served"))
    while not stepped.received:
        assert time.monotonic() < deadline, "message was not served"
        time.sleep(0.01)
    assert stepped.received[0].data == "served"
    assert int(stepped.sim.data.core.steps[0, 0]) == 0


def test_step_simulation_advances_exact_periods(stepped):
    assert call(stepped, StepSimulation, steps=3).result.result == Result.RESULT_OK
    assert int(stepped.sim.data.core.steps[0, 0]) == 15


def test_clock_follows_granted_steps(stepped):
    deadline = time.monotonic() + 5.0
    while not stepped.clocks:
        assert time.monotonic() < deadline, "no /clock message before the first grant"
        stepped.executor.spin_once(timeout_sec=0.1)
    assert latest_clock(stepped) == 0.0
    for granted in range(1, 4):
        assert call(stepped, StepSimulation).result.result == Result.RESULT_OK
        assert int(stepped.sim.data.core.steps[0, 0]) == 5 * granted
        assert latest_clock(stepped) == pytest.approx(float(stepped.sim.time[0, 0]), abs=1e-9)


def test_step_zero_is_noop(stepped):
    start = time.perf_counter()
    assert call(stepped, StepSimulation, steps=0).result.result == Result.RESULT_OK
    assert time.perf_counter() - start < 1.0
    assert int(stepped.sim.data.core.steps[0, 0]) == 0


def test_step_service_rejects_while_playing(stepped):
    assert call(stepped, SetSimulationState, state=PLAYING).result.result == Result.RESULT_OK
    assert call(stepped, StepSimulation).result.result == Result.RESULT_OPERATION_FAILED


def test_play_paces_real_time(stepped):
    call(stepped, StepSimulation)  # Compile the chunk before timing
    start = float(stepped.sim.time[0, 0])
    call(stepped, SetSimulationState, state=PLAYING)
    time.sleep(0.5)
    call(stepped, SetSimulationState, state=PAUSED)
    assert 0.4 <= float(stepped.sim.time[0, 0]) - start <= 0.6


def test_simulate_steps_action_feedback(stepped):
    feedback = []
    client = ActionClient(stepped.node, SimulateSteps, "/crazyflow_itl_stepping/simulate_steps")
    assert client.wait_for_server(timeout_sec=5.0)
    goal = SimulateSteps.Goal(steps=4)
    goal_handle = spin(stepped, client.send_goal_async(goal, feedback_callback=feedback.append))
    assert goal_handle.accepted
    # The client drops feedback that arrives after the result, so the result is requested last
    deadline = time.monotonic() + 30.0
    while len(feedback) < 4 and time.monotonic() < deadline:
        stepped.executor.spin_once(timeout_sec=0.1)
    response = spin(stepped, goal_handle.get_result_async())
    assert response.status == GoalStatus.STATUS_SUCCEEDED
    assert response.result.result.result == Result.RESULT_OK
    assert [message.feedback.completed_steps for message in feedback] == [1, 2, 3, 4]
    assert [message.feedback.remaining_steps for message in feedback] == [3, 2, 1, 0]
    assert int(stepped.sim.data.core.steps[0, 0]) == 20


def test_quitting_ends_run(stepped):
    quitting = SimulationState(state=SimulationState.STATE_QUITTING)
    assert call(stepped, SetSimulationState, state=quitting).result.result == Result.RESULT_OK
    stepped.running.result(timeout=5.0)


def test_features_reported(stepped):
    assert sorted(call(stepped, GetSimulatorFeatures).features.features) == [
        SimulatorFeatures.SIMULATION_STATE_GETTING,
        SimulatorFeatures.SIMULATION_STATE_SETTING,
        SimulatorFeatures.SIMULATION_STATE_PAUSE,
        SimulatorFeatures.STEP_SIMULATION_SINGLE,
        SimulatorFeatures.STEP_SIMULATION_MULTIPLE,
        SimulatorFeatures.STEP_SIMULATION_ACTION,
    ]


def test_state_transitions(stepped):
    stopped = SimulationState(state=SimulationState.STATE_STOPPED)
    assert call(stepped, GetSimulationState).state.state == SimulationState.STATE_PAUSED
    assert (
        call(stepped, SetSimulationState, state=PAUSED).result.result
        == SetSimulationState.Response.ALREADY_IN_TARGET_STATE
    )
    assert (
        call(stepped, SetSimulationState, state=stopped).result.result
        == SetSimulationState.Response.STATE_TRANSITION_ERROR
    )
