import time

import numpy as np
import pytest
from crazyflow.control import Control
from crazyflow.sim import Sim
from std_msgs.msg import Float64MultiArray

from crazyflow_itl import run
from crazyflow_itl.control import CONTROL_KEY, MODES
from crazyflow_itl.ros import CommandPlugin, SpinPlugin, drone_names


class Sender:
    """Publish one command in setup and serve callbacks until the command plugin has applied it."""

    freq = 100
    timeout = 5.0

    def __init__(self, node, topic, data, spin, command):
        self.publisher = node.create_publisher(Float64MultiArray, topic, 10)
        self.data = data
        self.spin = spin
        self.command = command

    def setup(self, sim):
        deadline = time.monotonic() + self.timeout
        while self.publisher.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "command subscription was not discovered"
            time.sleep(0.01)
        self.publisher.publish(Float64MultiArray(data=self.data))
        while not np.allclose(self.command.commands[0, : len(self.data)], self.data):
            assert time.monotonic() < deadline, "command was not delivered"
            self.spin.step(sim)
            time.sleep(0.01)
        return sim

    def step(self, sim):
        return sim

    def close(self):
        pass


def make_sim(z=0.0):
    sim = Sim(control=Control.force_torque)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=np.asarray([[[0.0, 0.0, z]]])))
    sim.build_default_data()
    return sim


def run_commands(node, sim, mode, data, duration):
    names = drone_names(sim)
    spin, command = SpinPlugin(node), CommandPlugin(node)
    plugins = [spin, command]
    if data is not None:
        topic = f"/crazyflow/{names[0]}/command/{mode}"
        plugins.append(Sender(node, topic, data, spin, command))
    return run(sim, plugins, duration=duration)


def test_state_command_moves_drone(node):
    data = [0.0, 0.0, 0.5, *[0.0] * 6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    sim = run_commands(node, make_sim(), Control.state, data, duration=3.0)
    assert float(sim.data.states.pos[0, 0, 2]) == pytest.approx(0.5, abs=0.15)


@pytest.mark.parametrize("mode", [Control.attitude, Control.body_rate])
def test_command_switches_mode(node, mode):
    data = [0.1, 0.0, 0.0, 0.3]
    sim = run_commands(node, make_sim(), mode, data, duration=0.2)
    control = sim.data.plugins[CONTROL_KEY]
    assert control.mode[0, 0] == MODES.index(mode)
    assert np.asarray(control.command[0, 0, :4]) == pytest.approx(data)


def test_motors_off_until_first_command(node):
    sim = run_commands(node, make_sim(z=1.0), Control.state, None, duration=1.0)
    assert float(sim.data.states.pos[0, 0, 2]) < 0.5, "The drone did not fall with its motors off"


def test_wrong_length_raises(node):
    with pytest.raises(AssertionError, match="Command of length 4, not 16"):
        run_commands(node, make_sim(), Control.state, [0.0] * 4, duration=0.5)
