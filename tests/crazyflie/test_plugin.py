import socket
import struct
import time
from concurrent.futures import Future
from contextlib import suppress

import numpy as np
import pytest
from crazyflow.control import Control, load_params
from crazyflow.control.transform import force2pwm
from crazyflow.sim import Sim

from crazyflow_itl import RealTimePlugin, run
from crazyflow_itl.control import CONTROL_KEY, MODES
from crazyflow_itl.crazyflie import CrazyfliePlugin, crtp
from crazyflow_itl.crazyflie.link import Link

pytestmark = pytest.mark.timeout(60)

HOST = "127.0.0.1"
# The compressed identity quaternion carries the index 3 of its largest component w in the top bits
IDENTITY = 3 << 30
PARAM_NAMES = [variable.name for variable in Link.PARAM_TOC]
STAB_MODE_YAW = PARAM_NAMES.index("flightmode.stabModeYaw")
STAB_MODES_ROLL_PITCH = [
    PARAM_NAMES.index(f"flightmode.stabMode{axis}") for axis in ("Roll", "Pitch")
]


def client() -> socket.socket:
    """Open a client socket on the loopback address."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, 0))
    return sock


def connect(sock: socket.socket, port: int):
    """Probe the plugin on a port until it answers, which it does once the sim runs."""
    sock.settimeout(0.1)
    reply = None
    while reply != b"\xff":
        sock.sendto(b"\xff", (HOST, port))
        with suppress(TimeoutError):
            reply = sock.recv(64)
    sock.settimeout(5.0)


def request(sock: socket.socket, port: int, datagram: bytes) -> bytes:
    """Send a datagram and return the next reply that is not a probe echo."""
    sock.sendto(datagram, (HOST, port))
    while (reply := sock.recv(64)) == b"\xff":
        pass
    return reply


def fly(sock: socket.socket, port: int, sim: Sim, future: Future, datagram: bytes, until: float):
    """Send a datagram at 50 Hz until the sim time reaches `until` or the run ends."""
    while not future.done() and float(sim.time[0, 0]) < until:
        sock.sendto(datagram, (HOST, port))
        time.sleep(0.02)


def full_state(z: float) -> bytes:
    """Pack a full state setpoint at height z above the origin with zero yaw at rest."""
    payload = struct.pack("<B9hI3h", 6, 0, 0, round(z * 1000), *[0] * 6, IDENTITY, 0, 0, 0)
    return crtp.encode(crtp.Port.GENERIC, 0, payload)


def test_plugin_echoes_probe(pool):
    sim = Sim(control=Control.force_torque)
    future = pool.submit(run, sim, [RealTimePlugin(), CrazyfliePlugin(base_port=19900)], 2.0)
    with client() as sock:
        connect(sock, 19900)
    future.result()


def test_plugin_replies_to_latest_peer(pool):
    sim = Sim(control=Control.force_torque)
    future = pool.submit(run, sim, [RealTimePlugin(), CrazyfliePlugin(base_port=19910)], 3.0)
    with client() as first, client() as second:
        connect(first, 19910)
        connect(second, 19910)
        version = crtp.encode(crtp.Port.PLATFORM, 1, b"\x00")
        assert request(second, 19910, version) == b"\xd1\x00\x0c\x00\x00\x00"
    future.result()


def test_plugin_full_state_flies(pool):
    sim = Sim(control=Control.force_torque)
    future = pool.submit(run, sim, [RealTimePlugin(), CrazyfliePlugin(base_port=19920)], 3.0)
    with client() as sock:
        connect(sock, 19920)
        fly(sock, 19920, sim, future, full_state(0.5), 3.0)
    assert abs(future.result().data.states.pos[0, 0, 2] - 0.5) < 0.15


def test_plugin_go_to_after_rpyt(pool):
    sim = Sim(control=Control.force_torque)
    future = pool.submit(run, sim, [RealTimePlugin(), CrazyfliePlugin(base_port=19930)], 4.0)
    with client() as sock:
        connect(sock, 19930)
        # RPYT setpoints take one mode on all axes, and the yaw axis boots in rate mode
        stab_mode_yaw = crtp.encode(crtp.Port.PARAM, 2, struct.pack("<HB", STAB_MODE_YAW, 1))
        assert request(sock, 19930, stab_mode_yaw) == stab_mode_yaw
        sock.sendto(crtp.encode(crtp.Port.RPYT, 0, struct.pack("<fffH", 0, 0, 0, 0)), (HOST, 19930))
        sock.sendto(crtp.encode(crtp.Port.GENERIC, 1, struct.pack("<BI", 0, 0)), (HOST, 19930))
        go_to = struct.pack("<BBBBfffff", 12, 0, 0, 0, 0, 0, 0.5, 0, 2)
        reply = request(sock, 19930, crtp.encode(crtp.Port.HLC, 0, go_to))
        assert reply == crtp.encode(crtp.Port.HLC, 0, bytes([12, 0, 0, 0]))
    assert abs(future.result().data.states.pos[0, 0, 2] - 0.5) < 0.15


def test_plugin_rpyt_rate_mode_spins(pool):
    sim = Sim(control=Control.force_torque)
    future = pool.submit(run, sim, [RealTimePlugin(), CrazyfliePlugin(base_port=19960)], 3.0)
    core = load_params("mellinger", sim.drone)["core"]
    weight = float(sim.data.params.mass[0]) * np.linalg.norm(sim.data.params.gravity_vec)
    lift = int(force2pwm(1.2 * weight, 4 * core["thrust_max"], core["pwm_max"]))
    with client() as sock:
        connect(sock, 19960)
        # The yaw axis boots in rate mode, and roll and pitch follow
        for index in STAB_MODES_ROLL_PITCH:
            rate_mode = crtp.encode(crtp.Port.PARAM, 2, struct.pack("<HB", index, 0))
            assert request(sock, 19960, rate_mode) == rate_mode
        sock.sendto(crtp.encode(crtp.Port.RPYT, 0, struct.pack("<fffH", 0, 0, 0, 0)), (HOST, 19960))
        # The firmware inverts the legacy yaw rate
        spin = crtp.encode(crtp.Port.RPYT, 0, struct.pack("<fffH", 0, 0, -30, lift))
        fly(sock, 19960, sim, future, spin, 2.0)
        assert sim.data.plugins[CONTROL_KEY].mode[0, 0] == MODES.index(Control.body_rate)
        assert sim.data.states.pos[0, 0, 2] > 0.1, "The drone did not take off"
        assert abs(sim.data.states.ang_vel[0, 0, 2] - np.radians(30)) < 0.05
    future.result()


def test_plugin_reboot_clears_lock(pool):
    sim = Sim(control=Control.force_torque)
    future = pool.submit(run, sim, [RealTimePlugin(), CrazyfliePlugin(base_port=19940)], 6.0)
    with client() as sock:
        connect(sock, 19940)
        fly(sock, 19940, sim, future, full_state(0.5), 2.0)
        sock.sendto(crtp.encode(crtp.Port.SUPERVISOR, 1, b"\x03"), (HOST, 19940))
        fly(sock, 19940, sim, future, full_state(0.5), 3.0)
        assert sim.data.states.pos[0, 0, 2] < 0.1, "The locked drone did not fall"
        sock.sendto(b"\xff\xfe\x03", (HOST, 19940))
        fly(sock, 19940, sim, future, full_state(0.5), 6.0)
    assert abs(future.result().data.states.pos[0, 0, 2] - 0.5) < 0.15


def test_plugin_unknown_packet_raises(pool):
    plugin = CrazyfliePlugin(base_port=19950)
    sim = Sim(n_drones=2, control=Control.force_torque)
    future = pool.submit(run, sim, [RealTimePlugin(), plugin], 10.0)
    with client() as sock:
        connect(sock, 19951)
        sock.sendto(crtp.encode(crtp.Port.CONSOLE, 0, b""), (HOST, 19951))
        with pytest.raises(NotImplementedError):
            future.result()
    assert all(sock.fileno() == -1 for sock in plugin.sockets)
