import asyncio
import socket
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from cflib2 import Crazyflie, LinkContext
from cflib2.error import LinkError
from cflib2.toc_cache import FileTocCache
from crazyflow.control import Control
from crazyflow.sim import Sim

from crazyflow_itl import RealTimePlugin, run
from crazyflow_itl.crazyflie import CrazyfliePlugin

pytestmark = pytest.mark.timeout(60)

HOST = "127.0.0.1"
# The param writes of a deployment client before flight, ending with the switch to low-level
# setpoints. They cover every param of the plugin and set the angle mode on all axes.
SETTINGS = [
    ("stabilizer.estimator", 2),
    ("supervisor.tmblChckEn", 1),
    ("stabilizer.controller", 2),
    ("flightmode.stabModeRoll", 1),
    ("flightmode.stabModePitch", 1),
    ("flightmode.stabModeYaw", 1),
    ("kalman.initialX", 0.0),
    ("kalman.initialY", 0.0),
    ("kalman.initialZ", 0.0),
    ("kalman.initialYaw", 0.0),
    ("kalman.resetEstimation", 1),
    ("kalman.resetEstimation", 0),
    ("commander.enHighLevel", 0),
]


def start(pool: ThreadPoolExecutor, sim: Sim, port: int, duration: float) -> Future:
    """Run the sim with a Crazyflie plugin in the background until the sim time reaches 0.1 s."""
    future = pool.submit(run, sim, [RealTimePlugin(), CrazyfliePlugin(base_port=port)], duration)
    # The step function compiles before the loop, and a client probing meanwhile times out
    while float(sim.time[0, 0]) < 0.1:
        assert not future.done(), future.exception()
        time.sleep(0.01)
    return future


async def connect(port: int, cache_dir: Path) -> Crazyflie:
    """Connect a client with a file table cache to the drone on a port."""
    uri = f"udp://{HOST}:{port}"
    return await Crazyflie.connect_from_uri(LinkContext(), uri, FileTocCache(str(cache_dir)))


async def sleep_until(sim: Sim, future: Future, t: float):
    """Sleep until the sim time reaches t, failing if the run ends first."""
    while float(sim.time[0, 0]) < t:
        assert not future.done(), future.exception()
        await asyncio.sleep(0.01)


async def hover(cf: Crazyflie, sim: Sim, future: Future, duration: float):
    """Stream full state setpoints 0.5 m above the origin and the sim pose for a sim duration."""
    end = float(sim.time[0, 0]) + duration
    while float(sim.time[0, 0]) < end:
        assert not future.done(), future.exception()
        states = sim.data.states
        await cf.commander().send_setpoint_full_state(
            [0.0, 0.0, 0.5], [0.0] * 3, [0.0] * 3, [0.0, 0.0, 0.0, 1.0], [0.0] * 3
        )
        # A deployment client feeds the motion capture pose to the onboard estimator
        await (
            cf.localization()
            .external_pose()
            .send_external_pose(states.pos[0, 0].tolist(), states.quat[0, 0].tolist())
        )
        await asyncio.sleep(0.02)


def test_cflib2_has_udp_support(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((HOST, 0))
        port = sock.getsockname()[1]
    # A cflib2 build without the UDP link rejects the URI as invalid instead of probing the port
    with pytest.raises(LinkError, match="refused"):
        asyncio.run(connect(port, tmp_path))


async def set_params(port: int, cache_dir: Path):
    """Write the settings and read them back."""
    cf = await connect(port, cache_dir)
    for name, value in SETTINGS:
        await cf.param().set(name, value)
    values = dict(SETTINGS)
    assert {name: await cf.param().get(name) for name in values} == values
    await cf.disconnect()


def test_cflib2_connects_and_sets_params(pool, tmp_path):
    future = start(pool, Sim(control=Control.force_torque), 19810, 5.0)
    try:
        asyncio.run(set_params(19810, tmp_path))
    finally:
        future.result()


async def reconnect(port: int, cache_dir: Path):
    """Change a param, reconnect with the cached tables and read the param from the drone."""
    cf = await connect(port, cache_dir)
    await cf.param().set("flightmode.stabModeYaw", 1)
    await cf.disconnect()
    # The log and param tables have distinct checksums and are cached separately
    assert len(list(cache_dir.glob("*.json"))) == 2
    cf = await connect(port, cache_dir)
    assert await cf.param().get("flightmode.stabModeYaw") == 1
    await cf.disconnect()


def test_cflib2_toc_cache_reconnect(pool, tmp_path):
    future = start(pool, Sim(control=Control.force_torque), 19820, 6.0)
    try:
        asyncio.run(reconnect(19820, tmp_path))
    finally:
        future.result()


async def power_cycle_then_connect(port: int, cache_dir: Path):
    """Change a param, power cycle the drone and read the param after the reboot."""
    cf = await connect(port, cache_dir)
    await cf.param().set("flightmode.stabModeYaw", 1)
    await cf.disconnect()
    context, uri = LinkContext(), f"udp://{HOST}:{port}"
    await Crazyflie.power_off_stm32_domain(context, uri)
    await asyncio.sleep(0.1)
    await Crazyflie.power_on_stm32_domain(context, uri)
    cf = await connect(port, cache_dir)
    assert await cf.param().get("flightmode.stabModeYaw") == 0, "The drone did not reboot"
    await cf.disconnect()


def test_cflib2_power_cycle_then_connect(pool, tmp_path):
    future = start(pool, Sim(control=Control.force_torque), 19830, 7.0)
    try:
        asyncio.run(power_cycle_then_connect(19830, tmp_path))
    finally:
        future.result()


async def fly_full_state(sim: Sim, future: Future, port: int, cache_dir: Path):
    """Hover with full state setpoints for three seconds."""
    cf = await connect(port, cache_dir)
    for name, value in SETTINGS:
        await cf.param().set(name, value)
    await hover(cf, sim, future, 3.0)
    assert abs(sim.data.states.pos[0, 0, 2] - 0.5) < 0.15
    await cf.disconnect()


def test_cflib2_full_state_hover(pool, tmp_path):
    sim = Sim(control=Control.force_torque)
    future = start(pool, sim, 19840, 8.0)
    try:
        asyncio.run(fly_full_state(sim, future, 19840, tmp_path))
    finally:
        future.result()


async def rpyt_then_go_to(sim: Sim, future: Future, port: int, cache_dir: Path):
    """Arm and unlock the thrust, hand over to the high-level commander and go to a point."""
    cf = await connect(port, cache_dir)
    for name, value in SETTINGS:
        await cf.param().set(name, value)
    await cf.supervisor().send_arming_request(True)
    # Thrust stays locked after boot until a setpoint with zero thrust arrives
    await cf.commander().send_setpoint_rpyt(0.0, 0.0, 0.0, 0)
    await cf.commander().send_stop_setpoint()
    await cf.commander().send_notify_setpoint_stop(0)
    await cf.param().set("commander.enHighLevel", 1)
    await cf.high_level_commander().go_to(0.3, 0.0, 0.5, 0.0, 2.0, False, False, None)
    await sleep_until(sim, future, float(sim.time[0, 0]) + 3.0)
    np.testing.assert_allclose(sim.data.states.pos[0, 0, :2], [0.3, 0.0], atol=0.05)
    assert abs(sim.data.states.pos[0, 0, 2] - 0.5) < 0.15
    await cf.disconnect()


def test_cflib2_rpyt_and_go_to(pool, tmp_path):
    sim = Sim(control=Control.force_torque)
    future = start(pool, sim, 19850, 8.0)
    try:
        asyncio.run(rpyt_then_go_to(sim, future, 19850, tmp_path))
    finally:
        future.result()


async def log_position(sim: Sim, port: int, cache_dir: Path):
    """Stream the estimated position at 10 ms and compare one sample with the sim position."""
    cf = await connect(port, cache_dir)
    names = [f"stateEstimate.{axis}" for axis in "xyz"]
    block = await cf.log().create_block()
    for name in names:
        await block.add_variable(name)
    stream = await block.start(10)
    data = (await stream.next()).data
    # Dropping a running stream ends the log dispatch of the client
    await stream.stop()
    await cf.disconnect()
    np.testing.assert_allclose([data[name] for name in names], sim.data.states.pos[0, 0], atol=1e-3)


def test_cflib2_log_stream(pool, tmp_path):
    sim = Sim(control=Control.force_torque)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=np.asarray([[[0.5, -0.3, 0.0]]]))
    )
    future = start(pool, sim, 19860, 5.0)
    try:
        asyncio.run(log_position(sim, 19860, tmp_path))
    finally:
        future.result()


async def emergency_stop(sim: Sim, future: Future, port: int, cache_dir: Path):
    """Take off, stop the drone in flight and keep sending setpoints."""
    cf = await connect(port, cache_dir)
    for name, value in SETTINGS:
        await cf.param().set(name, value)
    await hover(cf, sim, future, 2.0)
    assert sim.data.states.pos[0, 0, 2] > 0.3, "The drone did not take off"
    await cf.supervisor().send_emergency_stop()
    await hover(cf, sim, future, 1.0)
    assert sim.data.states.pos[0, 0, 2] < 0.1, "The stopped drone did not fall"
    await cf.disconnect()


def test_cflib2_emergency_stop(pool, tmp_path):
    sim = Sim(control=Control.force_torque)
    future = start(pool, sim, 19870, 8.0)
    try:
        asyncio.run(emergency_stop(sim, future, 19870, tmp_path))
    finally:
        future.result()
