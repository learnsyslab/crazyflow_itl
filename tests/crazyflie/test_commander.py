import errno
import struct

import numpy as np
import pytest
from crazyflow.control import Control, load_params
from crazyflow.control.transform import pwm2force
from crazyflow.drones import Drone
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation
from test_crtp import compress

from crazyflow_itl.crazyflie import crtp
from crazyflow_itl.crazyflie.commander import Commander
from crazyflow_itl.crazyflie.link import Link

STATE = np.array([0.5, -0.25, 0.1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
CORE = load_params("mellinger", Drone.cf21B_500)["core"]
THRUST_30000 = pwm2force(30000, 4 * CORE["thrust_max"], CORE["pwm_max"])
NOTIFY_STOP = struct.pack("<BI", 0, 0)
QUAT = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()
YAWED = np.concatenate([STATE[:6], Rotation.from_euler("z", 0.4).as_quat(), STATE[10:]])


def rpyt(roll, pitch, yaw, thrust):
    """Pack an RPYT setpoint like cflib2, which negates the pitch on the wire."""
    return struct.pack("<fffH", roll, -pitch, yaw, thrust)


def full_state(pos, vel=(0, 0, 0), acc=(0, 0, 0), quat=QUAT, rates=(0, 0, 0)):
    """Pack a full state setpoint in the layout of cflib2."""
    values = [round(value * 1000) for value in (*pos, *vel, *acc)]
    return struct.pack(
        "<B9hI3h", 6, *values, compress(quat), *(round(rate * 1000) for rate in rates)
    )


def go_to(goal, duration, group_mask=0):
    """Pack an absolute smooth go-to to the goal with zero yaw."""
    return struct.pack("<BBBBfffff", 12, group_mask, 0, 0, *goal, 0.0, duration)


def create(stab_mode=1):
    """Create a commander on fresh params whose RPYT setpoints use one mode on all axes.

    Mode 1 is the angle mode and mode 0 the rate mode.
    """
    axes = ("Roll", "Pitch", "Yaw")
    params = Link().params | {f"flightmode.stabMode{axis}": stab_mode for axis in axes}
    return Commander(Drone.cf21B_500, params)


def unlocked(t=0.0, stab_mode=1):
    """Create a commander whose thrust lock was released at time t."""
    commander = create(stab_mode)
    assert commander.handle(3, 0, rpyt(0, 0, 0, 0), t, STATE) == []
    return commander


def test_boot_is_motors_off():
    assert create().setpoint(0.0, STATE) is None


def test_rpyt_needs_unlock():
    commander = create()
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 0.0, STATE)
    assert commander.setpoint(0.0, STATE) is None
    commander.handle(3, 0, rpyt(0, 0, 0, 0), 0.01, STATE)
    assert commander.setpoint(0.01, STATE) is None
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 0.02, STATE)
    control, cmd = commander.setpoint(0.02, STATE)
    assert control == Control.attitude
    assert cmd[3] == THRUST_30000


def test_rpyt_sign_conventions():
    commander = unlocked()
    payload = rpyt(10, -5, 20, 30000)
    assert payload == struct.pack("<fffH", 10, 5, 20, 30000)
    commander.handle(3, 0, payload, 0.0, STATE)
    assert_allclose(commander.setpoint(0.0, STATE).cmd[:3], np.radians([10, -5, 20]), rtol=1e-6)


def test_rpyt_rate_mode_commands_body_rates():
    commander = unlocked(stab_mode=0)
    commander.handle(3, 0, rpyt(10, -5, 20, 30000), 0.0, STATE)
    control, cmd = commander.setpoint(0.0, STATE)
    assert control == Control.body_rate
    # The pitch rate on the wire enters unchanged and the legacy yaw rate inverted
    assert_allclose(cmd, [*np.radians([10, 5, -20]), THRUST_30000], rtol=1e-6)
    # Leveling out stops only the yaw rate
    control, cmd = commander.setpoint(0.6, STATE)
    assert control == Control.body_rate
    assert_allclose(cmd, [*np.radians([10, 5]), 0, THRUST_30000], rtol=1e-6)


def test_rpyt_mixed_modes_raise():
    commander = Commander(Drone.cf21B_500, Link().params)
    with pytest.raises(NotImplementedError, match="mixed"):
        commander.handle(3, 0, rpyt(0, 0, 0, 0), 0.0, STATE)


def test_rpyt_below_min_thrust_is_off():
    commander = unlocked()
    commander.handle(3, 0, rpyt(0, 0, 0, int(CORE["pwm_min"]) - 1), 0.0, STATE)
    assert commander.setpoint(0.0, STATE) is None
    commander.handle(3, 0, rpyt(0, 0, 0, int(CORE["pwm_min"])), 0.01, STATE)
    assert commander.setpoint(0.01, STATE) is not None


def test_generic_stop_is_off():
    commander = unlocked()
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 0.0, STATE)
    assert commander.handle(7, 0, b"\x00", 0.01, STATE) == []
    assert commander.setpoint(0.01, STATE) is None


def test_full_state_setpoint():
    commander = create()
    payload = full_state((1, -0.5, 0.75), (0.1, 0.2, -0.3), (0.5, 0, -1), QUAT, (0.1, -0.2, 0.3))
    commander.handle(7, 0, payload, 0.0, STATE)
    control, cmd = commander.setpoint(0.0, STATE)
    assert control == Control.state
    assert_allclose(cmd[:9], [1, -0.5, 0.75, 0.1, 0.2, -0.3, 0.5, 0, -1])
    assert min(np.abs(cmd[9:13] - QUAT).max(), np.abs(cmd[9:13] + QUAT).max()) < 2e-3
    assert_allclose(cmd[13:], [0.1, -0.2, 0.3])


def test_mode_follows_setpoint():
    commander = unlocked()
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 0.0, STATE)
    assert commander.setpoint(0.0, STATE).control == Control.attitude
    commander.handle(7, 0, full_state((0, 0, 1)), 0.01, STATE)
    assert commander.setpoint(0.01, STATE).control == Control.state


def test_go_to_after_notify_stop():
    commander = create()
    commander.handle(7, 0, full_state((0, 0, 1)), 0.0, STATE)
    assert commander.handle(7, 1, NOTIFY_STOP, 0.1, STATE) == []
    reply = commander.handle(8, 0, go_to((1, 0, 1), 2.0), 0.2, STATE)
    assert reply == [crtp.encode(8, 0, bytes([12, 0, 0, 0]))]
    control, cmd = commander.setpoint(0.2, STATE)
    assert control == Control.state
    # The firmware planner computes in float32
    assert_allclose(cmd[:3], STATE[:3], atol=1e-5)
    assert_allclose(commander.setpoint(2.2, STATE).cmd[:3], [1, 0, 1], atol=1e-5)


def test_go_to_for_other_groups_is_ignored():
    commander = create()
    reply = commander.handle(8, 0, go_to((1, 0, 1), 2.0, group_mask=1), 0.0, STATE)
    assert reply == [crtp.encode(8, 0, bytes([12, 1, 0, 0]))]
    assert commander.setpoint(1.0, STATE) is None


def test_low_level_setpoint_stops_trajectory():
    commander = create()
    commander.handle(8, 0, go_to((1, 0, 1), 2.0), 0.0, STATE)
    assert commander.setpoint(1.0, STATE) is not None
    commander.handle(7, 0, full_state((0, 0, 0.3)), 1.0, STATE)
    assert_allclose(commander.setpoint(1.0, STATE).cmd[:3], [0, 0, 0.3])
    commander.handle(7, 1, NOTIFY_STOP, 1.1, STATE)
    assert commander.setpoint(1.1, STATE) is None


def test_idle_high_level_is_off():
    commander = create()
    commander.handle(7, 0, full_state((0, 0, 1)), 0.0, STATE)
    commander.handle(7, 1, NOTIFY_STOP, 0.1, STATE)
    assert commander.setpoint(0.1, STATE) is None


def test_watchdog_level_out_and_lock():
    commander = unlocked()
    commander.handle(3, 0, rpyt(10, -5, 20, 30000), 0.0, STATE)
    assert_allclose(commander.setpoint(0.6, YAWED).cmd, [0, 0, 0.4, THRUST_30000])
    assert commander.setpoint(2.1, STATE) is None
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 2.2, STATE)
    assert commander.setpoint(2.2, STATE) is None
    fresh = unlocked(2.3)
    fresh.handle(3, 0, rpyt(0, 0, 0, 30000), 2.3, STATE)
    assert fresh.setpoint(2.3, STATE) is not None


def test_watchdog_locks_after_motors_off():
    commander = unlocked()
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 0.0, STATE)
    assert commander.setpoint(0.0, STATE) is not None
    commander.handle(7, 0, b"\x00", 0.01, STATE)
    assert commander.setpoint(2.1, STATE) is None
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 2.2, STATE)
    assert commander.setpoint(2.2, STATE) is None


def test_state_level_out():
    commander = create()
    payload = full_state((1, 2, 0.75), (0.1, 0, -0.2), (0.5, 0, 0), rates=(0.1, -0.2, 0.3))
    commander.handle(7, 0, payload, 0.0, STATE)
    moving = np.concatenate([YAWED[:3], [0.3, -0.1, 0.5], YAWED[6:]])
    cmd = commander.setpoint(0.6, moving).cmd
    assert_allclose(cmd[:9], [STATE[0], STATE[1], 0.75, 0.3, -0.1, -0.2, 0, 0, 0])
    assert_allclose(Rotation.from_quat(cmd[9:13]).as_euler("xyz"), [0, 0, 0.4], atol=1e-9)
    assert_allclose(cmd[13:], [0.1, -0.2, 0])


def test_watchdog_keeps_idle_motors_unlocked():
    commander = unlocked()
    assert commander.setpoint(3.0, STATE) is None
    commander.handle(3, 0, rpyt(0, 0, 0, 30000), 3.0, STATE)
    assert commander.setpoint(3.0, STATE) is not None


def test_emergency_stop_locks():
    for port in (6, 9):
        commander = unlocked()
        commander.handle(3, 0, rpyt(0, 0, 0, 30000), 0.0, STATE)
        assert commander.handle(port, 1, b"\x03", 0.1, STATE) == []
        assert commander.setpoint(0.1, STATE) is None
        reply = commander.handle(8, 0, go_to((1, 0, 1), 2.0), 0.2, STATE)
        assert reply == [crtp.encode(8, 0, bytes([12, 0, 0, errno.EBUSY]))]
        commander.handle(3, 0, rpyt(0, 0, 0, 30000), 0.3, STATE)
        assert commander.setpoint(0.3, STATE) is None
        fresh = unlocked(0.4)
        fresh.handle(3, 0, rpyt(0, 0, 0, 30000), 0.4, STATE)
        assert fresh.setpoint(0.4, STATE) is not None


def test_arming_replies():
    commander = create()
    assert commander.handle(13, 0, b"\x01\x01", 0.0, STATE) == [crtp.encode(13, 0, b"\x01\x01\x01")]
    assert commander.handle(9, 1, b"\x01\x01", 0.0, STATE) == [crtp.encode(9, 1, b"\x81\x01\x01")]


def test_external_pose_ignored():
    commander = unlocked()
    commander.handle(3, 0, rpyt(10, -5, 20, 30000), 0.0, STATE)
    before = commander.setpoint(0.0, STATE)
    pose = struct.pack("<B7f", 8, 1, 2, 3, 0, 0, 0, 1)
    assert commander.handle(6, 1, pose, 0.0, STATE) == []
    after = commander.setpoint(0.0, STATE)
    assert after.control == before.control
    assert_array_equal(after.cmd, before.cmd)


def test_unknown_packet_raises():
    with pytest.raises(NotImplementedError, match="port 8"):
        create().handle(8, 0, b"\x01", 0.0, STATE)
