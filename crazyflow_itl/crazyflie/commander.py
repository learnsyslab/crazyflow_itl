import errno
import struct
from functools import partial
from typing import NamedTuple

import cffirmware
import numpy as np
from crazyflow.control import Control, load_params
from crazyflow.control.transform import pwm2force
from crazyflow.drones import Drone
from scipy.spatial.transform import Rotation

from crazyflow_itl.crazyflie import crtp


class Setpoint(NamedTuple):
    """Describe the command that the onboard controller follows.

    Attributes:
        control: Control mode of the command.
        cmd: Command of shape (16,) in state and (4,) in attitude and body rate control.
    """

    control: Control
    cmd: np.ndarray


class Commander:
    """Reimplementation of the commander to translate commander messages into drone commands."""

    # Watchdog timeouts of the firmware in seconds
    WATCHDOG_LEVEL_OUT = 0.5
    WATCHDOG_LOCK = 2.0

    def __init__(self, drone: Drone, params: dict[str, int | float]):
        """Create a commander in the boot state of the firmware.

        Args:
            drone: Drone model.
            params: Live drone params.
        """
        core = load_params("mellinger", drone)["core"]
        self.pwm_to_force = partial(
            pwm2force, thrust_max=4 * core["thrust_max"], pwm_max=core["pwm_max"]
        )
        self.pwm_min = core["pwm_min"]
        self.params = params
        self.locked = False
        self.thrust_locked = True
        self.high_level = True
        self.planner = cffirmware.planner()
        cffirmware.plan_init(self.planner)
        self.last_setpoint = None
        self.last_time = None
        self.motors_on_time = -np.inf

    def handle(
        self, port: int, channel: int, payload: bytes, t: float, state: np.ndarray
    ) -> list[bytes]:
        """Apply one commander, localization, supervisor or arming packet from the client.

        Args:
            port: CRTP port of the packet.
            channel: CRTP channel of the packet.
            payload: Packet data.
            t: Sim time in seconds.
            state: Drone state `[pos(3), vel(3), quat_xyzw(4), ang_vel(3)]` of shape (13,).

        Returns:
            The reply datagrams.
        """
        match port, channel:
            case crtp.Port.RPYT, 0:
                self._rpyt(payload, t)
            case crtp.Port.GENERIC, 0 if payload == b"\x00":  # Stop
                self._set_low_level(None, t)
            case crtp.Port.GENERIC, 0 if payload[0] == 6:  # Full state
                self._full_state(payload, t)
            case crtp.Port.GENERIC, 1 if payload[0] == 0:  # Notify setpoint stop
                self.high_level = True
            case crtp.Port.HLC, 0 if payload[0] == 12:  # Go to
                return [crtp.encode(port, channel, self._go_to(payload, t, state))]
            case crtp.Port.LOCALIZATION, 1 if payload[0] == 8:  # External pose
                pass
            # Emergency stop on either port
            case ((crtp.Port.LOCALIZATION | crtp.Port.SUPERVISOR), 1) if payload == b"\x03":
                self.locked = True
            case crtp.Port.PLATFORM, 0 if payload == b"\x01\x01":  # Arm
                return [crtp.encode(port, channel, b"\x01\x01\x01")]
            case crtp.Port.SUPERVISOR, 1 if payload == b"\x01\x01":  # Arm with response bit
                return [crtp.encode(port, channel, b"\x81\x01\x01")]
            case _:
                raise NotImplementedError(f"Unsupported packet on port {port} channel {channel}")
        return []

    def setpoint(self, t: float, state: np.ndarray) -> Setpoint | None:
        """Return the command the onboard controller follows at the given time.

        Low-level setpoints level out after 0.5 s without an update. A drone that was flying when
        its setpoint leveled out is locked off after 2 s.

        Args:
            t: Sim time in seconds.
            state: Drone state `[pos(3), vel(3), quat_xyzw(4), ang_vel(3)]` of shape (13,).

        Returns:
            The setpoint, or None if the motors are off.
        """
        if not self.high_level:
            # The supervisor counts the drone as flying for two seconds after the motors last ran
            level_out = self.last_time + self.WATCHDOG_LEVEL_OUT
            flying = level_out - self.motors_on_time < self.WATCHDOG_LOCK
            self.locked = self.locked or (flying and t - self.last_time > self.WATCHDOG_LOCK)
        setpoint = self._command(t, state)
        if setpoint is not None:
            self.motors_on_time = t
        return setpoint

    def _command(self, t: float, state: np.ndarray) -> Setpoint | None:
        """Return the setpoint of the active command source after the supervisor checks."""
        if self.locked:
            return None
        if self.high_level:
            if cffirmware.plan_is_stopped(self.planner):
                return None
            return Setpoint(Control.state, self._goal(t))
        if self.last_setpoint is None or t - self.last_time <= self.WATCHDOG_LEVEL_OUT:
            return self.last_setpoint
        control, cmd = self.last_setpoint
        # The supervisor levels at the current yaw, stops yawing and keeps the thrust or height
        # Firmware reference: https://github.com/bitcraze/crazyflie-firmware/blob/b48aba18a675fe950a88d96586fd09e49d82a38b/src/modules/src/supervisor.c#L625
        yaw = Rotation.from_quat(state[6:10]).as_euler("xyz")[2]
        if control == Control.attitude:
            return Setpoint(control, np.array([0, 0, yaw, cmd[3]]))
        # The roll and pitch rates of the last setpoint stay in the rate error
        if control == Control.body_rate:
            return Setpoint(control, np.array([*cmd[:2], 0, cmd[3]]))
        # Without horizontal position control, only the height and its velocity have errors
        quat = Rotation.from_euler("z", yaw).as_quat()
        level = np.concatenate(
            [state[:2], cmd[2:3], state[3:5], cmd[5:6], np.zeros(3), quat, cmd[13:15], [0]]
        )
        return Setpoint(control, level)

    def _rpyt(self, payload: bytes, t: float):
        """Apply a roll, pitch, yaw and thrust setpoint in angle or rate mode."""
        rate = [self.params[f"flightmode.stabMode{axis}"] == 0 for axis in ("Roll", "Pitch", "Yaw")]
        if any(rate) and not all(rate):
            raise NotImplementedError("RPYT setpoints with mixed rate and angle modes")
        roll, pitch, yaw, thrust = struct.unpack("<fffH", payload)
        self.thrust_locked = self.thrust_locked and thrust != 0
        control = Control.body_rate if all(rate) else Control.attitude
        # The firmware inverts the legacy pitch angle and yaw rate
        angles = [roll, pitch, -yaw] if all(rate) else [roll, -pitch, yaw]
        cmd = np.append(np.radians(angles), self.pwm_to_force(thrust))
        off = self.thrust_locked or thrust < self.pwm_min
        self._set_low_level(None if off else Setpoint(control, cmd), t)

    def _full_state(self, payload: bytes, t: float):
        """Apply a full state setpoint."""
        values = struct.unpack("<B9hI3h", payload)
        cmd = np.concatenate(
            [
                np.array(values[1:10]) / 1000,
                crtp.quat_decompress(values[10]),
                np.array(values[11:]) / 1000,
            ]
        )
        self._set_low_level(Setpoint(Control.state, cmd), t)

    def _set_low_level(self, setpoint: Setpoint | None, t: float):
        """Take over from the high-level commander with a low-level setpoint."""
        self.high_level = False
        self.last_setpoint = setpoint
        self.last_time = t
        cffirmware.plan_stop(self.planner)

    def _go_to(self, payload: bytes, t: float, state: np.ndarray) -> bytes:
        """Plan a go-to unless the drone is locked and return the reply payload."""
        _, group_mask, relative, linear, *goal, goal_yaw, duration = struct.unpack(
            "<BBBBfffff", payload
        )
        # The reply echoes the command id, the group mask and the relative flag
        reply = payload[:3]
        if self.locked:
            return reply + bytes([errno.EBUSY])
        # The firmware's own group mask is 0 by default, so only commands to all groups apply
        if group_mask != 0:
            return reply + bytes([0])
        relative, linear, goal = bool(relative), bool(linear), cffirmware.mkvec(*goal)
        if not cffirmware.plan_is_stopped(self.planner):
            result = cffirmware.plan_go_to(
                self.planner, relative, linear, goal, goal_yaw, duration, t
            )
            return reply + bytes([result])
        # A stopped planner starts from the drone state, like in the firmware
        current = cffirmware.traj_eval_zero()
        # The bindings take only Python floats, not the float32 of the sim state
        current.pos = cffirmware.mkvec(*state[:3].tolist())
        current.vel = cffirmware.mkvec(*state[3:6].tolist())
        current.yaw = Rotation.from_quat(state[6:10]).as_euler("xyz")[2]
        result = cffirmware.plan_go_to_from(
            self.planner, current, relative, linear, goal, goal_yaw, duration, t
        )
        return reply + bytes([result])

    def _goal(self, t: float) -> np.ndarray:
        """Return the planner goal at the given time as a state command."""
        goal = cffirmware.plan_current_goal(self.planner, t)
        quat = Rotation.from_euler("z", goal.yaw).as_quat()
        return np.concatenate([goal.pos, goal.vel, goal.acc, quat, goal.omega])
