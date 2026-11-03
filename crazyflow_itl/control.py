from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.control.mellinger import (
    MellingerAttitudeData,
    MellingerBodyRateData,
    MellingerStateData,
    control_attitude2force_torque,
    control_body_rate2force_torque,
    control_state2attitude,
)
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import insert_fn_before
from crazyflow.utils import CORE_NDIM_KEY, leaf_replace
from flax.struct import dataclass, field
from jax import Array

from crazyflow_itl.stop import STOP_KEY

CONTROL_KEY = "mixed_control"
MODES = (Control.state, Control.attitude, Control.body_rate)


@dataclass
class ControlData:
    """Hold the control mode, command and controllers of every drone.

    Attributes:
        mode: Indices of the state, attitude and body rate modes of shape (n_worlds, n_drones).
        command: Commands of shape (n_worlds, n_drones, 16).
        state: State controller.
        attitude: Attitude controller that follows the state controller in state mode.
        body_rate: Body rate controller.
    """

    mode: Array = field(metadata={CORE_NDIM_KEY: 0})
    command: Array = field(metadata={CORE_NDIM_KEY: 1})
    state: MellingerStateData
    attitude: MellingerAttitudeData
    body_rate: MellingerBodyRateData


def control_stage(data: SimData) -> SimData:
    """Stage the force and torque that the controller of each drone's mode commands.

    The controllers of the levels that a drone does not use and all controllers of stopped drones
    start from scratch.

    Args:
        data: Sim data with the control and stop data.

    Returns:
        The updated sim data.
    """
    control = data.plugins[CONTROL_KEY]
    command, stopped = control.command, data.plugins[STOP_KEY].stopped
    state_mode, attitude_mode, body_rate_mode = (control.mode == mode for mode in range(3))
    # 1. state -> attitude -> force and torque chain
    controls = data.controls.replace(
        state=control.state.replace(staged_cmd=command), attitude=control.attitude
    )
    state_data = control_attitude2force_torque(
        control_state2attitude(data.replace(controls=controls))
    )
    # 2. attitude -> force and torque chain
    attitude = control.attitude.replace(
        staged_cmd=command[..., :4], staged_ang_vel_des=jnp.zeros_like(control.attitude.ang_vel_des)
    )
    attitude_data = control_attitude2force_torque(
        data.replace(controls=data.controls.replace(attitude=attitude))
    )
    # 3. body rate -> force and torque chain
    body_rate = control.body_rate.replace(staged_cmd=command[..., :4])
    body_rate_data = control_body_rate2force_torque(
        data.replace(controls=data.controls.replace(body_rate=body_rate))
    )
    force_torque = jnp.select(
        [state_mode[..., None], attitude_mode[..., None]],
        [
            state_data.controls.force_torque.staged_cmd,
            attitude_data.controls.force_torque.staged_cmd,
        ],
        body_rate_data.controls.force_torque.staged_cmd,
    )
    # Both chains update the attitude rates and steps alike, so only the setpoints and errors differ
    attitude = attitude_data.controls.attitude
    attitude = leaf_replace(
        state_data.controls.attitude,
        attitude_mode,
        cmd=attitude.cmd,
        staged_cmd=attitude.staged_cmd,
        ang_vel_des=attitude.ang_vel_des,
        staged_ang_vel_des=attitude.staged_ang_vel_des,
        r_int_error=attitude.r_int_error,
    )
    state = leaf_replace(
        state_data.controls.state,
        stopped | ~state_mode,
        pos_err_i=jnp.zeros_like(control.state.pos_err_i),
        steps=-jnp.ones_like(control.state.steps),
    )
    attitude = leaf_replace(
        attitude,
        stopped | ~(state_mode | attitude_mode),
        ang_vel_des=jnp.zeros_like(control.attitude.ang_vel_des),
        r_int_error=jnp.zeros_like(control.attitude.r_int_error),
        prev_ang_vel=jnp.zeros_like(control.attitude.prev_ang_vel),
        steps=-jnp.ones_like(control.attitude.steps),
    )
    body_rate = leaf_replace(
        body_rate_data.controls.body_rate,
        stopped | ~body_rate_mode,
        cmd=jnp.zeros_like(control.body_rate.cmd),
        r_int_error=jnp.zeros_like(control.body_rate.r_int_error),
        prev_ang_vel=jnp.zeros_like(control.body_rate.prev_ang_vel),
        steps=-jnp.ones_like(control.body_rate.steps),
    )
    control = control.replace(state=state, attitude=attitude, body_rate=body_rate)
    force_torque = data.controls.force_torque.replace(staged_cmd=force_torque)
    return data.replace(
        controls=data.controls.replace(force_torque=force_torque),
        plugins=data.plugins | {CONTROL_KEY: control},
    )


def insert_control_stage(
    sim: Sim, state_freq: int = 100, attitude_freq: int = 500, body_rate_freq: int = 500
) -> Sim:
    """Add the control stage to the step pipeline with all drones in state mode.

    The sim must use force and torque control with first principles dynamics and have the stop
    stage.

    Args:
        sim: The simulation.
        state_freq: State controller rate in Hz.
        attitude_freq: Attitude controller rate in Hz.
        body_rate_freq: Body rate controller rate in Hz.

    Returns:
        The simulation with the stage.
    """
    assert sim.control == Control.force_torque and sim.dynamics == Dynamics.first_principles
    # Controllers keep one step counter per drone to restart each drone on its own
    steps = -jnp.ones((1, sim.n_drones), int, device=sim.device)
    state = MellingerStateData.create(1, sim.n_drones, state_freq, sim.drone, sim.device)
    attitude = MellingerAttitudeData.create(1, sim.n_drones, attitude_freq, sim.drone, sim.device)
    body_rate = MellingerBodyRateData.create(1, sim.n_drones, body_rate_freq, sim.drone, sim.device)
    control = ControlData(
        mode=jnp.zeros((1, sim.n_drones), int, device=sim.device),
        command=state.cmd,
        state=state.replace(steps=steps),
        attitude=attitude.replace(steps=steps),
        body_rate=body_rate.replace(steps=steps),
    )
    sim.data = sim.data.replace(plugins=sim.data.plugins | {CONTROL_KEY: control})
    insert_fn_before(sim.step_pipeline, "force_torque_controller", control_stage, name=CONTROL_KEY)
    sim.build_default_data()
    sim.build_step_fn()
    return sim


def command_drones(sim: Sim, modes: Sequence[Control], commands: np.ndarray):
    """Set the control mode and command of every drone.

    Args:
        sim: The simulation with the control stage.
        modes: Control mode per drone, either state, attitude or body rate.
        commands: Commands of shape (n_drones, 16). Attitude and body rate commands take the first
            4 entries.
    """
    # Arrays on the sim device keep the step function from recompiling
    control = sim.data.plugins[CONTROL_KEY].replace(
        mode=jnp.asarray([[MODES.index(mode) for mode in modes]], device=sim.device),
        command=jnp.asarray(commands[None], device=sim.device),
    )
    sim.data = sim.data.replace(plugins=sim.data.plugins | {CONTROL_KEY: control})
