from functools import partial

import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import insert_fn_after
from crazyflow.utils import CORE_NDIM_KEY
from flax.struct import dataclass, field
from jax import Array

STOP_KEY = "motor_stop"


@dataclass
class StopData:
    """Mark the stopped drones.

    Attributes:
        stopped: Stop flags of shape (n_worlds, n_drones).
    """

    stopped: Array = field(metadata={CORE_NDIM_KEY: 0})


def stop_stage(data: SimData, dynamics: Dynamics) -> SimData:
    """Zero the thrust and controller integrators of stopped drones.

    Args:
        data: Sim data with the stop data.
        dynamics: Sim dynamics.

    Returns:
        The updated sim data.
    """
    stopped = data.plugins[STOP_KEY].stopped
    controls = data.controls
    # The firmware resets its controller while the motors are off. Which controllers exist
    # depends on the control mode.
    if controls.state is not None:
        pos_err_i = jnp.where(stopped[..., None], 0, controls.state.pos_err_i)
        controls = controls.replace(state=controls.state.replace(pos_err_i=pos_err_i))
    if controls.attitude is not None:
        r_int_error = jnp.where(stopped[..., None], 0, controls.attitude.r_int_error)
        controls = controls.replace(
            attitude=controls.attitude.replace(r_int_error=r_int_error)
        )
    if dynamics == Dynamics.first_principles:
        rotor_vel = jnp.where(stopped[..., None], 0, controls.rotor_vel)
        return data.replace(controls=controls.replace(rotor_vel=rotor_vel))
    attitude = controls.attitude
    cmd = attitude.cmd.at[..., 3].set(jnp.where(stopped, 0, attitude.cmd[..., 3]))
    return data.replace(controls=controls.replace(attitude=attitude.replace(cmd=cmd)))


def insert_stop_stage(sim: Sim) -> Sim:
    """Add the stop stage to the step pipeline with all drones stopped.

    Args:
        sim: The simulation.

    Returns:
        The simulation with the stage.
    """
    stop = StopData(stopped=jnp.ones((1, sim.n_drones), bool, device=sim.device))
    sim.data = sim.data.replace(plugins=sim.data.plugins | {STOP_KEY: stop})
    stage = partial(stop_stage, dynamics=sim.dynamics)
    insert_fn_after(sim.step_pipeline, "clip_rotor_vel_cmd", stage, name=STOP_KEY)
    sim.build_default_data()
    sim.build_step_fn()
    return sim


def stop_drones(sim: Sim, stopped: np.ndarray):
    """Stop the flagged drones and run all others.

    Args:
        sim: The simulation with the stop stage.
        stopped: Stop flags of shape (n_drones,).
    """
    stop = sim.data.plugins[STOP_KEY].replace(stopped=jnp.asarray(stopped[None], device=sim.device))
    sim.data = sim.data.replace(plugins=sim.data.plugins | {STOP_KEY: stop})
