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


def stop_stage(data: SimData) -> SimData:
    """Zero the rotor velocities of stopped drones.

    Args:
        data: Sim data with the stop data.

    Returns:
        The updated sim data.
    """
    stopped = data.plugins[STOP_KEY].stopped
    rotor_vel = jnp.where(stopped[..., None], 0, data.controls.rotor_vel)
    return data.replace(controls=data.controls.replace(rotor_vel=rotor_vel))


def insert_stop_stage(sim: Sim) -> Sim:
    """Add the stop stage to the step pipeline with all drones stopped.

    The sim must use first principles dynamics.

    Args:
        sim: The simulation.

    Returns:
        The simulation with the stage.
    """
    assert sim.dynamics == Dynamics.first_principles
    stop = StopData(stopped=jnp.ones((1, sim.n_drones), bool, device=sim.device))
    sim.data = sim.data.replace(plugins=sim.data.plugins | {STOP_KEY: stop})
    insert_fn_after(sim.step_pipeline, "clip_rotor_vel_cmd", stop_stage, name=STOP_KEY)
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
