import jax
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim

from crazyflow_itl.stop import insert_stop_stage, stop_drones, stop_stage


def stopped_sim() -> Sim:
    """Create a two drone sim whose first drone is stopped by the stage."""
    sim = insert_stop_stage(Sim(n_drones=2, control=Control.attitude))
    stop_drones(sim, np.array([True, False]))
    return sim


def test_stop_stage_zeroes_thrust():
    sim = stopped_sim()
    # More thrust than hover lifts every drone off the floor once its motors have spun up
    thrust = 1.5 * sim.data.params.mass[0] * np.linalg.norm(sim.data.params.gravity_vec)
    sim.attitude_control(np.tile([0, 0, 0, thrust], (1, 2, 1)))
    sim.step(sim.freq // 2)
    z = np.asarray(sim.data.states.pos[0, :, 2])
    assert z[0] < 0.001, "The stopped drone left the floor"
    assert z[1] > 0.05, "The drone that is not stopped did not take off"


def test_stop_stage_is_jitted():
    jax.make_jaxpr(stop_stage)(stopped_sim().data)
