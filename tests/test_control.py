import jax
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim

from crazyflow_itl.control import CONTROL_KEY, command_drones, control_stage, insert_control_stage
from crazyflow_itl.stop import insert_stop_stage, stop_drones

MODES = [Control.state, Control.attitude, Control.body_rate]


def running_sim(n_drones: int) -> Sim:
    """Create a sim with the stop and control stages and all motors on."""
    sim = insert_control_stage(insert_stop_stage(Sim(n_drones=n_drones, control="force_torque")))
    stop_drones(sim, np.zeros(n_drones, bool))
    return sim


def hover_thrust(sim: Sim) -> float:
    return float(sim.data.params.mass[0]) * np.linalg.norm(sim.data.params.gravity_vec)


def state_command(position) -> np.ndarray:
    """Return a state command to hold a position at zero yaw."""
    return np.array([*position, *[0.0] * 6, 0, 0, 0, 1, 0, 0, 0])


def test_modes_match_crazyflow():
    sim = running_sim(3)
    thrust = hover_thrust(sim)
    commands = np.stack(
        [
            state_command(sim.data.states.pos[0, 0] + np.array([0.2, 0, 0.5])),
            [0.1, -0.1, 0.3, 1.1 * thrust, *[0] * 8, 1, 0, 0, 0],
            [0.5, -0.2, 1.0, 1.2 * thrust, *[0] * 8, 1, 0, 0, 0],
        ]
    )
    # Each drone flies its command in a Crazyflow sim of its own mode as a reference
    references = [Sim(control=mode) for mode in MODES]
    for index, reference in enumerate(references):
        pos = sim.data.states.pos[:, [index]]
        reference.data = reference.data.replace(states=reference.data.states.replace(pos=pos))
    references[0].state_control(commands[None, [0]])
    references[1].attitude_control(commands[None, [1], :4])
    references[2].body_rate_control(commands[None, [2], :4])
    command_drones(sim, MODES, commands)
    sim.step(sim.freq)
    for index, reference in enumerate(references):
        reference.step(sim.freq)
        position = np.asarray(sim.data.states.pos[0, index])
        np.testing.assert_allclose(position, reference.data.states.pos[0, 0], atol=1e-5)


def test_switch_from_state_to_attitude_keeps_thrust():
    sim = running_sim(1)
    goal = sim.data.states.pos[0, 0] + np.array([0, 0, 0.5])
    command_drones(sim, [Control.state], state_command(goal)[None])
    sim.step(2 * sim.freq)
    sim.step()
    compiled = sim._step._cache_size()
    control = sim.data.plugins[CONTROL_KEY]
    force_torque = np.asarray(sim.data.controls.force_torque.cmd[0, 0])
    # The attitude command continues the last output of the state controller
    command = np.zeros((1, 16))
    command[0, :4] = control.attitude.cmd[0, 0]
    command_drones(sim, [Control.attitude], command)
    sim.step()
    control = sim.data.plugins[CONTROL_KEY]
    assert not control.state.pos_err_i.any()
    assert control.attitude.r_int_error.any()
    switched = np.asarray(sim.data.controls.force_torque.cmd[0, 0])
    assert abs(switched[0] - force_torque[0]) < 1e-3 * force_torque[0]
    assert sim._step._cache_size() == compiled, "Switching the mode recompiled the step function"


def test_switch_and_stop_reset_integrators():
    sim = running_sim(1)
    # A goal away from the drone winds up the integrators of the state and attitude controllers
    command_drones(sim, [Control.state], state_command([1, 1, 1])[None])
    sim.step(sim.freq // 2)
    control = sim.data.plugins[CONTROL_KEY]
    assert control.state.pos_err_i.any() and control.attitude.r_int_error.any()
    command = np.zeros((1, 16))
    command[0, :4] = [0.5, 0, 0, hover_thrust(sim)]
    command_drones(sim, [Control.body_rate], command)
    sim.step(sim.freq // 2)
    control = sim.data.plugins[CONTROL_KEY]
    assert not control.state.pos_err_i.any() and not control.attitude.r_int_error.any()
    assert control.body_rate.r_int_error.any()
    stop_drones(sim, np.ones(1, bool))
    sim.step()
    control = sim.data.plugins[CONTROL_KEY]
    assert not control.body_rate.r_int_error.any() and not control.body_rate.prev_ang_vel.any()


def test_control_stage_is_jitted():
    jax.make_jaxpr(control_stage)(running_sim(2).data)
