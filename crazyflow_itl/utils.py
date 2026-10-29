import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim


def default_commands(sim: Sim) -> np.ndarray:
    """Return the default command of the sim for every drone.

    Args:
        sim: The simulation.

    Returns:
        Commands of shape (n_drones, 16) in state control and (n_drones, 4) otherwise.
    """
    commands = np.zeros((sim.n_drones, 16 if sim.control == Control.state else 4))
    if sim.control == Control.state:
        commands[:, 12] = 1.0
    return commands


def apply_commands(sim: Sim, commands: np.ndarray):
    """Pass one command per drone to the control setter of the sim.

    Args:
        sim: The simulation.
        commands: Commands of shape (n_drones, 16) in state control and (n_drones, 4) otherwise.
    """
    cmd = jnp.asarray(commands[None], device=sim.device)  # Avoid recompilation on different devices
    match sim.control:
        case Control.state:
            sim.state_control(cmd)
        case Control.attitude:
            sim.attitude_control(cmd)
        case Control.body_rate:
            sim.body_rate_control(cmd)
        case Control.force_torque:
            sim.force_torque_control(cmd)
        case Control.rotor_vel:
            sim.rotor_vel_control(cmd)
