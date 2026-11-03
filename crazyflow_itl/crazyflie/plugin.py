import socket

import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim
from scipy.spatial.transform import Rotation

from crazyflow_itl.control import CONTROL_KEY, command_drones, insert_control_stage
from crazyflow_itl.crazyflie import crtp
from crazyflow_itl.crazyflie.commander import Commander
from crazyflow_itl.crazyflie.link import Link
from crazyflow_itl.stop import insert_stop_stage, stop_drones


class CrazyfliePlugin:
    """Serve every drone as a Crazyflie over CRTP on UDP."""

    def __init__(self, base_port: int = 19850, host: str = "127.0.0.1", freq: int = 100):
        """Create a Crazyflie plugin.

        Args:
            base_port: UDP port of the first drone, counting up per drone.
            host: Bind address.
            freq: Update rate in Hz.
        """
        self.base_port = base_port
        self.host = host
        self.freq = freq
        self.drone = None  # Set in setup
        self.links = []  # Set in setup
        self.commanders = []  # Set in setup
        self.peers = []  # Set in setup
        self.sockets = []  # Set in setup
        self.modes = []  # Set in setup
        self.commands = None  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Bind one port per drone, switch all motors off and return the sim.

        The sim must use force and torque control with first principles dynamics.
        """
        sim = insert_control_stage(insert_stop_stage(sim))
        self.drone = sim.drone
        self.links = [Link() for _ in range(sim.n_drones)]
        self.commanders = [Commander(self.drone, link.params) for link in self.links]
        self.peers = [None] * sim.n_drones
        self.sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in self.links]
        for index, sock in enumerate(self.sockets):
            sock.bind((self.host, self.base_port + index))
            sock.setblocking(False)
        self.modes = [Control.state] * sim.n_drones
        self.commands = np.array(sim.data.plugins[CONTROL_KEY].command[0])
        return sim

    def step(self, sim: Sim) -> Sim:
        """Answer the pending datagrams, stream log data, apply the setpoints and return the sim."""
        t = int(sim.data.core.steps[0, 0]) / sim.freq
        states = sim.data.states
        state = np.concatenate([states.pos, states.vel, states.quat, states.ang_vel], axis=-1)[0]
        # The firmware logs its legacy attitude in degrees with the pitch negated
        rpy = Rotation.from_quat(state[:, 6:10]).as_euler("xyz", degrees=True) * [1, -1, 1]
        stopped = np.zeros(sim.n_drones, bool)
        for index, sock in enumerate(self.sockets):
            while True:  # Respond to all pending commands/messages from the drone client
                try:
                    datagram, self.peers[index] = sock.recvfrom(64)
                except BlockingIOError:
                    break
                for reply in self._receive(index, datagram, t, state[index]):
                    sock.sendto(reply, self.peers[index])
            # The log table lists position, velocity, legacy Euler angles and quaternion in order
            values = np.concatenate([state[index, :6], rpy[index], state[index, 6:10]])
            # Send log data to the client if requested
            for datagram in self.links[index].log_data(
                t, dict(zip((variable.name for variable in Link.LOG_TOC), values))
            ):
                sock.sendto(datagram, self.peers[index])
            # Save the setpoint for the drone or stop it
            setpoint = self.commanders[index].setpoint(t, state[index])
            if setpoint is None:
                stopped[index] = True
            else:
                self.modes[index] = setpoint.control
                self.commands[index, : len(setpoint.cmd)] = setpoint.cmd
        stop_drones(sim, stopped)
        command_drones(sim, self.modes, self.commands)
        return sim

    def close(self):
        """Close the ports."""
        for sock in self.sockets:
            sock.close()

    def _receive(self, index: int, datagram: bytes, t: float, state: np.ndarray) -> list[bytes]:
        """Handle one datagram from the client of a drone and return the replies."""
        match datagram:
            case b"\xff":  # Link probe
                return [datagram]
            case b"\xff\xfe\x02":  # Power off
                return []
            case b"\xff\xfe\x03":  # Power on, which reboots the drone
                self.links[index] = Link()
                self.commanders[index] = Commander(self.drone, self.links[index].params)
                return []
        port, channel, payload = crtp.decode(datagram)
        link_ports = (crtp.Port.PARAM, crtp.Port.MEM, crtp.Port.LOG)
        if port in link_ports or (port, channel) == (crtp.Port.PLATFORM, 1):
            return self.links[index].handle(port, channel, payload, t)
        return self.commanders[index].handle(port, channel, payload, t, state)
