from crazyflow.sim import Sim
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from simulation_interfaces.action import SimulateSteps
from simulation_interfaces.msg import Result, SimulationState, SimulatorFeatures
from simulation_interfaces.srv import (
    GetSimulationState,
    GetSimulatorFeatures,
    SetSimulationState,
    StepSimulation,
)

from crazyflow_itl.exceptions import StopRun
from crazyflow_itl.realtime import RealTimePlugin
from crazyflow_itl.ros.spin import SpinPlugin


class SteppingPlugin:
    """Pause, play and step the sim through the ROS simulation interfaces.

    No other plugin or executor may spin the node.
    """

    def __init__(self, node: Node, freq: int = 100, paused: bool = True):
        """Create a stepping plugin.

        A step is one plugin period. Playing runs in real time.

        Args:
            node: Node whose callbacks are served.
            freq: Step rate in Hz.
            paused: Start paused.
        """
        self.node = node
        self.freq = freq
        self.state = SimulationState.STATE_PAUSED if paused else SimulationState.STATE_PLAYING
        self.spin = SpinPlugin(node)
        self.realtime = RealTimePlugin()
        self.remaining_steps = 0
        self.steps_done = Future()
        self.services = []  # Set in setup
        self.action_server = None  # Set in setup

    def setup(self, sim: Sim) -> Sim:
        """Create the services and the action server and return the sim."""
        # Step requests suspend until their steps have run, so other requests must not wait
        group = ReentrantCallbackGroup()
        self.services = [
            self.node.create_service(
                GetSimulatorFeatures,
                "~/get_simulator_features",
                self._get_simulator_features,
                callback_group=group,
            ),
            self.node.create_service(
                GetSimulationState,
                "~/get_simulation_state",
                self._get_simulation_state,
                callback_group=group,
            ),
            self.node.create_service(
                SetSimulationState,
                "~/set_simulation_state",
                self._set_simulation_state,
                callback_group=group,
            ),
            self.node.create_service(
                StepSimulation, "~/step_simulation", self._step_simulation, callback_group=group
            ),
        ]
        self.action_server = ActionServer(
            self.node, SimulateSteps, "~/simulate_steps", self._simulate_steps, callback_group=group
        )
        sim = self.realtime.setup(sim)
        return self.spin.setup(sim)

    def step(self, sim: Sim) -> Sim:
        """Serve requests until the sim may advance by one period and return the sim.

        Raises:
            StopRun: If a client quits the simulation.
        """
        self.spin.step(sim)
        while self.state == SimulationState.STATE_PAUSED and not self.remaining_steps:
            self.spin.executor.spin_once(timeout_sec=0.01)
        if self.state == SimulationState.STATE_QUITTING:
            raise StopRun
        if not self.remaining_steps:
            return self.realtime.step(sim)
        self.remaining_steps -= 1
        if not self.remaining_steps:
            self.steps_done.set_result(None)
            self.steps_done = Future()
        return sim

    async def _advance(self, steps: int):
        """Grant steps to the run loop and return once it has run them."""
        self.remaining_steps += steps
        await self.steps_done

    def _get_simulator_features(
        self, request: GetSimulatorFeatures.Request, response: GetSimulatorFeatures.Response
    ) -> GetSimulatorFeatures.Response:
        """Report the supported state and stepping features."""
        response.features.features = [
            SimulatorFeatures.SIMULATION_STATE_GETTING,
            SimulatorFeatures.SIMULATION_STATE_SETTING,
            SimulatorFeatures.SIMULATION_STATE_PAUSE,
            SimulatorFeatures.STEP_SIMULATION_SINGLE,
            SimulatorFeatures.STEP_SIMULATION_MULTIPLE,
            SimulatorFeatures.STEP_SIMULATION_ACTION,
        ]
        return response

    def _get_simulation_state(
        self, request: GetSimulationState.Request, response: GetSimulationState.Response
    ) -> GetSimulationState.Response:
        """Report the current simulation state."""
        response.state.state = self.state
        response.result.result = Result.RESULT_OK
        return response

    def _set_simulation_state(
        self, request: SetSimulationState.Request, response: SetSimulationState.Response
    ) -> SetSimulationState.Response:
        """Switch between playing, paused and quitting and report the outcome."""
        state = request.state.state
        if state == self.state:
            response.result.result = SetSimulationState.Response.ALREADY_IN_TARGET_STATE
        elif state in (
            SimulationState.STATE_PLAYING,
            SimulationState.STATE_PAUSED,
            SimulationState.STATE_QUITTING,
        ):
            if state == SimulationState.STATE_PLAYING:
                self.realtime.reset()
            self.state = state
            response.result.result = Result.RESULT_OK
        else:
            response.result.result = SetSimulationState.Response.STATE_TRANSITION_ERROR
            response.result.error_message = "Only playing, paused and quitting are supported"
        return response

    async def _step_simulation(
        self, request: StepSimulation.Request, response: StepSimulation.Response
    ) -> StepSimulation.Response:
        """Run the requested number of steps of the paused sim and report the outcome."""
        if self.state != SimulationState.STATE_PAUSED:
            response.result.result = Result.RESULT_OPERATION_FAILED
            response.result.error_message = "The simulation is not paused"
            return response
        if request.steps:
            await self._advance(request.steps)
        response.result.result = Result.RESULT_OK
        return response

    async def _simulate_steps(self, goal_handle: ServerGoalHandle) -> SimulateSteps.Result:
        """Run the goal number of steps of the paused sim with feedback after every step."""
        if self.state != SimulationState.STATE_PAUSED:
            goal_handle.abort()
            return SimulateSteps.Result(
                result=Result(
                    result=Result.RESULT_OPERATION_FAILED,
                    error_message="The simulation is not paused",
                )
            )
        steps = goal_handle.request.steps
        for completed in range(1, steps + 1):
            await self._advance(1)
            goal_handle.publish_feedback(
                SimulateSteps.Feedback(completed_steps=completed, remaining_steps=steps - completed)
            )
        goal_handle.succeed()
        return SimulateSteps.Result(result=Result(result=Result.RESULT_OK))

    def close(self):
        """Destroy the services and the action server and shut the executor down."""
        self.action_server.destroy()
        for service in self.services:
            self.node.destroy_service(service)
        self.spin.close()
