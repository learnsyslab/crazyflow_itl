import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor


@pytest.fixture
def node():
    rclpy.init()
    node = rclpy.create_node("crazyflow_itl_test")
    yield node
    node.destroy_node()
    rclpy.shutdown()


class Listener:
    """Collect messages on a separate node and act as a plugin that waits for discovery.

    As a plugin, it also records the states at its last step.
    """

    freq = 100

    def __init__(self, publisher_node):
        self.publisher_node = publisher_node
        self.node = rclpy.create_node("crazyflow_itl_listener")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.messages = {}

    def listen(self, message_type, topic):
        self.messages[topic] = []
        self.node.create_subscription(message_type, topic, self.messages[topic].append, 100)

    def setup(self, sim):
        for topic in self.messages:
            while self.publisher_node.count_subscribers(topic) == 0:
                self.executor.spin_once(timeout_sec=0.01)
        return sim

    def step(self, sim):
        self.states = sim.data.states
        return sim

    def close(self):
        pass

    def collect(self):
        """Spin until no more messages arrive."""
        previous, total = -1, 0
        while total != previous:
            previous = total
            self.executor.spin_once(timeout_sec=0.1)
            total = sum(map(len, self.messages.values()))


@pytest.fixture
def listener(node):
    listener = Listener(node)
    yield listener
    listener.node.destroy_node()
