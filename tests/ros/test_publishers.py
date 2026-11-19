import time
from functools import partial

import numpy as np
import pytest
from crazyflow.sim import Sim
from geometry_msgs.msg import TwistStamped
from rclpy.parameter import Parameter
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

from crazyflow_itl import run
from crazyflow_itl.ros import ClockPlugin, SpinPlugin, TFPlugin, TwistPlugin, drone_names
from crazyflow_itl.ros.utils import stamp


def make_sim(**states):
    sim = Sim(n_drones=2)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(**{key: np.asarray(value) for key, value in states.items()})
    )
    sim.build_default_data()
    return sim


def seconds(time_message):
    return time_message.sec + time_message.nanosec * 1e-9


def test_drone_names():
    assert drone_names(Sim(n_drones=2)) == ["cf21B_500_0", "cf21B_500_1"]


def run_tf(node, listener, names):
    sim = make_sim(pos=[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]])
    listener.listen(TFMessage, "/tf")
    run(sim, [TFPlugin(node, names), listener], duration=0.2)
    listener.collect()
    last = {}
    for message in listener.messages["/tf"]:
        for transform in message.transforms:
            last[transform.child_frame_id] = transform
    return last, listener.states


def test_tf_publishes_pose(node, listener):
    names = ["cf21B_500_0", "cf21B_500_1"]
    last, states = run_tf(node, listener, names)
    for name, pos, quat in zip(names, states.pos[0], states.quat[0]):
        transform = last[name]
        translation, rotation = transform.transform.translation, transform.transform.rotation
        assert transform.header.frame_id == "world"
        assert [translation.x, translation.y, translation.z] == pytest.approx(pos.tolist())
        assert [rotation.x, rotation.y, rotation.z, rotation.w] == pytest.approx(quat.tolist())


def test_tf_custom_names(node, listener):
    last, _ = run_tf(node, listener, ["cf10", "cf52"])
    assert set(last) == {"cf10", "cf52"}


def test_tf_publishes_mocap_bodies(node, listener):
    sim = Sim()
    rotation = Rotation.from_euler("z", 90, degrees=True)
    body = sim.spec.worldbody.add_body(
        name="gate1", pos=[1.0, 2.0, 3.0], quat=rotation.as_quat(scalar_first=True), mocap=True
    )
    body.add_geom(size=[0.1, 0.1, 0.1])
    sim.build_mjx()
    listener.listen(TFMessage, "/tf")
    run(sim, [TFPlugin(node, mocap_names=["gate1"]), listener], duration=0.2)
    listener.collect()
    transform = listener.messages["/tf"][-1].transforms[-1]
    translation, quat = transform.transform.translation, transform.transform.rotation
    assert transform.child_frame_id == "gate1"
    assert [translation.x, translation.y, translation.z] == pytest.approx([1.0, 2.0, 3.0])
    assert [quat.x, quat.y, quat.z, quat.w] == pytest.approx(rotation.as_quat().tolist())


def test_twist_world_frame(node, listener):
    quat = Rotation.from_euler("z", 90, degrees=True).as_quat()
    sim = make_sim(
        quat=[[quat, [0.0, 0.0, 0.0, 1.0]]], ang_vel=[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    listener.listen(TwistStamped, "/crazyflow/cf21B_500_0/twist")
    plugin = TwistPlugin(node)
    sim = plugin.setup(sim)
    listener.setup(sim)
    plugin.step(sim)
    listener.collect()
    plugin.close()
    (message,) = listener.messages["/crazyflow/cf21B_500_0/twist"]
    angular = message.twist.angular
    assert [angular.x, angular.y, angular.z] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert message.header.frame_id == "world"


def test_clock_publishes_sim_time(node, listener):
    listener.listen(Clock, "/clock")
    run(make_sim(), [ClockPlugin(node), listener], duration=0.2)
    listener.collect()
    # The last tick publishes the time before the final physics step
    assert seconds(listener.messages["/clock"][-1].clock) == pytest.approx(0.19, abs=1e-6)


def test_stamp_follows_use_sim_time(node):
    sim = make_sim()
    sim.step(100)
    node.set_parameters([Parameter("use_sim_time", value=True)])
    assert stamp(node, sim) == Time(nanoseconds=200_000_000).to_msg()


def test_spin_plugin_drains_all_ready_callbacks(node):
    calls = []
    for index in range(5):
        node.create_timer(0.05, partial(calls.append, index))
    plugin = SpinPlugin(node)
    sim = plugin.setup(make_sim())
    time.sleep(0.1)
    plugin.step(sim)
    plugin.close()
    assert sorted(calls) == [0, 1, 2, 3, 4]
