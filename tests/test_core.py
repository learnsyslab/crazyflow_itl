import pytest
from crazyflow.sim import Sim

from crazyflow_itl import run
from crazyflow_itl.exceptions import StopRun


class Recorder:
    def __init__(self, name="", freq=100, log=None, replacement=None):
        self.name, self.freq = name, freq
        self.log = [] if log is None else log
        self.replacement = replacement
        self.steps, self.setup_sims, self.step_sims = [], [], []

    def setup(self, sim):
        self.setup_sims.append(sim)
        self.log.append(("setup", self.name))
        return sim if self.replacement is None else self.replacement

    def step(self, sim):
        self.step_sims.append(sim)
        self.steps.append(int(sim.data.core.steps[0, 0]))
        self.log.append(("step", self.name))
        return sim

    def close(self):
        self.log.append(("close", self.name))


class Failing(Recorder):
    def __init__(self, *args, fail_in, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_in = fail_in

    def setup(self, sim):
        assert self.fail_in != "setup", "boom"
        return super().setup(sim)

    def step(self, sim):
        if self.fail_in == "step":
            raise RuntimeError("boom")
        return super().step(sim)


class Quitting(Recorder):
    def step(self, sim):
        if int(sim.data.core.steps[0, 0]) == 10:
            raise StopRun
        return super().step(sim)


def test_plugins_run_at_their_rate():
    a, b = Recorder(), Recorder(freq=50)
    run(Sim(n_drones=2), [a, b], duration=0.1)
    assert a.steps == list(range(0, 50, 5)) and b.steps == list(range(0, 50, 10))


def test_rate_must_divide_sim_freq():
    with pytest.raises(AssertionError):
        run(Sim(), [Recorder(freq=300)], duration=0.1)


def test_plugins_run_in_list_order():
    log = []
    a, b = Recorder("a", log=log), Recorder("b", freq=50, log=log)
    run(Sim(), [a, b], duration=0.1)
    steps = [entry for entry in log if entry[0] == "step"]
    assert steps[:4] == [("step", "a"), ("step", "b"), ("step", "a"), ("step", "a")]
    for index, entry in enumerate(steps):
        if entry == ("step", "b"):
            assert steps[index - 1] == ("step", "a")


def test_setup_returns_replace_sim():
    replacement = Sim()
    a, b = Recorder("a", replacement=replacement), Recorder("b")
    original = Sim()
    run(original, [a, b], duration=0.05)
    assert a.setup_sims == [original]
    assert b.setup_sims == [replacement]
    assert all(sim is replacement for sim in a.step_sims + b.step_sims)


def test_close_runs_in_reverse_order():
    log = []
    run(Sim(), [Recorder("a", log=log), Recorder("b", log=log)], duration=0.02)
    assert [name for kind, name in log if kind == "close"] == ["b", "a"]


def test_step_error_closes_plugins():
    log = []
    a, b = Recorder("a", log=log), Failing("b", log=log, fail_in="step")
    with pytest.raises(RuntimeError, match="boom"):
        run(Sim(), [a, b], duration=0.1)
    assert [name for kind, name in log if kind == "close"] == ["b", "a"]


def test_setup_error_closes_already_set_up():
    log = []
    a, b = Recorder("a", log=log), Failing("b", log=log, fail_in="setup")
    with pytest.raises(AssertionError, match="boom"):
        run(Sim(), [a, b], duration=0.1)
    assert [name for kind, name in log if kind == "close"] == ["a"]


def test_single_world():
    with pytest.raises(AssertionError):
        run(Sim(n_worlds=2), [], duration=0.1)


def test_plugins_step_before_physics():
    sim = Sim()
    plugin = Recorder()
    run(sim, [plugin], duration=0.02)
    assert plugin.steps == [0, 5]
    assert int(sim.data.core.steps[0, 0]) == 10


def test_stop_run_ends_run():
    log = []
    sim = run(Sim(), [Recorder("a", log=log), Quitting("b", log=log)])
    assert int(sim.data.core.steps[0, 0]) == 10
    assert [name for kind, name in log if kind == "close"] == ["b", "a"]
