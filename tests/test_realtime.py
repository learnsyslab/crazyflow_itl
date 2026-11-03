import logging
import time

from crazyflow.sim import Sim

from crazyflow_itl import RealTimePlugin, run


class SlowPlugin:
    freq = 100

    def setup(self, sim):
        return sim

    def step(self, sim):
        time.sleep(0.02)
        return sim

    def close(self):
        pass


def warm_sim():
    """Create a sim whose step function of five steps is compiled."""
    sim = Sim()
    sim.step(5)
    return sim


def advance(plugin, sim, chunks):
    """Alternate plugin steps and physics chunks of five steps like the run loop does."""
    for _ in range(chunks):
        plugin.step(sim)
        sim.step(5)


def test_realtime_paces_wall_time():
    plugin, sim = RealTimePlugin(), warm_sim()
    plugin.setup(sim)
    advance(plugin, sim, 1)
    start = time.perf_counter()
    advance(plugin, sim, 11)
    assert 0.09 <= time.perf_counter() - start <= 0.15


def test_realtime_warns_when_behind(caplog):
    plugin, sim = RealTimePlugin(warn_lag=0.01), warm_sim()
    plugin.setup(sim)
    advance(plugin, sim, 2)
    time.sleep(0.05)
    advance(plugin, sim, 1)
    assert "lags real time" in caplog.text


def test_realtime_rate_limits_warnings(caplog):
    plugin, sim = RealTimePlugin(warn_lag=0.01, warn_interval=5.0), warm_sim()
    plugin.setup(sim)
    advance(plugin, sim, 2)
    time.sleep(0.05)
    advance(plugin, sim, 2)
    assert caplog.text.count("lags real time") == 1


def test_realtime_reset_restarts_pacing(caplog):
    plugin, sim = RealTimePlugin(warn_lag=0.01), warm_sim()
    plugin.setup(sim)
    advance(plugin, sim, 2)
    time.sleep(0.05)
    plugin.reset()
    advance(plugin, sim, 2)
    assert "lags real time" not in caplog.text


def test_run_compiles_before_pacing(caplog):
    run(Sim(), [RealTimePlugin(warn_lag=0.01)], duration=0.2)
    assert "lags real time" not in caplog.text


def test_realtime_never_skips_steps(caplog):
    caplog.set_level(logging.WARNING)
    sim = Sim()
    run(sim, [RealTimePlugin(), SlowPlugin()], duration=0.1)
    assert int(sim.data.core.steps[0, 0]) == 50
