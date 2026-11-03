import os

import pytest
from crazyflow.sim import Sim

from crazyflow_itl import RenderPlugin, run


@pytest.mark.skipif("DISPLAY" not in os.environ, reason="The viewer needs a display")
def test_render_opens_and_closes_viewer():
    sim = Sim()
    plugin = RenderPlugin()
    run(sim, [plugin], 0.1)
    assert sim.viewer is None
