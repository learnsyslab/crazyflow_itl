import subprocess
import sys

import jax
import numpy as np
import pytest
from crazyflow.sim import Sim
from sensor_msgs.msg import Image

from crazyflow_itl import run
from crazyflow_itl.ros import drone_names


def write_scene(path):
    import splax

    count = 64
    rng = np.random.default_rng(0)
    means = rng.uniform(-0.5, 0.5, (count, 3)).astype(np.float32)
    log_scales = np.full((count, 3), np.log(0.05), np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (count, 1))
    sh_colors = splax.io.rgb_to_sh(rng.uniform(0.2, 0.8, (count, 3)))[:, None].astype(np.float32)
    logit_opacities = np.full((count,), 2.0, np.float32)
    splax.io.write_ply(path, means, log_scales, quats, sh_colors, logit_opacities)


@pytest.mark.skipif(
    all(device.platform != "gpu" for device in jax.devices()), reason="splat rendering needs a GPU"
)
def test_splat_images_published(node, listener, tmp_path):
    from crazyflow_itl.ros.splat_camera import SplatCameraPlugin

    scene = tmp_path / "scene.ply"
    write_scene(scene)
    sim = Sim(n_drones=2, device="gpu")
    names = drone_names(sim)
    for name in names:
        listener.listen(Image, f"/crazyflow/{name}/image")
    plugin = SplatCameraPlugin(node, scene)
    run(sim, [plugin, listener], duration=0.1)
    listener.collect()
    for name in names:
        messages = listener.messages[f"/crazyflow/{name}/image"]
        assert messages
        assert messages[0].encoding == "rgb8"
        assert (messages[0].width, messages[0].height) == (640, 480)
        assert messages[0].header.frame_id == name
        assert len(messages[0].data) == 640 * 480 * 3


def test_plugin_import_is_lazy():
    code = (
        "import sys, crazyflow_itl.ros\n"
        "assert 'splax' not in sys.modules\n"
        "assert 'crazyflow.sim.splat' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
