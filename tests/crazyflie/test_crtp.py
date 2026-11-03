import numpy as np

from crazyflow_itl.crazyflie.crtp import decode, encode, quat_decompress


def compress(quat: np.ndarray) -> int:
    """Compress a quaternion like the full state setpoint of crazyflie-lib-rs."""
    quat = quat / np.linalg.norm(quat)
    largest = int(np.argmax(np.abs(quat)))
    negate = bool(quat[largest] < 0)
    comp = largest
    for index, component in enumerate(quat.tolist()):
        if index != largest:
            magnitude = int(511 * abs(component) / np.sqrt(0.5) + 0.5)
            comp = comp << 10 | ((component < 0) ^ negate) << 9 | magnitude
    return comp


def test_header_roundtrip():
    assert encode(13, 1, b"\x00") == b"\xd1\x00"
    assert decode(b"\xd1\x00") == (13, 1, b"\x00")


def test_quat_decompress_matches_lib_encoding():
    quats = np.random.default_rng(0).normal(size=(100, 4))
    for quat in quats / np.linalg.norm(quats, axis=1, keepdims=True):
        decompressed = quat_decompress(compress(quat))
        assert min(np.abs(decompressed - quat).max(), np.abs(decompressed + quat).max()) < 2e-3
