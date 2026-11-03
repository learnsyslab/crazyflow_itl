import struct

import numpy as np
import pytest

from crazyflow_itl.crazyflie import crtp
from crazyflow_itl.crazyflie.link import Link

LOG_NAMES = [variable.name for variable in Link.LOG_TOC]
PARAM_NAMES = [variable.name for variable in Link.PARAM_TOC]
VALUES = {name: 0.0 for name in LOG_NAMES} | {"stateEstimate.x": 1.5, "stateEstimate.z": -0.25}


def request(link, port, channel, payload, t=0.0):
    """Send one request and return the payload of its single reply on the same port and channel."""
    (reply,) = link.handle(port, channel, payload, t)
    assert crtp.decode(reply)[:2] == (port, channel)
    return crtp.decode(reply)[2]


def append(link, block, name):
    """Append a log variable to a block and return the reply payload."""
    return request(link, 5, 1, struct.pack("<BBBH", 7, block, 7, LOG_NAMES.index(name)))


def start_block(link, names, t=0.0):
    """Create log block 0 with the given variables and start it with a period of 10 ms."""
    assert request(link, 5, 1, b"\x06\x00") == b"\x06\x00\x00"
    for name in names:
        assert append(link, 0, name) == b"\x07\x00\x00"
    assert request(link, 5, 1, b"\x03\x00\x01", t) == b"\x03\x00\x00"


def test_protocol_version():
    assert Link().handle(13, 1, b"\x00", 0.0) == [b"\xd1\x00\x0c\x00\x00\x00"]


def test_memory_count_zero():
    assert Link().handle(4, 0, b"\x01", 0.0) == [b"\x40\x01\x00"]


def test_log_and_param_crc_differ():
    assert Link.crc(5, Link.LOG_TOC) != Link.crc(2, Link.PARAM_TOC)


def test_log_toc_items():
    link = Link()
    info = struct.pack("<BHIBB", 3, len(Link.LOG_TOC), Link.crc(5, Link.LOG_TOC), 16, 128)
    assert request(link, 5, 0, b"\x03") == info
    for index, variable in enumerate(Link.LOG_TOC):
        data = request(link, 5, 0, struct.pack("<BH", 2, index))
        assert struct.unpack_from("<BHB", data) == (2, index, 7)
        assert data[4:] == variable.name.replace(".", "\0").encode() + b"\0"


def test_param_toc_items():
    link = Link()
    info = struct.pack("<BHI", 3, len(Link.PARAM_TOC), Link.crc(2, Link.PARAM_TOC))
    assert request(link, 2, 0, b"\x03") == info
    for index, variable in enumerate(Link.PARAM_TOC):
        data = request(link, 2, 0, struct.pack("<BH", 2, index))
        assert struct.unpack_from("<BHB", data) == (2, index, variable.type)
        assert data[4:] == variable.name.replace(".", "\0").encode() + b"\0"
    assert Link.PARAM_TOC[PARAM_NAMES.index("supervisor.tmblChckEn")].type == 0x18


def test_param_write_echo_and_store():
    link = Link()
    payload = struct.pack("<Hf", PARAM_NAMES.index("kalman.initialX"), 1.5)
    assert request(link, 2, 2, payload) == payload
    assert link.params["kalman.initialX"] == 1.5


def test_param_read_returns_value():
    index = PARAM_NAMES.index("stabilizer.controller")
    assert request(Link(), 2, 1, struct.pack("<H", index)) == struct.pack("<HB", index, 0) + b"\x02"


def test_param_unknown_id_fails():
    link = Link()
    with pytest.raises(IndexError):
        link.handle(2, 1, struct.pack("<H", 999), 0.0)
    with pytest.raises(IndexError):
        link.handle(2, 2, struct.pack("<HB", 999, 1), 0.0)


def test_log_block_streams_at_period():
    link = Link()
    start_block(link, ["stateEstimate.x", "stateEstimate.z"])
    packets = [packet for k in range(51) for packet in link.log_data(k / 500, VALUES)]
    assert len(packets) == 10
    for packet in packets:
        port, channel, data = crtp.decode(packet)
        assert (port, channel, data[0]) == (5, 2, 0)
        assert struct.unpack("<ff", data[4:]) == (1.5, -0.25)
    assert int.from_bytes(crtp.decode(packets[-1])[2][1:4], "little") == 100
    assert link.log_data(0.1, VALUES) == []


def test_log_block_float32_time():
    link = Link()
    start_block(link, ["stateEstimate.x"])
    times = [float(np.float32(k / 500)) for k in range(251)]
    packets = [packet for t in times for packet in link.log_data(t, VALUES)]
    stamps = [int.from_bytes(crtp.decode(packet)[2][1:4], "little") for packet in packets]
    assert stamps == list(range(10, 501, 10))


def test_log_stop_and_delete_end_streaming():
    link = Link()
    start_block(link, ["stateEstimate.x"])
    assert len(link.log_data(0.01, VALUES)) == 1
    assert request(link, 5, 1, b"\x04\x00") == b"\x04\x00\x00"
    assert not any(link.log_data(k / 500, VALUES) for k in range(6, 51))
    assert request(link, 5, 1, b"\x03\x00\x01", 0.1) == b"\x03\x00\x00"
    assert len(link.log_data(0.11, VALUES)) == 1
    assert request(link, 5, 1, b"\x02\x00") == b"\x02\x00\x00"
    assert not any(link.log_data(k / 500, VALUES) for k in range(56, 101))
    with pytest.raises(AssertionError):
        link.handle(5, 1, b"\x03\x00\x01", 0.0)


def test_log_block_too_large_fails():
    link = Link()
    assert request(link, 5, 1, b"\x06\x00") == b"\x06\x00\x00"
    for name in LOG_NAMES[:6]:
        assert append(link, 0, name) == b"\x07\x00\x00"
    with pytest.raises(AssertionError):
        append(link, 0, LOG_NAMES[6])


def test_invalid_log_control_fails():
    link = Link()
    for block in range(16):
        assert request(link, 5, 1, bytes([6, block])) == bytes([6, block, 0])
    for payload in (b"\x06\x00", b"\x06\x10", b"\x01\x00"):
        with pytest.raises(AssertionError):
            link.handle(5, 1, payload, 0.0)
    with pytest.raises(AssertionError):
        append(link, 16, "stateEstimate.x")
    with pytest.raises(IndexError):
        link.handle(5, 1, struct.pack("<BBBH", 7, 0, 7, 999), 0.0)


def test_log_reset_clears_blocks():
    link = Link()
    start_block(link, ["stateEstimate.x"])
    assert request(link, 5, 1, b"\x05") == b"\x05\x00\x00"
    assert link.log_data(0.01, VALUES) == []
    assert request(link, 5, 1, b"\x06\x00") == b"\x06\x00\x00"
