from enum import IntEnum

import numpy as np


class Port(IntEnum):
    """CRTP ports of the firmware."""

    CONSOLE = 0
    PARAM = 2
    RPYT = 3
    MEM = 4
    LOG = 5
    LOCALIZATION = 6
    GENERIC = 7
    HLC = 8
    SUPERVISOR = 9
    PLATFORM = 13


def encode(port: int, channel: int, payload: bytes) -> bytes:
    """Build the datagram that carries one CRTP packet.

    Args:
        port: CRTP port in [0, 15].
        channel: CRTP channel in [0, 3].
        payload: Packet data of at most 30 bytes.

    Returns:
        The header byte followed by the payload.
    """
    assert len(payload) <= 30, f"CRTP payload of {len(payload)} bytes exceeds 30 bytes"
    return bytes([port << 4 | channel]) + payload


def decode(datagram: bytes) -> tuple[int, int, bytes]:
    """Split a datagram into the parts of its CRTP packet.

    Args:
        datagram: The header byte followed by the payload.

    Returns:
        The port, the channel and the payload.
    """
    return datagram[0] >> 4, datagram[0] & 3, datagram[1:]


def quat_decompress(comp: int) -> np.ndarray:
    """Decompress a quaternion from its 32 bit encoding.

    Args:
        comp: Quaternion in the 32 bit compressed format of the firmware.

    Returns:
        Unit quaternion of shape (4,).
    """
    # Firmware reference: https://github.com/bitcraze/crazyflie-firmware/blob/b48aba18a675fe950a88d96586fd09e49d82a38b/src/modules/interface/quatcompress.h#L65
    largest = comp >> 30
    quat = np.zeros(4)
    for index in (j for j in (3, 2, 1, 0) if j != largest):
        quat[index] = (-1) ** ((comp >> 9) & 1) * np.sqrt(0.5) * (comp & 511) / 511
        comp >>= 10
    quat[largest] = np.sqrt(1 - np.sum(quat**2))
    return quat
