import struct
import zlib
from collections import namedtuple
from typing import NamedTuple

from crazyflow_itl.crazyflie import crtp

# Variables are log table ids in append order. The period is in ms. Started is the ms tick of
# the start or of the last emission, and None while the block is stopped.
Block = namedtuple("Block", ("variables", "period", "started"))


class Variable(NamedTuple):
    """Describe one entry of the log or param table.

    Attributes:
        name: Full name in the form group.name.
        type: Firmware type byte of the table item reply.
        format: Struct format of the value.
        default: Default value of a param, None for log variables.
    """

    name: str
    type: int
    format: str
    default: int | float | None = None


def _toc_item(toc: tuple[Variable, ...], payload: bytes) -> bytes:
    """Answer a table item request."""
    command, index = struct.unpack("<BH", payload)
    assert command == 2, f"Unsupported table command {command}"  # Table item
    variable = toc[index]
    group, name = variable.name.split(".")
    return struct.pack("<BHB", command, index, variable.type) + f"{group}\0{name}\0".encode()


class Link:
    """Answer the platform, memory, log and param requests of a Crazyflie client."""

    # Log limits of the firmware, with the block size in bytes
    MAX_BLOCKS = 16
    MAX_OPS = 128
    MAX_BLOCK_SIZE = 26

    LOG_TOC = tuple(
        Variable(f"stateEstimate.{name}", 7, "f")
        for name in (*"xyz", "vx", "vy", "vz", "roll", "pitch", "yaw", "qx", "qy", "qz", "qw")
    )
    PARAM_TOC = (
        Variable("stabilizer.estimator", 0x08, "B", 2),
        Variable("stabilizer.controller", 0x08, "B", 2),
        Variable("supervisor.tmblChckEn", 0x18, "B", 1),
        Variable("flightmode.stabModeRoll", 0x08, "B", 1),
        Variable("flightmode.stabModePitch", 0x08, "B", 1),
        Variable("flightmode.stabModeYaw", 0x08, "B", 0),
        Variable("kalman.resetEstimation", 0x08, "B", 0),
        Variable("kalman.initialX", 0x06, "f", 0.0),
        Variable("kalman.initialY", 0x06, "f", 0.0),
        Variable("kalman.initialZ", 0x06, "f", 0.0),
        Variable("kalman.initialYaw", 0x06, "f", 0.0),
        Variable("commander.enHighLevel", 0x08, "B", 0),
    )

    def __init__(self):
        """Create a link with default params and no log blocks."""
        self.params = {variable.name: variable.default for variable in self.PARAM_TOC}
        self.blocks = {}

    def handle(self, port: int, channel: int, payload: bytes, t: float) -> list[bytes]:
        """Answer one request from the client.

        Args:
            port: CRTP port.
            channel: CRTP channel.
            payload: Request data.
            t: Sim time in seconds.

        Returns:
            The reply datagrams.
        """
        match port, channel:
            case crtp.Port.PLATFORM, 1 if payload == b"\x00":  # Protocol version
                reply = struct.pack("<Bi", 0, 12)
            case crtp.Port.MEM, 0 if payload == b"\x01":  # Memory count
                reply = b"\x01\x00"
            case crtp.Port.LOG, 0 if payload == b"\x03":  # Table info
                reply = self._log_info()
            case crtp.Port.LOG, 0:
                reply = _toc_item(self.LOG_TOC, payload)
            case crtp.Port.LOG, 1:
                reply = self._log_control(payload, t)
            case crtp.Port.PARAM, 0 if payload == b"\x03":  # Table info
                reply = struct.pack(
                    "<BHI", 3, len(self.PARAM_TOC), self.crc(crtp.Port.PARAM, self.PARAM_TOC)
                )
            case crtp.Port.PARAM, 0:
                reply = _toc_item(self.PARAM_TOC, payload)
            case crtp.Port.PARAM, 1:
                reply = self._param_read(payload)
            case crtp.Port.PARAM, 2:
                reply = self._param_write(payload)
            case _:
                raise NotImplementedError(f"Unsupported request on port {port} channel {channel}")
        return [crtp.encode(port, channel, reply)]

    def log_data(self, t: float, values: dict[str, float]) -> list[bytes]:
        """Return the log data of all started blocks that are due.

        Args:
            t: Sim time in seconds.
            values: Log variable values by full name.

        Returns:
            One datagram per due block.
        """
        ms = round(t * 1000)
        datagrams = []
        for block, (variables, period, started) in self.blocks.items():
            if started is None or ms // period <= started // period:
                continue
            self.blocks[block] = self.blocks[block]._replace(started=ms)
            data = struct.pack(
                self._log_format(variables),
                *(values[self.LOG_TOC[variable].name] for variable in variables),
            )
            payload = bytes([block]) + (ms & 0xFFFFFF).to_bytes(3, "little") + data
            datagrams.append(crtp.encode(crtp.Port.LOG, 2, payload))
        return datagrams

    @staticmethod
    def crc(port: int, toc: tuple[Variable, ...]) -> int:
        """Compute the table checksum for the table cache of the client.

        Args:
            port: CRTP port of the table.
            toc: Table entries in id order.

        Returns:
            A stable 32 bit checksum.
        """
        return zlib.crc32(bytes([port]) + repr(toc).encode())

    def _log_format(self, variables: tuple[int, ...]) -> str:
        """Return the struct format of the values of a log block."""
        return "<" + "".join(self.LOG_TOC[variable].format for variable in variables)

    def _log_info(self) -> bytes:
        """Answer a log table info request with the table size, checksum and log limits."""
        return struct.pack(
            "<BHIBB",
            3,
            len(self.LOG_TOC),
            self.crc(crtp.Port.LOG, self.LOG_TOC),
            self.MAX_BLOCKS,
            self.MAX_OPS,
        )

    def _log_control(self, payload: bytes, t: float) -> bytes:
        """Apply a log control command and return the reply."""
        # The reply echoes the command and the block id with a zero error code
        reply = payload[:2].ljust(2, b"\0") + b"\0"
        command = payload[0]
        if command == 5:  # Reset
            self.blocks = {}
            return reply
        assert command in (2, 3, 4, 6, 7), f"Unsupported log command {command}"
        block = payload[1]
        if command == 6:  # Create
            assert block not in self.blocks, f"Log block {block} exists"
            assert len(self.blocks) < self.MAX_BLOCKS, f"More than {self.MAX_BLOCKS} log blocks"
            self.blocks[block] = Block((), None, None)
            return reply
        assert block in self.blocks, f"Log block {block} does not exist"
        if command == 7:  # Append
            *_, variable = struct.unpack("<BBBH", payload)
            variables = self.blocks[block].variables + (variable,)
            size = struct.calcsize(self._log_format(variables))
            assert size <= self.MAX_BLOCK_SIZE, (
                f"Log block of {size} bytes exceeds {self.MAX_BLOCK_SIZE}"
            )
            self.blocks[block] = self.blocks[block]._replace(variables=variables)
        elif command == 3:  # Start
            assert payload[2] > 0, "Single shot log blocks are not supported"
            self.blocks[block] = self.blocks[block]._replace(
                period=payload[2] * 10, started=round(t * 1000)
            )
        elif command == 4:  # Stop
            self.blocks[block] = self.blocks[block]._replace(started=None)
        else:  # Delete
            del self.blocks[block]
        return reply

    def _param_read(self, payload: bytes) -> bytes:
        """Answer a param read request with the current value."""
        (index,) = struct.unpack("<H", payload)
        variable = self.PARAM_TOC[index]
        return struct.pack(f"<HB{variable.format}", index, 0, self.params[variable.name])

    def _param_write(self, payload: bytes) -> bytes:
        """Store the value of a param write request and echo the request."""
        (index,) = struct.unpack_from("<H", payload)
        variable = self.PARAM_TOC[index]
        (self.params[variable.name],) = struct.unpack(f"<{variable.format}", payload[2:])
        return payload
