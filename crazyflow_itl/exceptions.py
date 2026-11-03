class CrazyflowITLException(Exception):
    """Base class for all Crazyflow ITL exceptions."""


class StopRun(CrazyflowITLException):
    """Stop the run loop when a plugin raises it."""
