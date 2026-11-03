"""Fly the drones of the Crazyflie sim with cflib2 through a takeoff, a square and a landing.

The client connects to every drone, arms it and flies relative moves with the high-level
commander. After every move it logs the estimated position of the drone and prints it. After the
landing it stops the motors and hands control back to the high-level commander.
"""

import asyncio

import fire
from cflib2 import Crazyflie, LinkContext

# Takeoff to 0.5 m, a square with 0.5 m sides and the landing, each relative to the last position
MOVES = [(0, 0, 0.5), (0.5, 0, 0), (0, 0.5, 0), (-0.5, 0, 0), (0, -0.5, 0), (0, 0, -0.5)]
POSITION = ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]


async def fly(uri: str, leg_duration: float):
    """Fly the drone at the URI through the moves."""
    cf = await Crazyflie.connect_from_uri(LinkContext(), uri)
    block = await cf.log().create_block()
    for name in POSITION:
        await block.add_variable(name)
    await cf.supervisor().send_arming_request(True)
    for x, y, z in MOVES:
        await cf.high_level_commander().go_to(x, y, z, 0.0, leg_duration, True, False, None)
        await asyncio.sleep(leg_duration)
        stream = await block.start(10)
        data = (await stream.next()).data
        block = await stream.stop()
        print(f"{uri} at {[round(data[name], 2) for name in POSITION]}")
    await cf.commander().send_stop_setpoint()
    await cf.commander().send_notify_setpoint_stop(0)
    await cf.disconnect()


async def fly_all(uris: list[str], leg_duration: float):
    """Fly all drones at once."""
    await asyncio.gather(*(fly(uri, leg_duration) for uri in uris))


def main(n_drones: int = 2, leg_duration: float = 2.0):
    uris = [f"udp://127.0.0.1:{19850 + i}" for i in range(n_drones)]
    asyncio.run(fly_all(uris, leg_duration))


if __name__ == "__main__":
    fire.Fire(main)
