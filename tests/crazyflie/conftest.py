from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def pool():
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield pool
