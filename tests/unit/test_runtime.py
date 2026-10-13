from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
import time
import pytest
from rag_langgraph.models.runtime import FIFOQueue, ModelRegistry


def test_initializations_serial_and_shared_instances():
    registry = ModelRegistry()
    active = maximum = calls = 0
    guard = Lock()
    start = Barrier(8)
    def factory():
        nonlocal active, maximum, calls
        with guard:
            active += 1
            calls += 1
            maximum = max(maximum, active)
        time.sleep(.01)
        with guard:
            active -= 1
        return object()
    def load(i):
        start.wait(timeout=3)
        return registry.get(i % 4, factory)
    with ThreadPoolExecutor(8) as pool:
        result = list(pool.map(load, range(8)))
    assert maximum == 1 and calls == 4
    assert all(result[i] is result[i + 4] for i in range(4))


def test_fifo_and_exception_release():
    queue = FIFOQueue()
    order = []
    def call(i):
        with queue.hold():
            order.append(i)
            if i == 1:
                raise ValueError('failure')
    with ThreadPoolExecutor(4) as pool:
        with queue.hold():
            futures = []
            for i in range(4):
                futures.append(pool.submit(call, i))
                deadline = time.monotonic() + 2
                while queue._next != i + 2:
                    assert time.monotonic() < deadline
                    time.sleep(.001)
        for i, future in enumerate(futures):
            if i == 1:
                with pytest.raises(ValueError): future.result(timeout=2)
            else:
                future.result(timeout=2)
    assert order == [0, 1, 2, 3]


def test_same_model_serial_different_models_parallel():
    registry = ModelRegistry()
    barrier = Barrier(2)
    class Model:
        def call(self):
            barrier.wait(timeout=2)
    a = registry.get('a', Model)
    b = registry.get('b', Model)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(a.call)
        second = pool.submit(b.call)
        first.result(timeout=3); second.result(timeout=3)
    active = maximum = 0
    class Serial:
        def call(self):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(.01)
            active -= 1
    model = registry.get('serial', Serial)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: model.call(), range(8)))
    assert maximum == 1


def test_load_failure_retries_and_cache_hit_does_not_wait_for_other_load():
    registry = ModelRegistry()
    def fail(): raise RuntimeError('load failed')
    with pytest.raises(RuntimeError): registry.get('a', fail)
    a = registry.get('a', object)
    entered, release = Event(), Event()
    def blocking():
        entered.set()
        assert release.wait(timeout=3)
        return object()
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(registry.get, 'b', blocking)
        assert entered.wait(timeout=2)
        try:
            assert pool.submit(registry.get, 'a', fail).result(timeout=1) is a
        finally:
            release.set()
        future.result(timeout=3)
