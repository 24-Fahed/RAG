"""Process-wide model lifecycle and FIFO inference scheduling.

Run one worker process per deployment. These queues are not distributed locks.
"""
from contextlib import contextmanager
from functools import wraps
from threading import Condition, Lock


class FIFOQueue:
    def __init__(self):
        self._condition = Condition()
        self._next = 0
        self._serving = 0

    @contextmanager
    def hold(self):
        with self._condition:
            ticket = self._next
            self._next += 1
            self._condition.wait_for(lambda: ticket == self._serving)
        try:
            yield
        finally:
            with self._condition:
                self._serving += 1
                self._condition.notify_all()


class ManagedModel:
    """Serialize calls without forcing model adapters to implement locking."""
    def __init__(self, model):
        self._model = model
        self._queue = FIFOQueue()

    def __getattr__(self, name):
        attribute = getattr(self._model, name)
        if not callable(attribute):
            return attribute

        @wraps(attribute)
        def call(*args, **kwargs):
            with self._queue.hold():
                return attribute(*args, **kwargs)
        return call


class ModelRegistry:
    def __init__(self):
        self._initialization = FIFOQueue()
        self._cache_lock = Lock()
        self._models = {}

    def get(self, key, factory):
        # A loaded model remains callable while another model is loading.
        with self._cache_lock:
            if key in self._models:
                return self._models[key]
        with self._initialization.hold():
            with self._cache_lock:
                if key in self._models:
                    return self._models[key]
            model = ManagedModel(factory())
            with self._cache_lock:
                self._models[key] = model
            return model


registry = ModelRegistry()
