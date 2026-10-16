"""Optional synchronized language-model forward profiling during generate()."""
from __future__ import annotations

import time


class DecoderTimer:
    def __init__(self, model, sync):
        self.model, self.sync = model, sync
        self.durations = []
        self.handles = []

    def _before(self, *_):
        self.sync()
        self.started = time.perf_counter()

    def _after(self, *_):
        self.sync()
        self.durations.append(time.perf_counter() - self.started)

    def __enter__(self):
        self.handles = [self.model.register_forward_pre_hook(self._before),
                        self.model.register_forward_hook(self._after)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    @property
    def prefill(self):
        return self.durations[0] if self.durations else None

    @property
    def decode(self):
        return sum(self.durations[1:]) if self.durations else None
