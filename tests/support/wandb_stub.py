"""Minimal in-memory stand-in for the ``wandb`` module.

Used by the W&B logger tests so they can drive the logger without a live run.
The ``wandb_stub`` fixture in ``conftest.py`` patches the logger's accessor with
a fresh instance of this class.
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeWandb:
    """Records what would have been logged to a real ``wandb`` run."""

    class Image:
        def __init__(self, fig):
            self.fig = fig

    class Video:
        def __init__(self, data, fps, format):
            self.data = data
            self.fps = fps
            self.format = format

    class Table:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    def __init__(self):
        self.logged: list[dict] = []
        self.defined_metrics: list[str] = []
        self.inits: list[dict] = []

    def log(self, data, step=None):
        self.logged.append({"data": data, "step": step})

    def define_metric(self, pattern):
        self.defined_metrics.append(pattern)

    def init(self, *, project=None, name=None, config=None, **kwargs):
        # Mirror a real strict ``wandb.init`` closely enough to reject the
        # parameters this codebase never forwards — an open ``**kwargs`` sink
        # would silently accept typo'd/nonexistent args and hide broken callers.
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        call = {"project": project, "name": name, "config": config}
        self.inits.append(call)
        return SimpleNamespace(**call)
