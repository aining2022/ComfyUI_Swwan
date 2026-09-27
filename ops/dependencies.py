"""Resolve optional libraries only when their feature is executed."""
import importlib


class _LazyModule:
    def __init__(self, name, extra):
        self.name = name
        self.extra = extra
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = importlib.import_module(self.name)
            except ImportError as exc:
                raise RuntimeError(
                    f"This operation needs {self.name}; install the '{self.extra}' optional dependencies."
                ) from exc
        return getattr(self._module, name)


def lazy_module(name, extra):
    return _LazyModule(name, extra)


def lazy_function(module, name, extra):
    library = lazy_module(module, extra)
    def call(*args, **kwargs):
        return getattr(library, name)(*args, **kwargs)
    return call
