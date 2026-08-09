from .base_adapter import AdapterOutput, PlannerModelAdapter
from .pluto_adapter import PlutoModelAdapter

__all__ = [
    "AdapterOutput",
    "PlannerModelAdapter",
    "PlutoModelAdapter",
    "DiffusionModelAdapter",
]


def __getattr__(name):
    # Diffusion adapter is imported lazily so the PLUTO path never needs the
    # diffusion_planner package.
    if name == "DiffusionModelAdapter":
        from .diffusion_adapter import DiffusionModelAdapter

        return DiffusionModelAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
