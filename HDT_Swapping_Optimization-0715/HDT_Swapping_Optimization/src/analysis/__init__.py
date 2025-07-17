"""Convenience imports for analysis utilities.
This module re-exports the most frequently used helpers from
``post_analysis`` alongside the :func:`check_task_feasibility` utility.
The explicit imports avoid circular initialization errors when ``post_analysis``
itself imports this package.
"""
from importlib import import_module

_POST_FUNCS = [
    "safe_value",
    "plot_road_network_with_routes",
    "plot_station_energy_schedule",
    "print_task_assignments",
    "print_vehicle_swap_nodes",
    "print_vehicle_routes",
    "plot_hdt_metrics",
]

_PRE_FUNCS = ["check_task_feasibility"]

__all__ = ["post_analysis"] + _POST_FUNCS + _PRE_FUNCS



def __getattr__(name):
    """Lazily import helpers or the module itself to avoid circular imports."""
    if name == "post_analysis":
        return import_module(".post_analysis", __name__)
    if name in _POST_FUNCS:
        module = import_module(".post_analysis", __name__)
        return getattr(module, name)
    if name in _PRE_FUNCS:
        module = import_module(".pre_checks", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")