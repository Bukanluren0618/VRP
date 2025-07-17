from __future__ import annotations
import importlib
from types import ModuleType


_post_mod: ModuleType | None = None
_pre_mod: ModuleType | None = None

_POST_FUNCS = {
    "safe_value",
    "plot_road_network_with_routes",
    "plot_station_energy_schedule",
    "print_task_assignments",
    "print_vehicle_swap_nodes",
    "print_vehicle_routes",
    "plot_hdt_metrics",
}

_PRE_FUNCS = {
    "check_task_feasibility",
}

__all__ = [
    "post_analysis",
    "pre_checks",
    *_PRE_FUNCS,
    *_POST_FUNCS,
]

def _load_post() -> ModuleType:
    global _post_mod
    if _post_mod is None:
        _post_mod = importlib.import_module(".post_analysis", __name__)
    return _post_mod


def _load_pre() -> ModuleType:
    global _pre_mod
    if _pre_mod is None:
        _pre_mod = importlib.import_module(".pre_checks", __name__)
    return _pre_mod


def __getattr__(name: str):
    if name == "post_analysis":
        return _load_post()
    if name == "pre_checks":
        return _load_pre()
    if name in _POST_FUNCS:
        return getattr(_load_post(), name)
    if name in _PRE_FUNCS:
        return getattr(_load_pre(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


