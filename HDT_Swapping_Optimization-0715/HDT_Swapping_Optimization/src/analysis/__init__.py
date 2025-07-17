from importlib import import_module

__all__ = [
    "post_analysis",
    "pre_checks",
    "check_task_feasibility",
    "safe_value",
    "plot_road_network_with_routes",
    "plot_station_energy_schedule",
    "print_task_assignments",
    "print_vehicle_swap_nodes",
    "print_vehicle_routes",
    "plot_hdt_metrics",
]

_post_mod = None
_pre_mod = None


def _load_post():
    """Import :mod:`post_analysis` on demand and cache the module."""
    global _post_mod
    if _post_mod is None:
        _post_mod = import_module(".post_analysis", __name__)