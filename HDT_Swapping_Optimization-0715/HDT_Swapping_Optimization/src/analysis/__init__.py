"""Convenience imports for analysis utilities.
This module re-exports the most frequently used helpers from
``post_analysis`` alongside the :func:`check_task_feasibility` utility.
The explicit imports avoid circular initialization errors when ``post_analysis``
itself imports this package.
"""

from .pre_checks import check_task_feasibility
from .post_analysis import (
    safe_value,
    plot_road_network_with_routes,
    plot_station_energy_schedule,
    print_task_assignments,
    print_vehicle_swap_nodes,
    print_vehicle_routes,
    plot_hdt_metrics,
)
__all__ = [
    "safe_value",
    "plot_road_network_with_routes",
    "plot_station_energy_schedule",
    "print_task_assignments",
    "print_vehicle_swap_nodes",
    "print_vehicle_routes",
    "plot_hdt_metrics",
    "check_task_feasibility",
]