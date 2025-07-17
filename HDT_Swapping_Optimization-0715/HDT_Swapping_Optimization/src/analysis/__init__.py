from . import post_analysis  # noqa: F401

from . import pre_checks  # noqa: F401

from . import post_analysis
print_task_assignments = post_analysis.print_task_assignments


# Re-export frequently used helpers for easier access
check_task_feasibility = pre_checks.check_task_feasibility
safe_value = post_analysis.safe_value
plot_road_network_with_routes = post_analysis.plot_road_network_with_routes
plot_station_energy_schedule = post_analysis.plot_station_energy_schedule
print_vehicle_swap_nodes = post_analysis.print_vehicle_swap_nodes
print_vehicle_routes = post_analysis.print_vehicle_routes

# Keep __all__ in sync with the helpers we re-export so that
# ``from src.analysis import *`` works as expected and IDEs can
# discover the available utilities.
plot_hdt_metrics = post_analysis.plot_hdt_metrics

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
