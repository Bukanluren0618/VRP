# src/analysis/visualizations.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from pyomo.environ import value
# If you need constants like BESS_CAPACITY_KWH below, import config here:
from src.common import config_final as config

# --- Global plotting settings (English-only; no CJK glyphs) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


def safe_value(var_or_data):
    """Safely get a Pyomo var value; return 0 if None/invalid."""
    if hasattr(var_or_data, "is_variable_type") and var_or_data.is_variable_type():
        return value(var_or_data, exception=False) if var_or_data.value is not None else 0
    return var_or_data


# --- Figure 1: Road network and fleet routes ---
def plot_road_network_with_routes(road_network, solution_routes, data, output_dir, title="Fleet Routing Plan"):
    """Visualize the road network, facilities and vehicle trajectories."""
    print("-> Generating Plot 1: Road Network and Fleet Routes...")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(24, 20))

    pos = nx.get_node_attributes(road_network, "pos")
    if not pos:
        pos = nx.spring_layout(road_network, seed=42)

        # Draw the base road network
        nx.draw_networkx_edges(road_network, pos, alpha=0.2, edge_color="gray", ax=ax)
        nx.draw_networkx_nodes(
            road_network,
            pos,
            node_size=80,
            node_color="lightgray",
            alpha=0.7,
            label="Road Node",
        )

    # Extract node types from data
    locations = data["locations"]
    depots = [info["node_id"] for info in locations.values() if info["type"] == "Depot"]
    customers = [info["node_id"] for info in locations.values() if info["type"] == "Customer"]
    stations = [info["node_id"] for info in locations.values() if info["type"] == "SwapStation"]

    # Overlay facility nodes with distinctive markers
    if depots:
        nx.draw_networkx_nodes(
            road_network,
            pos,
            nodelist=depots,
            node_color="red",
            node_shape="s",
            node_size=420,
            label="Depot",
            edgecolors="black",
            linewidths=1.2,
        )
    if stations:
        nx.draw_networkx_nodes(
            road_network,
            pos,
            nodelist=stations,
            node_color="lightgreen",
            node_shape="p",
            node_size=360,
            label="Swap Station",
            edgecolors="black",
            linewidths=1.2,
        )
    if customers:
        nx.draw_networkx_nodes(
            road_network,
            pos,
            nodelist=customers,
            node_color="skyblue",
            node_size=260,
            label="Customer",
            edgecolors="black",
            linewidths=0.8,
        )

    # Number every road node
    for node_id, (x_coord, y_coord) in pos.items():
        ax.text(
            x_coord,
            y_coord,
            str(node_id),
            fontsize=6,
            color="dimgray",
            ha="center",
            va="center",
            alpha=0.8,
        )

    # Facility name labels (slightly offset to improve readability)
    facility_labels = {info["node_id"]: name for name, info in locations.items()}
    for node_id, label in facility_labels.items():
        if node_id in pos:
            ax.text(
                pos[node_id][0],
                pos[node_id][1] + 0.015,
                label,
                fontsize=8,
                fontweight="bold",
                color="black",
                ha="center",
            )

    # Vehicle routes
    if solution_routes:
        route_colors = plt.cm.get_cmap("gist_rainbow", max(len(solution_routes), 1))
        for i, (vehicle_id, route_info) in enumerate(solution_routes.items()):
            segments = []
            if isinstance(route_info, dict):
                segments = route_info.get("segments") or []
                if not segments:
                    node_path = route_info.get("node_path")
                    locations = route_info.get("locations")
                    if node_path:
                        segments = [{"node_path": node_path, "locations": locations}]

            if not segments:
                continue

            for seg_idx, segment in enumerate(segments, start=1):
                node_path = segment.get("node_path") or []
                if len(node_path) < 2:
                    continue

                route_edges = list(zip(node_path[:-1], node_path[1:]))
                nx.draw_networkx_edges(
                    road_network,
                    pos,
                    edgelist=route_edges,
                    width=3.0,
                    alpha=0.9,
                    edge_color=[route_colors(i)] * len(route_edges),
                    ax=ax,
                )

                start_node = node_path[0]
                end_node = node_path[-1]
                start_label = f"{vehicle_id} S{seg_idx} Start"
                end_label = f"{vehicle_id} S{seg_idx} End"

                if start_node in pos:
                    ax.scatter(
                        [pos[start_node][0]],
                        [pos[start_node][1]],
                        color=route_colors(i),
                        marker="o",
                        s=160,
                        edgecolors="white",
                        linewidths=1.2,
                        zorder=5,
                    )
                    ax.text(
                        pos[start_node][0],
                        pos[start_node][1] - 0.02,
                        start_label,
                        fontsize=7,
                        color=route_colors(i),
                        ha="center",
                        fontweight="bold",
                    )

                if end_node in pos:
                    ax.scatter(
                        [pos[end_node][0]],
                        [pos[end_node][1]],
                        color=route_colors(i),
                        marker="X",
                        s=160,
                        edgecolors="white",
                        linewidths=1.2,
                        zorder=5,
                    )
                    if end_node != start_node:
                        ax.text(
                            pos[end_node][0],
                            pos[end_node][1] - 0.02,
                            end_label,
                            fontsize=7,
                            color=route_colors(i),
                            ha="center",
                            fontweight="bold",
                        )


    ax.set_title(title, fontsize=28, fontweight="bold")
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Legend", loc="upper right")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/1_fleet_routing_plan.pdf", format="pdf")
    plt.close()


# --- Figures 2 & 3: Station energy flows and SOC trajectories ---
def plot_station_energy_details(station_id, scheduled_flows, unscheduled_flows, output_dir):
    """
    Generate detailed energy-flow and SOC comparison plots for a single station.
    Expected columns in flows: ['pv_power','bess_discharge','bess_charge','total_demand','bess_soc']
    The index should be a TimedeltaIndex (time from start); x-axis uses hours.
    """
    print(f"-> Generating Plots 2 & 3: Energy Details for {station_id}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16), sharex=True)

    # --- Energy flow comparison (scheduled vs. unscheduled) ---
    for ax, flows, sub_title in [
        (ax1, scheduled_flows, f"{station_id} Energy Flows (Scheduled)"),
        (ax2, unscheduled_flows, f"{station_id} Energy Flows (Unscheduled)"),
    ]:
        x_values = flows.index.total_seconds() / 3600.0

        # Positive is supply/discharge; negative is charge/consumption
        ax.bar(x_values, flows["pv_power"], width=0.2, label="PV Generation", color="gold")
        ax.bar(
            x_values,
            flows["bess_discharge"],
            width=0.2,
            bottom=flows["pv_power"],
            label="BESS Discharge",
            color="lightgreen",
        )
        ax.bar(x_values, -flows["bess_charge"], width=0.2, label="BESS Charge", color="skyblue")
        ax.plot(x_values, flows["total_demand"], color="red", linestyle="--", label="Total Demand (HDT+EV)")

        ax.set_title(sub_title, fontsize=16)
        ax.set_ylabel("Power (kW)")
        ax.legend(loc="upper left")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)

    plt.xlabel("Time (h)")
    fig.suptitle(f"Station {station_id} Energy Balance Comparison", fontsize=22, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/2_station_energy_flow_comparison.pdf", format="pdf")
    plt.close()

    # --- SOC trajectories ---
    fig, ax = plt.subplots(figsize=(18, 8))
    sched_x = scheduled_flows.index.total_seconds() / 3600.0
    unsched_x = unscheduled_flows.index.total_seconds() / 3600.0

    ax.plot(sched_x, scheduled_flows["bess_soc"], label="SOC (Scheduled)", color="blue", linewidth=2.5)
    ax.plot(
        unsched_x, unscheduled_flows["bess_soc"], label="SOC (Unscheduled)", color="orange", linestyle="--", linewidth=2.5
    )

    ax.set_title(f"Station {station_id} Inventory Battery SOC Trajectory", fontsize=18)
    ax.set_ylabel("Energy (kWh)")
    ax.set_xlabel("Time (h)")
    ax.legend()
    ax.grid(True)
    plt.savefig(f"{output_dir}/3_station_soc_trajectory.pdf", format="pdf")
    plt.close()

    # Print utilization table in console
    bess_capacity = config.BESS_CAPACITY_KWH
    util_sched = (scheduled_flows["bess_charge"].sum() * config.TIME_STEP_HOURS) / bess_capacity if bess_capacity else 0
    util_unsched = (unscheduled_flows["bess_charge"].sum() * config.TIME_STEP_HOURS) / bess_capacity if bess_capacity else 0
    df_util = pd.DataFrame(
        {
            "Strategy": ["Scheduled", "Unscheduled"],
            "Total Charge (kWh)": [
                scheduled_flows["bess_charge"].sum() * config.TIME_STEP_HOURS,
                unscheduled_flows["bess_charge"].sum() * config.TIME_STEP_HOURS,
            ],
            "Utilization (%)": [f"{util_sched:.2%}", f"{util_unsched:.2%}"],
        }
    )
    print("\n--- BESS Utilization ---")
    print(df_util.to_string(index=False))
    print("------------------------\n")


# --- Figure 4: Single HDT truck metrics ---
def plot_single_truck_metrics(vehicle_id, model, route, data, config, output_dir):
    """
    Plot SOC, payload, and energy-per-km over time for a single truck.
    `route` is a list of location names present in model indices.
    """
    print(f"-> Generating Plot 4: Metrics for Vehicle {vehicle_id}...")

    timeline, soc_line, weight_line, consumption_line = [], [], [], []

    # Extract states along the route
    for i in range(len(route)):
        loc_name = route[i]

        # Arrival state
        arrival_time = safe_value(model.arrival_time[loc_name, vehicle_id])
        arrival_soc = safe_value(model.soc_arrival[loc_name, vehicle_id])
        # Use 'weight_arrival' (matches your integrated model variable naming)
        arrival_weight = safe_value(model.weight_arrival[loc_name, vehicle_id])

        timeline.append(arrival_time)
        soc_line.append(arrival_soc)
        weight_line.append(arrival_weight)

        # Energy-per-km rate (simple affine model)
        consumption_rate = (
            config.HDT_BASE_CONSUMPTION_KWH_PER_KM
            + arrival_weight * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
        )
        consumption_line.append(consumption_rate)

        # Departure (if not the last stop)
        if i < len(route) - 1:
            departure_time = safe_value(model.departure_time[loc_name, vehicle_id])
            if abs(departure_time - arrival_time) > 0.01:  # stopped
                timeline.append(departure_time)
                # assume payload & rate unchanged during stop
                weight_line.append(arrival_weight)
                consumption_line.append(consumption_rate)
                # check battery swap
                if (loc_name in model.STATIONS) and (safe_value(model.swap_decision[loc_name, vehicle_id]) > 0.5):
                    soc_line.append(config.HDT_BATTERY_CAPACITY_KWH)
                else:
                    soc_line.append(arrival_soc)

    # Sort by time
    sorted_data = sorted(zip(timeline, soc_line, weight_line, consumption_line))
    timeline, soc_line, weight_line, consumption_line = zip(*sorted_data)

    fig, ax1 = plt.subplots(figsize=(18, 9))

    # SOC
    color = "tab:blue"
    ax1.set_xlabel("Time (h)", fontsize=14)
    ax1.set_ylabel("Battery Energy (kWh)", color=color, fontsize=14)
    ax1.plot(timeline, soc_line, color=color, marker="o", linestyle="-", label="SOC")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axhline(
        y=config.HDT_MIN_SOC_KWH, color="red", linestyle="--", label=f"Safety Threshold ({config.HDT_MIN_SOC_KWH} kWh)"
    )

    # Payload
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Payload (ton)", color=color, fontsize=14)
    ax2.step(timeline, weight_line, where="post", color=color, linestyle="--", label="Payload")
    ax2.tick_params(axis="y", labelcolor=color)

    # Energy-per-km
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    color = "tab:green"
    ax3.set_ylabel("Energy per km (kWh/km)", color=color, fontsize=14)
    ax3.step(timeline, consumption_line, where="post", color=color, linestyle=":", label="Energy per km")
    ax3.tick_params(axis="y", labelcolor=color)

    fig.suptitle(f"Vehicle {vehicle_id} Metrics", fontsize=20)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/4_truck_{vehicle_id}_metrics.pdf", format="pdf")
    plt.close()


# --- Figure 5: Peak shaving comparison ---
def plot_peak_shaving_comparison(scheduled_grid_load, unscheduled_grid_load, output_dir):
    """
    Compare grid loads under two strategies to show peak shaving effect.
    `scheduled_grid_load` and `unscheduled_grid_load` are pandas Series with TimedeltaIndex.
    """
    print("-> Generating Plot 5: Peak Shaving Comparison...")
    plt.figure(figsize=(18, 9))

    sched_x = scheduled_grid_load.index.total_seconds() / 3600.0
    unsched_x = unscheduled_grid_load.index.total_seconds() / 3600.0

    plt.plot(sched_x, scheduled_grid_load, label="Grid Load (Scheduled)", linewidth=2.5)
    plt.plot(unsched_x, unscheduled_grid_load, label="Grid Load (Unscheduled)", linestyle="--", linewidth=2.5)

    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.title("Peak Shaving Comparison", fontsize=20)
    plt.xlabel("Time (h)", fontsize=14)
    plt.ylabel("Power from Grid (kW)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/5_peak_shaving_comparison.pdf", format="pdf")
    plt.close()
