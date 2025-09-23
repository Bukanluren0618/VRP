# src/analysis/visualizations.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from pathlib import Path
from matplotlib.lines import Line2D

# --- Global Plotting Settings ---
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Arial")


def _normalize_solution_routes(solution_routes, node_info):
    """归一化 routes: {vehicle_id: [node1, node2, ...]}"""
    if not isinstance(solution_routes, dict):
        return {}

    normalized = {}
    valid_nodes = set(node_info.keys())
    for vid, seq in solution_routes.items():
        if not seq:
            continue
        cleaned, last = [], object()
        for n in seq:
            if n in valid_nodes and n != last:
                cleaned.append(n)
                last = n
        if len(cleaned) >= 2:
            normalized[vid] = cleaned
    return normalized
def _finalize_figure(fig, output_path, *, close=True, show=True, ensure_dir=True, **savefig_kwargs):
    """保存并显示图像"""
    output_path = Path(output_path)
    if ensure_dir:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, **savefig_kwargs)
    print(f"[visualizations] Saved figure to {Path(output_path).resolve()}")
    if close:
        plt.close(fig)

    backend_name = plt.get_backend().lower()
    canvas = getattr(fig, "canvas", None)
    manager = getattr(canvas, "manager", None)
    needs_skip = "backend_interagg" in backend_name and not hasattr(canvas, "tostring_rgb")

    message = None

    if not show:
        message = (
            f"[visualizations] Figure saved to {output_path} "
            "(display skipped by caller)."
        )
    elif needs_skip:
        message = (
            f"[visualizations] Skipping interactive display for backend "
            f"'{backend_name}'. Figure saved to {output_path}."
        )
    elif manager is None or not hasattr(manager, "show"):
        message = (
            f"[visualizations] Backend '{backend_name}' does not support interactive display. "
            f"Figure saved to {output_path}."
        )
    else:
        try:
            manager.show()
        except Exception as exc:
            message = (
                f"[visualizations] Interactive display failed ({exc}). "
                f"Figure saved to {output_path}."
            )

    if message:
        print(message)

    if close:
        plt.close(fig)


# --- 1. Road Network and Vehicle Routes Visualization ---
def plot_road_network_with_routes(
    road_network,
    solution_routes,
    output_dir="results",
    title="City Road Network and Vehicle Routes Overview",
):
    print("[visualizations] Generating comprehensive road-network visualization...")

    output_dir = Path(output_dir or "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fleet_routing_plan_overview.png"
    pdf_path = output_dir / "fleet_routing_plan_overview.pdf"

    pos = nx.get_node_attributes(road_network, "pos")
    if not pos:
        pos = nx.spring_layout(road_network, seed=42)

    node_info = {node: road_network.nodes[node].get("info", {}) for node in road_network.nodes}
    depots = [node for node, info in node_info.items() if info.get("type") == "Depot"]
    customers = [node for node, info in node_info.items() if info.get("type") == "Customer"]
    stations = [node for node, info in node_info.items() if info.get("type") == "SwapStation"]

    special_nodes = set(depots + customers + stations)
    background_nodes = [node for node in road_network.nodes if node not in special_nodes]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(18, 14))
        fig.patch.set_facecolor("white")

        # 画路网边
        nx.draw_networkx_edges(road_network, pos, ax=ax, width=0.6, alpha=0.4, edge_color="#999999")

        def draw_nodes(nodes, label, **style):
            if nodes:
                nx.draw_networkx_nodes(road_network, pos, nodelist=nodes, label=label, ax=ax, **style)

        # 各类节点
        draw_nodes(background_nodes, "Road Network Node", node_color="#d9d9d9",
                   node_shape="o", node_size=55, edgecolors="#777777", linewidths=0.2)
        draw_nodes(depots, "Depot", node_color="#ffcc4d", node_shape="s",
                   node_size=420, edgecolors="#b8860b", linewidths=1.4)
        draw_nodes(customers, "Customer", node_color="#74a9cf", node_shape="o",
                   node_size=260, edgecolors="#1f78b4", linewidths=1.0)
        draw_nodes(stations, "Battery Swap Station", node_color="#7fc97f", node_shape="p",
                   node_size=360, edgecolors="#3c763d", linewidths=1.2)

        # 节点标签
        node_labels = {node: str(node) for node in road_network.nodes}
        nx.draw_networkx_labels(road_network, pos, labels=node_labels,
                                font_size=7, font_color="#3d3d3d", ax=ax)

        # 注释名
        annotation_offset = {"Depot": (0, 14), "Customer": (0, -16), "SwapStation": (0, 14)}
        annotation_color = {"Depot": "#b36200", "Customer": "#0b559f", "SwapStation": "#1b7f5f"}
        for node in special_nodes:
            info = node_info.get(node, {})
            name, node_type = info.get("name"), info.get("type")
            if not name or name == str(node):
                continue
            dx, dy = annotation_offset.get(node_type, (0, 12))
            color = annotation_color.get(node_type, "#444444")
            ax.annotate(name, xy=pos[node], xytext=(dx, dy), textcoords="offset points",
                        ha="center", fontsize=8.5, color=color,
                        fontweight="bold" if node_type == "Depot" else "normal",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

        # 路径绘制
        normalized_routes = _normalize_solution_routes(solution_routes, node_info)
        route_handles = []
        # 名称 -> node_id
        name_to_id = {info.get("name"): node for node, info in node_info.items() if "name" in info}

        if normalized_routes:
            cmap = plt.cm.get_cmap("tab20", max(1, len(normalized_routes)))
            for idx, (vehicle_id, node_sequence) in enumerate(normalized_routes.items()):
                if len(node_sequence) < 2:
                    continue
                route_color = cmap(idx)
                all_coords = []

                for start_name, end_name in zip(node_sequence[:-1], node_sequence[1:]):
                    if start_name == end_name:
                        continue
                    try:
                        start_id, end_id = name_to_id[start_name], name_to_id[end_name]
                        path_nodes = nx.shortest_path(road_network, source=start_id, target=end_id, weight="distance")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    coords = [pos[n] for n in path_nodes if n in pos]
                    all_coords.extend(coords)

                if len(all_coords) >= 2:
                    xs, ys = zip(*all_coords)
                    (line,) = ax.plot(xs, ys, color=route_color, linewidth=2.6,
                                      alpha=0.95, solid_capstyle="round", zorder=5)
                    route_handles.append((vehicle_id, line))

                    # 在路径中间加车辆ID
                    mid = len(xs) // 2
                    ax.text(xs[mid], ys[mid], vehicle_id, fontsize=9, color=route_color,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

                    # 在经过的节点上加序号
                    for i, (x, y) in enumerate(all_coords):
                        ax.text(x, y, str(i), fontsize=6, color=route_color)

        # 图例
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9d9d9",
                   markeredgecolor="#777777", markersize=6, label="Road Network Node"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#ffcc4d",
                   markeredgecolor="#b8860b", markersize=10, label="Depot"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#74a9cf",
                   markeredgecolor="#1f78b4", markersize=8, label="Customer"),
            Line2D([0], [0], marker="p", color="w", markerfacecolor="#7fc97f",
                   markeredgecolor="#3c763d", markersize=9, label="Battery Swap Station"),
        ]
        for vehicle_id, line in route_handles:
            legend_handles.append(Line2D([0], [0], color=line.get_color(),
                                         linewidth=2.6, label=f"{vehicle_id} Route"))
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.92,
                  title="Legend", fontsize=9)

        ax.set_title(title, fontsize=24, fontweight="bold", pad=18)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.05)

        plt.tight_layout()
        _finalize_figure(fig, png_path, dpi=320, bbox_inches="tight", show=False, close=False)
        _finalize_figure(fig, pdf_path, format="pdf", bbox_inches="tight")
        print(f"[visualizations] Road-network map saved to {png_path} (PNG) and {pdf_path} (PDF).")


# --- 2. Case 1: Scheduled vs. Unscheduled Comparison Visualizations ---
def plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir):
    """
    Generates all comparison visualizations for Case 1.
    """
    print("\n" + "=" * 20 + " Visualizing Case 1: Scheduled vs. Unscheduled " + "=" * 20)

    # --- Economic Cost Comparison (Bar Chart) ---
    cost_df = pd.DataFrame({
        'Scenario': ['Scheduled Fleet', 'Unscheduled Fleet'],
        'Total Cost': [scheduled_stats['total_cost'], unscheduled_stats['total_cost']]
    })
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(data=cost_df, x='Scenario', y='Total Cost', hue='Scenario',
                palette=['#31a354', '#a1d99b'], dodge=False, legend=False, ax=ax)
    ax.set_title('Case 1: Economic Cost Comparison', fontsize=16)
    ax.set_ylabel('Total Daily Cost (Yuan)')
    _finalize_figure(fig, f"{output_dir}/case1_cost_comparison.pdf", format='pdf', bbox_inches='tight')

    # --- Energy Flow Comparison (Stacked Area Chart) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    for ax, stats, title in [(ax1, scheduled_stats, 'Scheduled Fleet'), (ax2, unscheduled_stats, 'Unscheduled Fleet')]:
        df_energy = stats['energy_flows']

        # --- MODIFICATION: Convert timedelta index to numeric hours for plotting ---
        x_values = df_energy.index.total_seconds() / 3600

        ax.stackplot(x_values, df_energy['grid_power'], df_energy['pv_power'], df_energy['bess_discharge'],
                     labels=['Grid Input', 'PV Output', 'BESS Discharge'],
                     colors=['salmon', 'gold', 'lightgreen'])
        ax.plot(x_values, df_energy['total_demand'], color='black', linestyle='--', label='Total Demand (HDT+EV)')
        ax.set_title(f'Energy Flow: {title}', fontsize=14)
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper left')
    ax2.set_xlabel('Time (Hour of Day)')
    fig.suptitle('Case 1: Station Energy Flow Comparison', fontsize=18, y=0.99)
    _finalize_figure(fig, f"{output_dir}/case1_energy_flow_comparison.pdf", format='pdf', bbox_inches='tight')

    # --- Grid Peak Shaving Comparison (Line Chart) ---
    fig, ax = plt.subplots(figsize=(15, 7))
    sched_x = scheduled_stats['energy_flows'].index.total_seconds() / 3600
    unsched_x = unscheduled_stats['energy_flows'].index.total_seconds() / 3600
    ax.plot(sched_x, scheduled_stats['energy_flows']['grid_power'], label='Scheduled Grid Load', color='#31a354',
            linewidth=2)
    ax.plot(unsched_x, unscheduled_stats['energy_flows']['grid_power'], label='Unscheduled Grid Load', color='#a1d99b',
            linestyle='--')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title('Case 1: Grid Peak Shaving Comparison', fontsize=16)
    ax.set_xlabel('Time (Hour of Day)')
    ax.set_ylabel('Power Drawn from Grid (kW)')
    ax.legend()
    _finalize_figure(fig, f"{output_dir}/case1_peak_shaving_comparison.pdf", format='pdf', bbox_inches='tight')

    # --- Delivery & Queue Time (Printed Table) ---
    df_times = pd.DataFrame({
        'Metric': ['Avg. Delivery Time (h)', 'Avg. Queue Time (min)', 'Total Wait Time (h)'],
        'Scheduled Fleet': [f"{scheduled_stats['avg_delivery_time']:.2f}", f"{scheduled_stats['avg_queue_time']:.2f}",
                            f"{scheduled_stats['total_wait_time']:.2f}"],
        'Unscheduled Fleet': [f"{unscheduled_stats['avg_delivery_time']:.2f}",
                              f"{unscheduled_stats['avg_queue_time']:.2f}",
                              f"{unscheduled_stats['total_wait_time']:.2f}"]
    }).set_index('Metric')
    print("\n--- Case 1: Time Performance Comparison ---")
    print(df_times)


# --- 3. Case 2: EV + EHDT Arrival Heatmap ---
def plot_case2_heatmap(arrival_matrix, output_dir):
    """Generates the station arrival heatmap for Case 2."""
    print("\n" + "=" * 20 + " Visualizing Case 2: Station Arrival Heatmap " + "=" * 20)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(arrival_matrix, cmap='YlOrRd', linewidths=.5, annot=True, fmt=".0f", ax=ax)
    ax.set_title('Case 2: Station Vehicle Arrivals (EV + EHDT)', fontsize=16)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Station ID')
    _finalize_figure(fig, f"{output_dir}/case2_arrival_heatmap.pdf", format='pdf', bbox_inches='tight')


# --- 4. Case 3: Grid Service Strategy Comparison ---
def plot_case3_comparison(stats_dict, output_dir):
    """Generates the V2G strategy comparison charts for Case 3."""
    print("\n" + "=" * 20 + " Visualizing Case 3: Grid Service Strategies " + "=" * 20)

    # --- Economic Cost Comparison ---
    cost_df = pd.DataFrame({
        'Strategy': list(stats_dict.keys()),
        'Total Cost': [stats['total_cost'] for stats in stats_dict.values()]
    })
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=cost_df, x='Strategy', y='Total Cost', hue='Strategy',
                palette=['#2c7fb8', '#7fcdbb', '#edf8b1'], dodge=False, legend=False, ax=ax)
    ax.set_title('Case 3: Economic Cost of Different Battery Strategies', fontsize=16)
    ax.set_ylabel('Total Daily Cost (Yuan)')
    ax.tick_params(axis='x', labelrotation=15)
    plt.setp(ax.get_xticklabels(), ha='right')
    _finalize_figure(fig, f"{output_dir}/case3_cost_comparison.pdf", format='pdf', bbox_inches='tight')

    # --- Peak Shaving Comparison ---
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ['#2c7fb8', '#7fcdbb', '#edf8b1']
    linestyles = ['-', '--', ':']
    for i, (name, stats) in enumerate(stats_dict.items()):
        x_values = stats['grid_load'].index.total_seconds() / 3600
        ax.plot(x_values, stats['grid_load'], label=name, color=colors[i], linestyle=linestyles[i], linewidth=2.5)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title('Case 3: Grid Peak Shaving Comparison', fontsize=16)
    ax.set_xlabel('Time (Hour of Day)')
    ax.set_ylabel('Power Drawn from Grid (kW)')
    ax.legend()
    _finalize_figure(fig, f"{output_dir}/case3_peak_shaving_comparison.pdf", format='pdf', bbox_inches='tight')