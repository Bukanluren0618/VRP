# src/analysis/visualizations.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx

# --- Global Plotting Settings ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Arial")
def _finalize_figure(
    fig,
    output_path,
    *,
    close=True,
    show=True,
    ensure_dir=True,
    **savefig_kwargs,
):
    """Save a matplotlib figure and show it when the backend supports it.

    The PyCharm ``backend_interagg`` backend bundled with recent versions of
    Matplotlib no longer exposes ``tostring_rgb``. Attempting to display the
    plot using that backend triggers an ``AttributeError``. This helper saves
    the figure and skips ``plt.show()`` when the backend lacks the required
    capability, preventing the application from crashing while still exporting
    the visuals.
    """
    output_path = Path(output_path)
    if ensure_dir:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, **savefig_kwargs)

    backend_name = plt.get_backend().lower()
    canvas = getattr(fig, "canvas", None)
    manager = getattr(canvas, "manager", None)
    needs_skip = "backend_interagg" in backend_name and not hasattr(canvas, "tostring_rgb")

    if not show:
        print(f"[visualizations] Figure saved to {output_path} (display skipped by caller).")
    elif needs_skip:
        print(
            f"[visualizations] Skipping interactive display for backend '{backend_name}'. "
            f"Figure saved to {output_path}."
        )
    elif manager is None or not hasattr(manager, "show"):
        print(
            f"[visualizations] Backend '{backend_name}' does not support interactive display. "
            f"Figure saved to {output_path}."
        )
    else:
        try:
            manager.show()
        except Exception as exc:
            print(f"[visualizations] Interactive display failed ({exc}). Figure saved to {output_path}.")

    if close:
        plt.close(fig)



# --- 1. Road Network and Vehicle Routes Visualization ---
def plot_road_network_with_routes(
    road_network,
    solution_routes,
    output_dir="results",
    title="城市路网与车辆行驶轨迹概览",
):


    print("[visualizations] 正在生成包含车辆轨迹的完整路网图...")

    output_dir = Path(output_dir or "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "fleet_routing_plan_overview.png"
    pdf_path = output_dir / "fleet_routing_plan_overview.pdf"

    pos = nx.get_node_attributes(road_network, 'pos')
    if not pos:
        pos = nx.spring_layout(road_network, seed=42)

    node_info = {node: road_network.nodes[node].get('info', {}) for node in road_network.nodes}
    depots = [node for node, info in node_info.items() if info.get('type') == 'Depot']
    customers = [node for node, info in node_info.items() if info.get('type') == 'Customer']
    stations = [node for node, info in node_info.items() if info.get('type') == 'SwapStation']

    special_nodes = set(depots + customers + stations)
    background_nodes = [node for node in road_network.nodes if node not in special_nodes]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor('white')

    nx.draw_networkx_edges(
        road_network, pos, ax=ax, width=0.6, alpha=0.4, edge_color='#999999'
    )

    if background_nodes:
        nx.draw_networkx_nodes(
            road_network,
            pos,
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffcc4d', markeredgecolor='#b8860b',
                    markersize=10, label='仓库'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#74a9cf', markeredgecolor='#1f78b4',
                    markersize=8, label='客户'),
        plt.Line2D([0], [0], marker='p', color='w', markerfacecolor='#7fc97f', markeredgecolor='#3c763d',
                    markersize=9, label='换电站'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='#333333',
                    markersize=12, label='车辆起点'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='white', markeredgecolor='#333333',
                    markersize=9, label='车辆终点')
    ]

    for vehicle_id, line in route_handles:
        legend_handles.append(
            plt.Line2D([0], [0], color=line.get_color(), linewidth=2.6, label=f'{vehicle_id} 行驶轨迹')
        )

    ax.legend(handles=legend_handles, loc='upper right', frameon=True, framealpha=0.92, title='图例', fontsize=9)
    ax.set_title(title, fontsize=24, fontweight='bold', pad=18)
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.05)

    plt.tight_layout()
    _finalize_figure(fig, png_path, dpi=320, bbox_inches='tight', show=False, close=False)
    _finalize_figure(fig, pdf_path, format='pdf', bbox_inches='tight')
    print(
        "[visualizations] 路网图已生成并保存到: "
        f"{png_path} (PNG) 和 {pdf_path} (PDF)。"
    )



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