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

def _finalize_figure(fig, output_path, *, close=True, **savefig_kwargs):
    """Save a matplotlib figure and show it when the backend supports it.

    The PyCharm ``backend_interagg`` backend bundled with recent versions of
    Matplotlib no longer exposes ``tostring_rgb``. Attempting to display the
    plot using that backend triggers an ``AttributeError``. This helper saves
    the figure and skips ``plt.show()`` when the backend lacks the required
    capability, preventing the application from crashing while still exporting
    the visuals.
    """

    fig.savefig(output_path, **savefig_kwargs)

    backend_name = plt.get_backend().lower()
    canvas = getattr(fig, "canvas", None)
    manager = getattr(canvas, "manager", None)
    needs_skip = "backend_interagg" in backend_name and not hasattr(canvas, "tostring_rgb")

    if needs_skip:
        print(f"[visualizations] Skipping interactive display for backend '{backend_name}'. Figure saved to {output_path}.")
    elif manager is None or not hasattr(manager, "show"):
        print(f"[visualizations] Backend '{backend_name}' does not support interactive display. Figure saved to {output_path}.")
    else:
        try:
            manager.show()
        except Exception as exc:
            print(f"[visualizations] Interactive display failed ({exc}). Figure saved to {output_path}.")

    if close:
        plt.close(fig)



# --- 1. Road Network and Vehicle Routes Visualization ---
def plot_road_network_with_routes(road_network, solution_routes, output_dir, title="Fleet Routing Plan"):
    """
    Visualizes the road network and plots the optimized vehicle routes on top.
    """
    print("--- Visualizing Fleet Routing Plan ---")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(20, 16))

    pos = nx.get_node_attributes(road_network, 'pos')
    if not pos:
        pos = nx.spring_layout(road_network, seed=42)

    node_info = nx.get_node_attributes(road_network, 'info')
    depots = [node for node, info in node_info.items() if info.get('type') == 'Depot']
    customers = [node for node, info in node_info.items() if info.get('type') == 'Customer']
    stations = [node for node, info in node_info.items() if info.get('type') == 'SwapStation']

    nx.draw_networkx_edges(road_network, pos, alpha=0.2, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(road_network, pos, nodelist=depots, node_color='gold', node_shape='s', node_size=400,
                           label='Depot')
    nx.draw_networkx_nodes(road_network, pos, nodelist=customers, node_color='skyblue', node_size=200, label='Customer')
    nx.draw_networkx_nodes(road_network, pos, nodelist=stations, node_color='lightgreen', node_shape='p', node_size=350,
                           label='Station')
    nx.draw_networkx_labels(road_network, pos, font_size=8, ax=ax)

    if solution_routes:
        route_colors = plt.cm.get_cmap('gist_rainbow', len(solution_routes))
        for i, (vehicle_id, route) in enumerate(solution_routes.items()):
            if not route or len(route) < 2: continue

            # This logic needs to be more robust, assuming 'info' contains the name
            route_node_ids = []
            for name in route:
                for node_id, info in node_info.items():
                    if info.get('name') == name:
                        route_node_ids.append(node_id)
                        break

            if len(route_node_ids) >= 2:
                route_edges = list(zip(route_node_ids[:-1], route_node_ids[1:]))
                nx.draw_networkx_edges(road_network, pos, edgelist=route_edges,
                                       width=2.5, alpha=0.9, edge_color=route_colors(i),
                                       label=vehicle_id, ax=ax, connectionstyle='arc3,rad=0.1')

    ax.set_title(title, fontsize=24, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], marker='s', color='w', label='Depot', markerfacecolor='gold', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Customer', markerfacecolor='skyblue', markersize=10),
        plt.Line2D([0], [0], marker='p', color='w', label='Station', markerfacecolor='lightgreen', markersize=10)
    ])
    ax.legend(handles=handles, title="Legend")

    plt.tight_layout()
    _finalize_figure(fig, f'{output_dir}/fleet_routing_plan_detailed.pdf', format='pdf')


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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
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