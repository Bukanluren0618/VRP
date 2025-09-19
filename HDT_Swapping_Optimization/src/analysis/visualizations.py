# src/analysis/visualizations.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx

# --- Global Plotting Settings ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示
sns.set_theme(style="whitegrid", font="Arial")


# --- 1. Road Network and Vehicle Routes Visualization (Preserved and Corrected) ---
def plot_road_network_with_routes(road_network, solution_routes, output_dir, title="Fleet Routing Plan"):
    """
    Visualizes the road network and plots the optimized vehicle routes on top.
    This version is corrected to properly handle node types from the data loader.
    """
    print("--- Visualizing Fleet Routing Plan ---")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(20, 16))

    pos = nx.get_node_attributes(road_network, 'pos')
    if not pos:
        print("Node positions not found, generating spring layout.")
        pos = nx.spring_layout(road_network, seed=42)

    # Correctly extract node types from the data structure
    node_info = nx.get_node_attributes(road_network, 'info')

    # Define nodes by type
    depots = [node for node, info in node_info.items() if info.get('type') == 'Depot']
    customers = [node for node, info in node_info.items() if info.get('type') == 'Customer']
    stations = [node for node, info in node_info.items() if info.get('type') == 'SwapStation']

    # Draw the base road network
    nx.draw_networkx_edges(road_network, pos, alpha=0.2, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(road_network, pos, nodelist=depots, node_color='gold', node_shape='s', node_size=400,
                           label='Depot')
    nx.draw_networkx_nodes(road_network, pos, nodelist=customers, node_color='skyblue', node_size=200, label='Customer')
    nx.draw_networkx_nodes(road_network, pos, nodelist=stations, node_color='lightgreen', node_shape='p', node_size=350,
                           label='Station')
    nx.draw_networkx_labels(road_network, pos, font_size=8, ax=ax)

    # Draw the vehicle routes
    if solution_routes:
        route_colors = plt.cm.get_cmap('gist_rainbow', len(solution_routes))
        for i, (vehicle_id, route) in enumerate(solution_routes.items()):
            if not route or len(route) < 2:
                continue

            # Convert route names to node IDs for plotting
            route_node_ids = [node for name in route for node, info in node_info.items() if info.get('name') == name]

            if len(route_node_ids) >= 2:
                route_edges = list(zip(route_node_ids[:-1], route_node_ids[1:]))
                nx.draw_networkx_edges(road_network, pos, edgelist=route_edges,
                                       width=2.5, alpha=0.9, edge_color=route_colors(i),
                                       label=vehicle_id, ax=ax, connectionstyle='arc3,rad=0.1')

    ax.set_title(title, fontsize=24, fontweight='bold')

    # Create a clean legend
    handles, labels = ax.get_legend_handles_labels()
    # Manually add node type legend entries
    handles.extend([
        plt.Line2D([0], [0], marker='s', color='w', label='Depot', markerfacecolor='gold', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Customer', markerfacecolor='skyblue', markersize=10),
        plt.Line2D([0], [0], marker='p', color='w', label='Station', markerfacecolor='lightgreen', markersize=10)
    ])
    ax.legend(handles=handles, title="Legend")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/fleet_routing_plan_detailed.pdf', format='pdf', bbox_inches='tight')
    plt.close()


# --- 2. Case 1: Scheduled vs. Unscheduled Comparison Visualizations ---
def plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir):
    """
    Generates all comparison visualizations for Case 1.
    Accepts real statistics dictionaries from the simulation.
    """
    print("\n" + "=" * 20 + " Visualizing Case 1: Scheduled vs. Unscheduled " + "=" * 20)
    os.makedirs(output_dir, exist_ok=True)

    # --- Economic Cost Comparison (Bar Chart) ---
    costs = {
        'Scheduled Fleet': scheduled_stats['total_cost'],
        'Unscheduled Fleet': unscheduled_stats['total_cost']
    }
    df_cost = pd.DataFrame({
        "Scenario": list(costs.keys()),
        "Value": list(costs.values())
    })
    plt.figure(figsize=(8, 7))
    ax = sns.barplot(
        data=df_cost,
        x="Scenario",
        y="Value",
        hue="Scenario",                     # 修复 FutureWarning：配合 palette 使用
        palette=['#31a354', '#a1d99b'],
        legend=False
    )
    ax.set_title('Case 1: Economic Cost Comparison', fontsize=16)
    ax.set_ylabel('Total Daily Cost (Yuan)')
    # 数值标签
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f"{val:.1f}", (p.get_x() + p.get_width()/2, val),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case1_cost_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # --- Energy Flow Comparison (Stacked Area Chart) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    for ax, stats, title in [(ax1, scheduled_stats, 'Scheduled Fleet'), (ax2, unscheduled_stats, 'Unscheduled Fleet')]:
        df_energy = stats['energy_flows']
        ax.stackplot(df_energy.index, df_energy['grid_power'], df_energy['pv_power'], df_energy['bess_discharge'],
                     labels=['Grid Input', 'PV Output', 'BESS Discharge'],
                     colors=['salmon', 'gold', 'lightgreen'])
        ax.plot(df_energy['total_demand'], color='black', linestyle='--', label='Total Demand (HDT+EV)')
        ax.set_title(f'Energy Flow: {title}', fontsize=14)
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper left')
    plt.xlabel('Time (Hour of Day)')
    fig.suptitle('Case 1: Station Energy Flow Comparison', fontsize=18, y=0.99)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case1_energy_flow_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # --- Grid Peak Shaving Comparison (Line Chart) ---
    plt.figure(figsize=(15, 7))
    plt.plot(scheduled_stats['energy_flows'].index, scheduled_stats['energy_flows']['grid_power'],
             label='Scheduled Grid Load', color='#31a354', linewidth=2)
    plt.plot(unscheduled_stats['energy_flows'].index, unscheduled_stats['energy_flows']['grid_power'],
             label='Unscheduled Grid Load', color='#a1d99b', linestyle='--')
    plt.axhline(0, color='gray', linestyle=':')
    plt.title('Case 1: Grid Peak Shaving Comparison', fontsize=16)
    plt.xlabel('Time (Hour of Day)')
    plt.ylabel('Power Drawn from Grid (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case1_peak_shaving_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # --- Delivery & Queue Time (Printed Table) ---
    df_times = pd.DataFrame({
        'Metric': ['Avg. Delivery Time (h)', 'Avg. Queue Time (min)', 'Total Wait Time (h)'],
        'Scheduled Fleet': [f"{scheduled_stats.get('avg_delivery_time', 0.0):.2f}",
                            f"{scheduled_stats.get('avg_queue_time', 0.0):.2f}",
                            f"{scheduled_stats.get('total_wait_time', 0.0):.2f}"],
        'Unscheduled Fleet': [f"{unscheduled_stats.get('avg_delivery_time', 0.0):.2f}",
                              f"{unscheduled_stats.get('avg_queue_time', 0.0):.2f}",
                              f"{unscheduled_stats.get('total_wait_time', 0.0):.2f}"]
    }).set_index('Metric')
    print("\n--- Case 1: Time Performance Comparison ---")
    print(df_times.to_string())


# --- 3. Case 2: EV + EHDT Arrival Heatmap ---
def plot_case2_heatmap(arrival_matrix, output_dir):
    """
    Generates the station arrival heatmap for Case 2.
    Accepts a DataFrame of arrival data.
    """
    print("\n" + "=" * 20 + " Visualizing Case 2: Station Arrival Heatmap " + "=" * 20)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(20, 10))
    sns.heatmap(arrival_matrix, cmap='YlOrRd', linewidths=.5, annot=True, fmt=".0f")
    plt.title('Case 2: Station Vehicle Arrivals (EV + EHDT)', fontsize=16)
    plt.xlabel('Hour of Day')
    plt.ylabel('Station ID')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case2_arrival_heatmap.pdf", format='pdf', bbox_inches='tight')
    plt.close()


# --- 4. Case 3: Grid Service Strategy Comparison ---
def plot_case3_comparison(stats_dict, output_dir):
    """
    Generates the V2G strategy comparison charts for Case 3.
    Accepts a dictionary of statistics for each strategy.
    """
    print("\n" + "=" * 20 + " Visualizing Case 3: Grid Service Strategies " + "=" * 20)
    os.makedirs(output_dir, exist_ok=True)

    # --- Economic Cost Comparison ---
    costs = {name: stats['total_cost'] for name, stats in stats_dict.items()}
    df_cost = pd.DataFrame({
        "Strategy": list(costs.keys()),
        "Value": list(costs.values())
    })
    plt.figure(figsize=(12, 8))
    # 动态生成与策略数相等的调色板
    base_palette = ['#2c7fb8', '#7fcdbb', '#edf8b1', '#7bccc4', '#a1dab4', '#41b6c4', '#c7e9b4']
    palette = base_palette[:len(df_cost)]
    ax = sns.barplot(
        data=df_cost,
        x="Strategy",
        y="Value",
        hue="Strategy",                  # 修复 FutureWarning
        palette=palette,
        legend=False
    )
    ax.set_title('Case 3: Economic Cost of Different Battery Strategies', fontsize=16)
    ax.set_ylabel('Total Daily Cost (Yuan)')
    plt.xticks(rotation=15, ha='right')
    # 数值标签
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f"{val:.1f}", (p.get_x() + p.get_width()/2, val),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case3_cost_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # --- Peak Shaving Comparison ---
    plt.figure(figsize=(15, 7))
    line_colors = ['#2c7fb8', '#7fcdbb', '#edf8b1', '#7bccc4', '#a1dab4', '#41b6c4', '#c7e9b4']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (5, 2))]
    for i, (name, stats) in enumerate(stats_dict.items()):
        color = line_colors[i % len(line_colors)]
        ls = linestyles[i % len(linestyles)]
        plt.plot(stats['grid_load'].index, stats['grid_load'],
                 label=name, color=color, linestyle=ls, linewidth=2.5)
    plt.axhline(0, color='gray', linestyle=':')
    plt.title('Case 3: Grid Peak Shaving Comparison', fontsize=16)
    plt.xlabel('Time (Hour of Day)')
    plt.ylabel('Power Drawn from Grid (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case3_peak_shaving_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()

def _format_dataframe_for_print(df, float_cols=None, digits=2):
    if df.empty:
        return ""
    formatters = {}
    if float_cols:
        for col in float_cols:
            if col in df.columns:
                formatters[col] = lambda x, d=digits: f"{x:.{d}f}"
    return df.to_string(index=False, formatters=formatters)


def print_vehicle_operation_details(vehicle_event_log, output_dir=None, max_vehicles=10):
    """Prints a detailed log of vehicle movements and optionally saves it as CSV."""
    print("\n" + "=" * 30 + " 车辆执行动作明细 " + "=" * 30)
    if not vehicle_event_log:
        print("暂无车辆动作记录。")
        return

    df_events = pd.DataFrame(vehicle_event_log)
    df_events = df_events.sort_values(by=['vehicle_id', 'depart_time']).reset_index(drop=True)

    if max_vehicles is not None:
        selected = df_events['vehicle_id'].dropna().unique()[:max_vehicles]
        df_display = df_events[df_events['vehicle_id'].isin(selected)]
        if len(selected) < len(df_events['vehicle_id'].dropna().unique()):
            print(f"显示前 {len(selected)} 辆车的动作记录 (共 {len(df_events['vehicle_id'].dropna().unique())} 辆车)")
    else:
        df_display = df_events

    columns = ['vehicle_id', 'depart_time', 'arrive_time', 'from_node', 'to_node',
               'distance_km', 'travel_time_h', 'soc_start_kwh', 'soc_end_kwh',
               'load_start_ton', 'load_end_ton', 'delivered_amount_ton',
               'task_id', 'delivered_customer', 'path_nodes']
    rename_map = {
        'vehicle_id': '车辆',
        'depart_time': '出发时间(h)',
        'arrive_time': '到达时间(h)',
        'from_node': '起点',
        'to_node': '终点',
        'distance_km': '里程(km)',
        'travel_time_h': '行驶时间(h)',
        'soc_start_kwh': '出发SOC(kWh)',
        'soc_end_kwh': '到达SOC(kWh)',
        'load_start_ton': '出发载重(t)',
        'load_end_ton': '到达载重(t)',
        'delivered_amount_ton': '卸货量(t)',
        'task_id': '任务ID',
        'delivered_customer': '服务客户',
        'path_nodes': '行驶路径节点序列'
    }
    float_cols = ['出发时间(h)', '到达时间(h)', '里程(km)', '行驶时间(h)',
                  '出发SOC(kWh)', '到达SOC(kWh)', '出发载重(t)', '到达载重(t)', '卸货量(t)']

    df_print = df_display[[c for c in columns if c in df_display.columns]].rename(columns=rename_map)
    print(_format_dataframe_for_print(df_print, float_cols=float_cols, digits=2))

    if output_dir:
        csv_path = os.path.join(output_dir, 'vehicle_action_log.csv')
        df_events.to_csv(csv_path, index=False)
        print(f"车辆动作日志已保存至: {csv_path}")


def print_location_and_task_overview(data, task_sequences, output_dir=None):
    """Outputs depot/customer positions and task assignments."""
    print("\n" + "=" * 30 + " 场景基础信息总览 " + "=" * 30)

    locations = data.get('locations', {})
    depots, customers = [], []
    for name, info in locations.items():
        record = {
            '名称': name,
            '路网节点': info.get('node_id'),
            '类型': info.get('type'),
            'X坐标': info.get('pos', (None, None))[0],
            'Y坐标': info.get('pos', (None, None))[1]
        }
        if info.get('type') == 'Depot':
            depots.append(record)
        elif info.get('type') == 'Customer':
            customers.append(record)

    depot_df = pd.DataFrame(depots).sort_values('名称') if depots else pd.DataFrame(columns=['名称'])
    customer_df = pd.DataFrame(customers).sort_values('名称') if customers else pd.DataFrame(columns=['名称'])

    if not depot_df.empty:
        print("\n--- 仓库节点 ---")
        print(_format_dataframe_for_print(depot_df, float_cols=['X坐标', 'Y坐标']))
    else:
        print("未生成仓库节点数据。")

    if not customer_df.empty:
        print("\n--- 客户节点 ---")
        print(_format_dataframe_for_print(customer_df, float_cols=['X坐标', 'Y坐标']))
    else:
        print("未生成客户节点数据。")

    tasks = data.get('tasks', {})
    task_rows = []
    for vid, task_list in task_sequences.items():
        for order, task_id in enumerate(task_list, start=1):
            info = tasks.get(task_id, {})
            task_rows.append({
                '任务ID': task_id,
                '车辆': vid,
                '序号': order,
                '客户': info.get('delivery_to'),
                '需求量(t)': info.get('demand'),
                '交付截止时间(h)': info.get('due_time'),
                '出发仓库': info.get('depot')
            })

    tasks_df = pd.DataFrame(task_rows).sort_values(['车辆', '序号']) if task_rows else pd.DataFrame(columns=['任务ID'])

    if not tasks_df.empty:
        print("\n--- 配送任务列表 ---")
        print(_format_dataframe_for_print(tasks_df, float_cols=['需求量(t)', '交付截止时间(h)']))
    else:
        print("暂无配送任务数据。")

    if output_dir:
        if not depot_df.empty:
            depot_path = os.path.join(output_dir, 'scenario_depots.csv')
            depot_df.to_csv(depot_path, index=False)
            print(f"仓库节点信息已保存至: {depot_path}")
        if not customer_df.empty:
            customer_path = os.path.join(output_dir, 'scenario_customers.csv')
            customer_df.to_csv(customer_path, index=False)
            print(f"客户节点信息已保存至: {customer_path}")
        if not tasks_df.empty:
            task_path = os.path.join(output_dir, 'scenario_tasks.csv')
            tasks_df.to_csv(task_path, index=False)
            print(f"配送任务列表已保存至: {task_path}")


def print_customer_service_summary(customer_df, output_dir=None):
    """Prints how many vehicles served each customer and the delivered quantities."""
    print("\n" + "=" * 30 + " 客户服务统计 " + "=" * 30)
    if customer_df is None or customer_df.empty:
        print("暂无客户服务统计数据。")
        return

    rename_map = {
        'customer': '客户',
        'node_id': '路网节点',
        'pos_x': 'X坐标',
        'pos_y': 'Y坐标',
        'vehicles_served': '参与车辆',
        'num_vehicles_served': '车辆数量',
        'tasks_delivered': '相关任务',
        'total_delivered_ton': '累计卸货量(t)'
    }
    df_print = customer_df.rename(columns=rename_map)
    print(_format_dataframe_for_print(df_print, float_cols=['X坐标', 'Y坐标', '累计卸货量(t)']))

    if output_dir:
        summary_path = os.path.join(output_dir, 'customer_service_summary.csv')
        customer_df.to_csv(summary_path, index=False)
        print(f"客户服务统计已保存至: {summary_path}")