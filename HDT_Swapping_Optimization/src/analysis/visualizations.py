import os
from collections import defaultdict

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
def plot_vehicle_routes_on_network(data, vehicle_event_log, output_dir, title="Vehicle Route Trajectories", strategy_tag="scheduled"):
    """Visualize executed vehicle routes together with depot and station locations."""

    road_network = data.get('traffic_graph')
    if road_network is None:
        print("未找到路网图数据，跳过车辆轨迹可视化。")
        return

    if not vehicle_event_log:
        print("暂无车辆动作记录，跳过车辆轨迹可视化。")
        return

    pos = nx.get_node_attributes(road_network, 'pos')
    if not pos:
        pos = nx.spring_layout(road_network, seed=42)

    locations = data.get('locations', {})
    node_id_by_name = {name: info.get('node_id') for name, info in locations.items()
                       if info.get('node_id') is not None}
    node_type_by_id = {info.get('node_id'): info.get('type') for info in locations.values()
                       if info.get('node_id') is not None}

    depot_nodes = [info.get('node_id') for info in locations.values()
                   if info.get('type') == 'Depot' and info.get('node_id') is not None]
    customer_nodes = [info.get('node_id') for info in locations.values()
                      if info.get('type') == 'Customer' and info.get('node_id') is not None]
    station_nodes = [info.get('node_id') for info in locations.values()
                     if info.get('type') == 'SwapStation' and info.get('node_id') is not None]

    travelled_edges = defaultdict(list)
    visited_nodes = defaultdict(set)

    for event in vehicle_event_log:
        vid = event.get('vehicle_id')
        if not vid:
            continue

        path_nodes = event.get('path_nodes')
        node_sequence = []

        if isinstance(path_nodes, str) and path_nodes.strip():
            raw_tokens = [token for token in path_nodes.replace('->', ' ').split() if token]
            try:
                node_sequence = [int(float(token)) for token in raw_tokens]
            except ValueError:
                node_sequence = []
        elif isinstance(path_nodes, (list, tuple)):
            node_sequence = list(path_nodes)

        if len(node_sequence) < 2:
            from_node = node_id_by_name.get(event.get('from_node'))
            to_node = node_id_by_name.get(event.get('to_node'))
            if from_node is not None and to_node is not None:
                node_sequence = [from_node, to_node]

        if len(node_sequence) < 2:
            continue

        cleaned_sequence = []
        for node in node_sequence:
            if node in road_network:
                cleaned_sequence.append(node)

        if len(cleaned_sequence) < 2:
            continue

        edge_list = list(zip(cleaned_sequence[:-1], cleaned_sequence[1:]))
        if edge_list:
            travelled_edges[vid].extend(edge_list)
            visited_nodes[vid].update(cleaned_sequence)

    active_vehicles = [vid for vid, edges in travelled_edges.items() if edges]
    if not active_vehicles:
        print("未检测到实际的车辆行驶轨迹，跳过绘图。")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(18, 14))

    nx.draw_networkx_edges(road_network, pos, edge_color='#d0d0d0', alpha=0.25, width=0.8, ax=ax)

    other_nodes = [node for node in road_network.nodes if node not in node_type_by_id]
    if other_nodes:
        nx.draw_networkx_nodes(road_network, pos, nodelist=other_nodes, node_color='#f0f0f0',
                               node_size=20, ax=ax, alpha=0.6)

    if depot_nodes:
        nx.draw_networkx_nodes(road_network, pos, nodelist=depot_nodes, node_color='#ffcc00',
                               node_shape='s', node_size=350, ax=ax, label='Depot')
    if customer_nodes:
        nx.draw_networkx_nodes(road_network, pos, nodelist=customer_nodes, node_color='#66b3ff',
                               node_size=120, ax=ax, label='Customer')
    if station_nodes:
        nx.draw_networkx_nodes(road_network, pos, nodelist=station_nodes, node_color='#8dd3c7',
                               node_shape='p', node_size=260, ax=ax, label='Swap Station')

    cmap = plt.cm.get_cmap('tab20', max(len(active_vehicles), 1))
    for idx, vid in enumerate(sorted(active_vehicles)):
        edges = travelled_edges[vid]
        if not edges:
            continue
        color = cmap(idx)
        nx.draw_networkx_edges(road_network, pos, edgelist=edges, edge_color=[color], width=2.5,
                               ax=ax, label=f'{vid} route', arrows=False)

        nodes_to_mark = sorted(visited_nodes[vid])
        if nodes_to_mark:
            nx.draw_networkx_nodes(road_network, pos, nodelist=nodes_to_mark, node_size=60,
                                   node_color=[color], alpha=0.8, ax=ax)

    for name, info in locations.items():
        node_id = info.get('node_id')
        if node_id in pos:
            xy = pos[node_id]
            ax.text(xy[0] + 0.005, xy[1] + 0.005, name, fontsize=8, ha='left', va='bottom')

    ax.set_title(f"{title}", fontsize=18)
    ax.axis('off')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    filename = f'vehicle_routes_{strategy_tag}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Vehicle route plot saved to: {os.path.join(output_dir, filename)}")


def plot_full_road_network(data, output_dir, title="Complete Road Network with Key Facilities"):
    """Plot the entire road network with node identifiers and key facility highlights."""

    road_network = data.get('traffic_graph')
    if road_network is None:
        print("未找到路网图数据，跳过完整路网绘制。")
        return

    pos = nx.get_node_attributes(road_network, 'pos')
    if not pos:
        pos = nx.spring_layout(road_network, seed=42)

    locations = data.get('locations', {})
    node_to_type = {info.get('node_id'): info.get('type') for info in locations.values() if info.get('node_id') is not None}
    node_to_name = {info.get('node_id'): name for name, info in locations.items() if info.get('node_id') is not None}

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(18, 14))

    nx.draw_networkx_edges(road_network, pos, edge_color='#d0d0d0', alpha=0.6, width=0.8, ax=ax)

    all_nodes = list(road_network.nodes)
    nx.draw_networkx_nodes(road_network, pos, nodelist=all_nodes, node_color='#b0c4de', node_size=80, alpha=0.85, ax=ax)

    node_labels = {node: str(node) for node in all_nodes}
    nx.draw_networkx_labels(road_network, pos, labels=node_labels, font_size=6, ax=ax)

    depot_nodes = [node for node, ntype in node_to_type.items() if ntype == 'Depot']
    customer_nodes = [node for node, ntype in node_to_type.items() if ntype == 'Customer']
    station_nodes = [node for node, ntype in node_to_type.items() if ntype == 'SwapStation']

    if depot_nodes:
        nx.draw_networkx_nodes(road_network, pos, nodelist=depot_nodes, node_color='#ffcc00',
                               node_shape='s', node_size=260, edgecolors='black', linewidths=0.8, ax=ax, label='Depot')
    if customer_nodes:
        nx.draw_networkx_nodes(road_network, pos, nodelist=customer_nodes, node_color='#66b3ff',
                               node_size=150, edgecolors='black', linewidths=0.6, ax=ax, label='Customer')
    if station_nodes:
        nx.draw_networkx_nodes(road_network, pos, nodelist=station_nodes, node_color='#8dd3c7',
                               node_shape='p', node_size=220, edgecolors='black', linewidths=0.6, ax=ax, label='Swap Station')

    for node_id, name in node_to_name.items():
        if node_id in pos:
            xy = pos[node_id]
            ax.text(xy[0] + 0.005, xy[1] + 0.005, name, fontsize=7, ha='left', va='bottom', color='#333333')

    ax.set_title(title, fontsize=18)
    ax.axis('off')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'road_network_full.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Full road network plot saved to: {filepath}")

# --- 2. Case 1: Scheduled vs. Unscheduled Comparison Visualizations ---
def plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir):
    """
    Generates all comparison visualizations for Case 1.
    Accepts real statistics dictionaries from the simulation.
    """
    print("\n" + "=" * 20 + " Visualizing Case 1: Scheduled vs. Unscheduled " + "=" * 20)
    os.makedirs(output_dir, exist_ok=True)

    scenario_stats = {
        'Scheduled Fleet': scheduled_stats,
        'Unscheduled Fleet': unscheduled_stats,
    }

    # Backwards compatibility: prior revisions of this helper referenced a
    # ``stats_dict`` local when looping over the scenarios.  Providing this
    # alias ensures older inline snippets (or stale bytecode) still find the
    # expected name and prevents ``NameError`` crashes reported by users.
    stats_dict = scenario_stats

    # --- Economic Cost Comparison (Bar Chart) ---
    df_cost = pd.DataFrame({
        "Scenario": list(scenario_stats.keys()),
        "Value": [stats['total_cost'] for stats in scenario_stats.values()],
    })
    plt.figure(figsize=(8, 7))
    ax = sns.barplot(
        data=df_cost,
        x="Scenario",
        y="Value",
        hue="Scenario",
        palette=['#31a354', '#a1d99b'],
        legend=False,
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
    colors = ['#31a354', '#a1d99b']
    linestyles = ['-', '--']
    for (title, stats), color, ls in zip(scenario_stats.items(), colors, linestyles):
        plt.plot(stats['energy_flows'].index, stats['energy_flows']['grid_power'],
                 label=f'{title} Grid Load', color=color, linestyle=ls, linewidth=2)
    plt.axhline(0, color='gray', linestyle=':')
    plt.title('Case 1: Grid Peak Shaving Comparison', fontsize=16)
    plt.xlabel('Time (Hour of Day)')
    plt.ylabel('Power Drawn from Grid (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/case1_peak_shaving_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # --- Delivery & Queue Time (Printed Table) ---
    metrics = [
        ('Avg. Delivery Time (h)', 'avg_delivery_time'),
        ('Avg. Queue Time (min)', 'avg_queue_time'),
        ('Total Wait Time (h)', 'total_wait_time'),
    ]
    df_times = pd.DataFrame(
        {
            'Metric': [label for label, _ in metrics],
            **{
                title: [f"{stats.get(key, 0.0):.2f}" for _, key in metrics]
                for title, stats in scenario_stats.items()
            },
        }
    ).set_index('Metric')
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
            hue="Strategy",
            palette=palette,
            legend=False,
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
                formatters[col] = lambda x, d=digits: "--" if pd.isna(x) else f"{x:.{d}f}"
    return df.to_string(index=False, formatters=formatters, na_rep='--')


def print_vehicle_operation_details(vehicle_event_log, output_dir=None, max_vehicles=10, title=None, file_tag=None):
    """Prints a detailed log of vehicle movements and optionally saves it as CSV."""
    header = title or "车辆执行动作明细"
    print("\n" + "=" * 30 + f" {header} " + "=" * 30)
    if not vehicle_event_log:
        print("暂无车辆动作记录。")
        return

    df_events = pd.DataFrame(vehicle_event_log)
    df_events = df_events.sort_values(by=['vehicle_id', 'depart_time']).reset_index(drop=True)

    active_ids = df_events['vehicle_id'].dropna().unique()
    delivery_counts = (df_events[df_events['task_id'].notna()]
                       .groupby('vehicle_id')['task_id']
                       .count()
                       .sort_values(ascending=False))
    ordered_ids = list(delivery_counts.index)
    for vid in active_ids:
        if vid not in ordered_ids:
            ordered_ids.append(vid)

    if not ordered_ids:
        print("暂无车辆动作记录。")
        return

    if max_vehicles is not None:
        selected = ordered_ids[:max_vehicles]
        if len(ordered_ids) > len(selected):
            print(f"按任务数量降序展示前 {len(selected)} 辆车的动作记录 (共 {len(ordered_ids)} 辆车执行过动作)")
    else:
        selected = ordered_ids

    df_display = df_events[df_events['vehicle_id'].isin(selected)].copy()
    df_display['vehicle_id'] = pd.Categorical(df_display['vehicle_id'], categories=selected, ordered=True)
    df_display = df_display.sort_values(by=['vehicle_id', 'depart_time']).reset_index(drop=True)

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
        filename = 'vehicle_action_log.csv'
        if file_tag:
            filename = f'vehicle_action_log_{file_tag}.csv'
        csv_path = os.path.join(output_dir, filename)
        df_events.to_csv(csv_path, index=False)
        print(f"车辆动作日志已保存至: {csv_path}")


def print_location_and_task_overview(data, task_sequences, output_dir=None):
    """Outputs depot/customer positions and task assignments."""
    print("\n" + "=" * 30 + " 场景基础信息总览 " + "=" * 30)

    locations = data.get('locations', {})
    depots, customers, stations = [], [], []
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
        elif info.get('type') == 'SwapStation':
            stations.append(record)

    depot_df = pd.DataFrame(depots).sort_values('名称') if depots else pd.DataFrame(columns=['名称'])
    customer_df = pd.DataFrame(customers).sort_values('名称') if customers else pd.DataFrame(columns=['名称'])
    station_df = pd.DataFrame(stations).sort_values('名称') if stations else pd.DataFrame(columns=['名称'])

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

    if not station_df.empty:
        print("\n--- 换电站节点 ---")
        print(_format_dataframe_for_print(station_df, float_cols=['X坐标', 'Y坐标']))
    else:
        print("未生成换电站节点数据。")

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

    vehicle_rows = []
    vehicles_info = data.get('vehicles', {})
    for vid, info in vehicles_info.items():
        assigned_tasks = task_sequences.get(vid, [])
        total_demand = sum(tasks.get(tid, {}).get('demand', 0.0) for tid in assigned_tasks)
        vehicle_rows.append({
            '车辆': vid,
            '所属仓库': info.get('depot_id'),
            '任务数量': len(assigned_tasks),
            '累计需求(t)': total_demand
        })

    vehicle_df = pd.DataFrame(vehicle_rows).sort_values('车辆') if vehicle_rows else pd.DataFrame(columns=['车辆'])

    if not vehicle_df.empty:
        print("\n--- 车辆任务概览 ---")
        print(_format_dataframe_for_print(vehicle_df, float_cols=['累计需求(t)']))

    if output_dir:
        if not depot_df.empty:
            depot_path = os.path.join(output_dir, 'scenario_depots.csv')
            depot_df.to_csv(depot_path, index=False)
            print(f"仓库节点信息已保存至: {depot_path}")
        if not customer_df.empty:
            customer_path = os.path.join(output_dir, 'scenario_customers.csv')
            customer_df.to_csv(customer_path, index=False)
            print(f"客户节点信息已保存至: {customer_path}")
        if not station_df.empty:
            station_path = os.path.join(output_dir, 'scenario_stations.csv')
            station_df.to_csv(station_path, index=False)
            print(f"换电站节点信息已保存至: {station_path}")
        if not tasks_df.empty:
            task_path = os.path.join(output_dir, 'scenario_tasks.csv')
            tasks_df.to_csv(task_path, index=False)
            print(f"配送任务列表已保存至: {task_path}")
        if not vehicle_df.empty:
            vehicle_task_path = os.path.join(output_dir, 'scenario_vehicle_tasks.csv')
            vehicle_df.to_csv(vehicle_task_path, index=False)
            print(f"车辆任务概览已保存至: {vehicle_task_path}")


def print_customer_service_summary(customer_df, output_dir=None, title=None, file_tag=None):
    """Prints how many vehicles served each customer and the delivered quantities."""
    header = title or "客户服务统计"
    print("\n" + "=" * 30 + f" {header} " + "=" * 30)
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
        filename = 'customer_service_summary.csv'
        if file_tag:
            filename = f'customer_service_summary_{file_tag}.csv'
        summary_path = os.path.join(output_dir, filename)
        customer_df.to_csv(summary_path, index=False)
        print(f"客户服务统计已保存至: {summary_path}")


def print_vehicle_operation_summary(data, vehicle_summary_df, output_dir=None, title=None, file_tag=None):
    """Prints aggregated per-vehicle statistics to clarify fleet workload."""
    header = title or "车辆运营总览"
    print("\n" + "=" * 30 + f" {header} " + "=" * 30)

    vehicles_info = data.get('vehicles', {})
    base_rows = [{
        'vehicle_id': vid,
        'home_depot': info.get('depot_id')
    } for vid, info in vehicles_info.items()]
    base_df = pd.DataFrame(base_rows)

    if vehicle_summary_df is None or vehicle_summary_df.empty:
        summary_df = base_df.copy()
        summary_df['total_tasks'] = 0
        summary_df['unique_customers'] = 0
        summary_df['total_delivered_ton'] = 0.0
        summary_df['total_distance_km'] = 0.0
        summary_df['total_travel_time_h'] = 0.0
        summary_df['total_energy_kwh'] = 0.0
        summary_df['earliest_depart_h'] = np.nan
        summary_df['latest_return_h'] = np.nan
        summary_df['min_soc_kwh'] = [vehicles_info.get(row['vehicle_id'], {}).get('initial_soc', np.nan)
                                      for _, row in summary_df.iterrows()]
        summary_df['end_soc_kwh'] = summary_df['min_soc_kwh']
        summary_df['delivery_details'] = ''
    else:
        summary_df = vehicle_summary_df.copy()
        if not summary_df.empty and 'home_depot' not in summary_df.columns:
            summary_df = base_df.merge(summary_df, on='vehicle_id', how='left')

    if base_df.empty:
        if summary_df.empty:
            print("暂无车辆信息。")
            return
    else:
        if summary_df.empty:
            summary_df = base_df.copy()

    defaults = {
        'total_tasks': 0,
        'unique_customers': 0,
        'total_delivered_ton': 0.0,
        'total_distance_km': 0.0,
        'total_travel_time_h': 0.0,
        'total_energy_kwh': 0.0,
        'earliest_depart_h': np.nan,
        'latest_return_h': np.nan,
        'min_soc_kwh': np.nan,
        'end_soc_kwh': np.nan,
        'delivery_details': ''
    }

    for col, default in defaults.items():
        if col not in summary_df.columns:
            summary_df[col] = default

    initial_soc_map = {vid: info.get('initial_soc', np.nan) for vid, info in vehicles_info.items()}
    summary_df['min_soc_kwh'] = summary_df['min_soc_kwh'].fillna(summary_df['vehicle_id'].map(initial_soc_map))
    summary_df['end_soc_kwh'] = summary_df['end_soc_kwh'].fillna(summary_df['vehicle_id'].map(initial_soc_map))

    summary_df['total_tasks'] = summary_df['total_tasks'].fillna(0).astype(int)
    summary_df['unique_customers'] = summary_df['unique_customers'].fillna(0).astype(int)
    summary_df['total_delivered_ton'] = summary_df['total_delivered_ton'].fillna(0.0)
    summary_df['total_distance_km'] = summary_df['total_distance_km'].fillna(0.0)
    summary_df['total_travel_time_h'] = summary_df['total_travel_time_h'].fillna(0.0)
    summary_df['total_energy_kwh'] = summary_df['total_energy_kwh'].fillna(0.0)
    summary_df['delivery_details'] = summary_df['delivery_details'].fillna('')

    summary_df = summary_df.sort_values(by=['total_tasks', 'total_distance_km'], ascending=False)

    rename_map = {
        'vehicle_id': '车辆',
        'home_depot': '所属仓库',
        'total_tasks': '任务数量',
        'unique_customers': '服务客户数',
        'total_delivered_ton': '累计卸货量(t)',
        'total_distance_km': '累计行驶距离(km)',
        'total_travel_time_h': '累计行驶时间(h)',
        'total_energy_kwh': '能耗(kWh)',
        'earliest_depart_h': '最早出发(h)',
        'latest_return_h': '最晚返回(h)',
        'min_soc_kwh': '最低SOC(kWh)',
        'end_soc_kwh': '返回SOC(kWh)',
        'delivery_details': '客户任务汇总'
    }

    float_cols = ['累计卸货量(t)', '累计行驶距离(km)', '累计行驶时间(h)', '能耗(kWh)',
                  '最早出发(h)', '最晚返回(h)', '最低SOC(kWh)', '返回SOC(kWh)']

    display_df = summary_df.rename(columns=rename_map)
    print(_format_dataframe_for_print(display_df, float_cols=float_cols))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = 'vehicle_operation_summary.csv'
        if file_tag:
            filename = f'vehicle_operation_summary_{file_tag}.csv'
        summary_path = os.path.join(output_dir, filename)
        summary_df.to_csv(summary_path, index=False)
        print(f"车辆运营总览已保存至: {summary_path}")