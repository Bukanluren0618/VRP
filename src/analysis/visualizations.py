# src/analysis/visualizations.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx

# --- 全局绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Arial")


# --- 1. 路网与车辆路径可视化 ---
def plot_road_network_with_routes(road_network, solution_routes, output_dir, title="Fleet Routing Plan"):
    """
    可视化路网，并在其上绘制优化后的车辆路径。
    """
    print("--- Visualizing Fleet Routing Plan ---")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(20, 16))

    pos = nx.get_node_attributes(road_network, 'pos')
    if not pos:
        pos = nx.spring_layout(road_network, seed=42)

    # 假设节点信息存储在 'info' 属性中
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

            # 从节点名称映射回节点ID
            route_node_ids = []
            for name in route:
                # 这个查找逻辑需要与loader中的数据结构匹配
                found_node = False
                for node_id, info in node_info.items():
                    if info.get('name') == name:
                        route_node_ids.append(node_id)
                        found_node = True
                        break
                if not found_node:
                    # 如果在info中找不到，可能名称就是ID
                    if name in road_network:
                        route_node_ids.append(name)

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
    plt.savefig(f'{output_dir}/fleet_routing_plan_detailed.pdf', format='pdf')
    # plt.show() # 在自动化脚本中通常注释掉show()
    plt.close()


# --- 2. 案例1: 调度 vs. 非调度对比可视化 ---
def plot_case1_comparison(scheduled_stats, unscheduled_stats, output_dir):
    """
    生成案例1的所有对比可视化图表。
    """
    print("\n" + "=" * 20 + " Visualizing Case 1: Scheduled vs. Unscheduled " + "=" * 20)

    # --- 经济成本对比 (条形图) ---
    plt.figure(figsize=(8, 7))
    costs = {'Scheduled Fleet': scheduled_stats['total_cost'], 'Unscheduled Fleet': unscheduled_stats['total_cost']}
    # --- 【兼容性修复】 ---
    # 修复了未来版本中 `palette` 的使用警告
    sns.barplot(x=list(costs.keys()), y=list(costs.values()), palette=['#31a354', '#a1d99b'], hue=list(costs.keys()),
                legend=False)
    plt.title('Case 1: Economic Cost Comparison', fontsize=16)
    plt.ylabel('Total Daily Cost (Yuan)')
    plt.savefig(f"{output_dir}/case1_cost_comparison.pdf", format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    # --- 能源流对比 (堆叠面积图) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    for ax, stats, title in [(ax1, scheduled_stats, 'Scheduled Fleet'), (ax2, unscheduled_stats, 'Unscheduled Fleet')]:
        df_energy = stats['energy_flows']

        # 将Timedelta索引转换为小时数值用于绘图
        x_values = df_energy.index.total_seconds() / 3600

        ax.stackplot(x_values, df_energy['grid_power'], df_energy['pv_power'], df_energy['bess_discharge'],
                     labels=['Grid Input', 'PV Output', 'BESS Discharge'],
                     colors=['salmon', 'gold', 'lightgreen'])
        ax.plot(x_values, df_energy['total_demand'], color='black', linestyle='--', label='Total Demand (HDT+EV)')
        ax.set_title(f'Energy Flow: {title}', fontsize=14)
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper left')
    plt.xlabel('Time (Hour of Day)')
    fig.suptitle('Case 1: Station Energy Flow Comparison', fontsize=18, y=0.99)
    plt.savefig(f"{output_dir}/case1_energy_flow_comparison.pdf", format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    # --- 电网削峰对比 (折线图) ---
    plt.figure(figsize=(15, 7))
    sched_x = scheduled_stats['energy_flows'].index.total_seconds() / 3600
    unsched_x = unscheduled_stats['energy_flows'].index.total_seconds() / 3600
    plt.plot(sched_x, scheduled_stats['energy_flows']['grid_power'], label='Scheduled Grid Load', color='#31a354',
             linewidth=2)
    plt.plot(unsched_x, unscheduled_stats['energy_flows']['grid_power'], label='Unscheduled Grid Load', color='#a1d99b',
             linestyle='--')
    plt.axhline(0, color='gray', linestyle=':')
    plt.title('Case 1: Grid Peak Shaving Comparison', fontsize=16)
    plt.xlabel('Time (Hour of Day)')
    plt.ylabel('Power Drawn from Grid (kW)')
    plt.legend()
    plt.savefig(f"{output_dir}/case1_peak_shaving_comparison.pdf", format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    # --- 交付与排队时间 (打印表格) ---
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


# --- 3. 案例2: EV + EHDT 到达热力图 ---
def plot_case2_heatmap(arrival_matrix, output_dir):
    """为案例2生成换电站到达热力图。"""
    print("\n" + "=" * 20 + " Visualizing Case 2: Station Arrival Heatmap " + "=" * 20)
    plt.figure(figsize=(20, 10))
    sns.heatmap(arrival_matrix, cmap='YlOrRd', linewidths=.5, annot=True, fmt=".0f")
    plt.title('Case 2: Station Vehicle Arrivals (EV + EHDT)', fontsize=16)
    plt.xlabel('Hour of Day')
    plt.ylabel('Station ID')
    plt.savefig(f"{output_dir}/case2_arrival_heatmap.pdf", format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


# --- 4. 案例3: 电网服务策略对比 ---
def plot_case3_comparison(stats_dict, output_dir):
    """为案例3生成V2G策略对比图表。"""
    print("\n" + "=" * 20 + " Visualizing Case 3: Grid Service Strategies " + "=" * 20)

    # --- 经济成本对比 ---
    costs = {name: stats['total_cost'] for name, stats in stats_dict.items()}
    plt.figure(figsize=(12, 8))
    # --- 【兼容性修复】 ---
    # 修复了未来版本中 `palette` 的使用警告
    sns.barplot(x=list(costs.keys()), y=list(costs.values()), palette=['#2c7fb8', '#7fcdbb', '#edf8b1'],
                hue=list(costs.keys()), legend=False)
    plt.title('Case 3: Economic Cost of Different Battery Strategies', fontsize=16)
    plt.ylabel('Total Daily Cost (Yuan)')
    plt.xticks(rotation=15, ha='right')
    plt.savefig(f"{output_dir}/case3_cost_comparison.pdf", format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    # --- 削峰对比 ---
    plt.figure(figsize=(15, 7))
    colors = ['#2c7fb8', '#7fcdbb', '#edf8b1']
    linestyles = ['-', '--', ':']
    for i, (name, stats) in enumerate(stats_dict.items()):
        # 确保grid_load是Series，并有时间索引
        if isinstance(stats['grid_load'], pd.Series) and isinstance(stats['grid_load'].index, pd.TimedeltaIndex):
            x_values = stats['grid_load'].index.total_seconds() / 3600
            plt.plot(x_values, stats['grid_load'], label=name, color=colors[i], linestyle=linestyles[i], linewidth=2.5)
    plt.axhline(0, color='gray', linestyle=':')
    plt.title('Case 3: Grid Peak Shaving Comparison', fontsize=16)
    plt.xlabel('Time (Hour of Day)')
    plt.ylabel('Power Drawn from Grid (kW)')
    plt.legend()
    plt.savefig(f"{output_dir}/case3_peak_shaving_comparison.pdf", format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()