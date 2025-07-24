# HDT_Swapping_Optimization/src/analysis/post_analysis.py

import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyomo.environ import value
from src.common import config_final as config

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_value(var_or_data):
    """
    【最终修复版】: 创建一个更健壮的安全取值函数，兼容所有Pyomo版本。
    它能判断传入的是Pyomo变量还是普通数据。
    """
    # 检查传入的是否是Pyomo的Var或VarData对象
    if hasattr(var_or_data, 'is_variable_type') and var_or_data.is_variable_type():
        # 如果是变量，检查其.value是否为None (这是最安全的方式)
        if var_or_data.value is None:
            return 0
        # 如果有值，则使用Pyomo的value()函数获取
        return value(var_or_data)

    # 如果不是变量（例如，是一个普通的数字或numpy.float64），直接返回它自己
    return var_or_data


def plot_road_network_with_routes(model, data, filename="hdt_routing_plan.png"):
    """
    在路网图上绘制所有HDT的规划路径。
    """
    print("正在绘制HDT路径规划图...")
    G = data['traffic_graph']
    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(20, 16))

    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)

    node_types = nx.get_node_attributes(G, 'type')
    depots = [n for n, t in node_types.items() if t == 'Depot']
    customers = [n for n, t in node_types.items() if t == 'Customer']
    stations = [n for n, t in node_types.items() if t == 'SwapStation']

    nx.draw_networkx_nodes(G, pos, nodelist=depots, node_color='gold', node_size=500, node_shape='s', label='Depots')
    nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color='skyblue', node_size=300, label='Customers')
    nx.draw_networkx_nodes(G, pos, nodelist=stations, node_color='lightgreen', node_size=400, node_shape='p',
                           label='Stations')

    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    vehicle_colors = plt.cm.get_cmap('gist_rainbow', len(model.VEHICLES))
    for k_idx, k in enumerate(model.VEHICLES):
        color = vehicle_colors(k_idx)
        for i in model.LOCATIONS:
            for j in model.LOCATIONS:
                if i != j and safe_value(model.x[i, j, k]) > 0.5:
                    path_nodes = data['path_matrix'].loc[i, j]
                    if path_nodes and len(path_nodes) > 1:
                        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=2.5, style='dashed',
                                               alpha=0.8, label=f'Vehicle {k}')

    plt.title("HDT Fleet Routing Plan", fontsize=20)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, filename))
    plt.close()
    print(f"路径图已保存到: {os.path.join(config.RESULTS_DIR, filename)}")


def plot_station_energy_schedule(model, data):
    """
    绘制每个能源站的详细能源调度计划。
    """
    print("正在绘制能源站调度图...")
    time_index = pd.to_datetime(pd.Series(range(config.TOTAL_TIME_STEPS)) * config.TIME_STEP_MINUTES, unit='m')

    for s in model.STATIONS:
        # 使用我们新的安全取值函数
        df_data = {
            'Grid Power (kW)': [safe_value(model.P_grid[s, t]) for t in model.TIME],
            'PV Power (kW)': [safe_value(data['pv_generation'][s][t]) for t in model.TIME],
            'To Charge Station Batteries (kW)': [safe_value(model.P_charge_batt[s, t]) for t in model.TIME],
            'EV Demand (kW)': [safe_value(data['ev_demand_timestep'][s][t] / config.TIME_STEP_HOURS) for t in
                               model.TIME],
            'EV Unserved (kW)': [safe_value(model.ev_unserved[s, t]) / config.TIME_STEP_HOURS for t in model.TIME],
        }
        df = pd.DataFrame(df_data, index=time_index)

        fig, ax1 = plt.subplots(figsize=(18, 9))

        ax1.stackplot(df.index, df['Grid Power (kW)'], df['PV Power (kW)'], labels=['From Grid', 'From PV'], alpha=0.7)
        ax1.plot(df.index, df['EV Demand (kW)'], label='EV Demand', color='black', linestyle='--', linewidth=2)
        ax1.set_xlabel("Time of Day")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title(f"Energy Schedule and Battery Status for {s}", fontsize=20)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(time_index, [safe_value(model.N_full_batt[s, t]) for t in model.TIME], label='Full Batteries',
                 color='green', marker='o', linewidth=2.5)
        ax2.set_ylabel("Number of Full Batteries")
        ax2.legend(loc='upper right')

        fig.tight_layout()
        filename = f"energy_schedule_{s}.png"
        plt.savefig(os.path.join(config.RESULTS_DIR, filename))
        plt.close()
    print(f"能源调度图已保存到: {config.RESULTS_DIR}/")