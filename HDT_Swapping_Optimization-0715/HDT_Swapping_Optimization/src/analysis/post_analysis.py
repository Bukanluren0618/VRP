# HDT_Swapping_Optimization/src/analysis/post_analysis.py

import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyomo.environ import value
from src.common import config_final as config


"""Convenience imports for analysis utilities."""

from . import post_analysis
from .pre_checks import check_task_feasibility

# Re-export commonly used helpers for simplicity
from .post_analysis import *  # noqa: F401,F403

__all__ = post_analysis.__all__ + ["check_task_feasibility"]

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_value(var):
    """
    【关键修复】: 创建一个安全的取值函数，与旧版Pyomo兼容。
    如果变量未被求解器赋值(其 .value 为 None)，则返回0。
    """
    # 既要兼容 Pyomo 变量/表达式, 也要兼容普通的数值类型
    if hasattr(var, "value"):
        # Pyomo Var 或 Param
        if var.value is None:
            return 0
        return value(var)
    else:
        # 直接是数值(如 float/int)
        if var is None:
            return 0
        return float(var)


def plot_road_network_with_routes(model, data, filename="hdt_routing_plan.png"):
    """
    在路网图上绘制所有HDT的规划路径。
    """
    print("正在绘制HDT路径规划图...")
    G = data['traffic_graph']
    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(20, 16))

    # 绘制节点和边
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

    # 绘制路径
    vehicle_colors = plt.cm.get_cmap('gist_rainbow', len(model.VEHICLES))
    for k_idx, k in enumerate(model.VEHICLES):
        color = vehicle_colors(k_idx)
        for i in model.LOCATIONS:
            for j in model.LOCATIONS:
                # 使用我们新的安全取值函数
                if i != j and safe_value(model.x[i, j, k]) > 0.5:
                    path_nodes = data['path_matrix'].loc[i, j]
                    if path_nodes and len(path_nodes) > 1:
                        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                        nx.draw_networkx_edges(
                            G,
                            pos,
                            edgelist=path_edges,
                            edge_color=color,
                            width=2.5,
                            style='dashed',
                            alpha=0.8,
                            arrows=True,
                            arrowstyle='->',
                            arrowsize=15,
                            label=f'Vehicle {k}'
                        )

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

    def print_task_assignments(model, data):
        """打印每个任务由哪辆车执行"""
        print("\n========== 任务分配结果 ==========")
        for t in model.TASKS:
            customer = data['tasks'][t]['delivery_to']
            assigned = None
            for k in model.VEHICLES:
                if safe_value(model.y[customer, k]) > 0.5:
                    assigned = k
                    break
            print(f"任务 {t} -> 车辆 {assigned if assigned else '未分配'}")
        print("===================================\n")

    def print_vehicle_swap_nodes(model):
        """打印每辆车在哪些站点换电"""
        print("\n========== 换电站点选择 ==========")
        for k in model.VEHICLES:
            swaps = [s for s in model.STATIONS if safe_value(model.swap_decision[s, k]) > 0.5]
            if swaps:
                print(f"车辆 {k} 换电站点: {swaps}")
            else:
                print(f"车辆 {k} 未进行换电")
        print("===================================\n")

        def print_vehicle_routes(model, data):
            """按顺序打印每辆车的行驶路线(节点序列)"""
            print("\n========== 车辆行驶路线 ==========")
            for k in model.VEHICLES:
                depot = data['vehicles'][k]['depot_id']
                route = [depot]
                current = depot
                visited = set([depot])
                while True:
                    next_nodes = [j for j in model.LOCATIONS if
                                  current != j and safe_value(model.x[current, j, k]) > 0.5]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    route.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    if current == depot:
                        break
                print(f"车辆 {k}: {' -> '.join(route)}")
            print("===================================\n")

    def plot_hdt_metrics(model, data):
        """为每辆车绘制电量/载重/耗电率三坐标图"""
        for k in model.VEHICLES:
            visited = [(safe_value(model.arrival_time[n, k]), n) for n in model.LOCATIONS
                       if n in model.DEPOT or safe_value(model.y[n, k]) > 0.5]
            visited.sort()
            if not visited:
                continue

            soc = []
            load = []
            rate = []
            steps = []
            prev = None
            for step, (t, n) in enumerate(visited):
                soc.append(safe_value(model.soc_arrival[n, k]))
                load.append(safe_value(model.weight_on_arrival[n, k]))
                if prev is None:
                    rate.append(0)
                else:
                    i = prev
                    j = n
                    dist = data['dist_matrix'].loc[i, j]
                    demand_i = sum(info['demand'] for info in data['tasks'].values()
                                   if info['delivery_to'] == i)
                    depart_w = safe_value(model.weight_on_arrival[i, k]) - demand_i * safe_value(model.y[i, k])
                    energy = dist * (config.HDT_BASE_CONSUMPTION_KWH_PER_KM +
                                     depart_w * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
                    rate.append(energy / dist if dist > 0 else 0)
                steps.append(step)
                prev = n

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(steps, soc, label='电量(kWh)', color='tab:blue')
            ax1.set_ylabel('电量(kWh)', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.step(steps, load, where='post', label='载重(t)', color='tab:orange')
            ax2.set_ylabel('载重(t)', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            ax3.plot(steps, rate, label='耗电率(kWh/km)', color='tab:green')
            ax3.set_ylabel('耗电率(kWh/km)', color='tab:green')
            ax3.tick_params(axis='y', labelcolor='tab:green')

            lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            ax1.set_xlabel('步骤')
            ax1.set_title(f'Vehicle {k} 电量/载重/耗电率')

            plt.tight_layout()
            fname = f'hdt_metrics_{k}.png'
            plt.savefig(os.path.join(config.RESULTS_DIR, fname))
            plt.close()
            print(f"三坐标图已保存: {os.path.join(config.RESULTS_DIR, fname)}")