# HDT_Swapping_Optimization/src/analysis/post_analysis.py

import os
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.font_manager as fm
import pandas as pd
from pyomo.environ import value
from src.common import config_final as config



for fname in ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]:
    if any(f.name == fname for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [fname]
        break
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_road_network_with_routes(model, data, filename="hdt_routing_plan.png"):
    """
    在路网图上绘制所有HDT的规划路径。
    """
    print("正在绘制HDT路径规划图...")
    # 【核心修正】: 使用正确的键 'traffic_graph'
    G = data['traffic_graph']
    pos = nx.get_node_attributes(G, 'pos')
    key_locations = data['locations'].keys()

    plt.figure(figsize=(20, 16))

    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, d in G.nodes(data=True) if d.get('type') == 'junction'],
                           node_color='gray', node_size=50, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)

    node_colors = {'Depot': 'gold', 'SwapStation1': 'green', 'SwapStation2': 'green'}
    node_sizes = {'Depot': 500, 'SwapStation1': 400, 'SwapStation2': 400}

    location_nodes = [n for n in key_locations]
    location_colors = [node_colors.get(n, 'skyblue') for n in location_nodes]
    location_sizes = [node_sizes.get(n, 300) for n in location_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=location_nodes, node_color=location_colors, node_size=location_sizes,
                           edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels={n: n for n in key_locations}, font_size=10, font_weight='bold')

    vehicle_colors = ['red', 'blue', 'purple', 'orange']
    for k_idx, k in enumerate(model.VEHICLES):
        # 【核心修正】: 使用正确的变量名 vehicle_colors
        color = vehicle_colors[k_idx % len(vehicle_colors)]
        for i in model.LOCATIONS:
            for j in model.LOCATIONS:
                if i != j and value(model.x[i, j, k]) > 0.5:
                    path_nodes = data['path_matrix'].loc[i, j]
                    if path_nodes and len(path_nodes) > 1:
                        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=2.5, style='dashed',
                                               alpha=0.8)

    plt.title("HDT Fleet Routing Plan on Urban Road Network", fontsize=20)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, filename))
    plt.close()
    print(f"路径图已保存到: {os.path.join(config.RESULTS_DIR, filename)}")


def plot_station_energy_schedule(model, data):
    """
    绘制每个能源站的详细能源调度计划。
    """
    print("正在绘制能源站调度图...")
    for s in model.STATIONS:
        df_data = {
            'Grid Power (kW)': [value(model.P_grid[s, t]) for t in model.TIME],
            'PV Power (kW)': [data['pv_generation'][s][t] for t in model.TIME],
            'Battery Charge (kW)': [value(model.P_charge_batt[s, t]) for t in model.TIME],
            'EV Demand (kW)': [data['ev_demand_timestep'][s][t] / config.TIME_STEP_HOURS for t in model.TIME],
            'EV Unserved (kW)': [value(model.ev_unserved[s, t]) / config.TIME_STEP_HOURS for t in model.TIME],
        }
        df_index = pd.to_datetime(pd.Series(range(config.TOTAL_TIME_STEPS)) * config.TIME_STEP_MINUTES, unit='m')
        df = pd.DataFrame(df_data, index=df_index)

        fig, ax1 = plt.subplots(figsize=(18, 9))

        ax1.stackplot(df.index, df['Grid Power (kW)'], df['PV Power (kW)'], labels=['From Grid', 'From PV'], alpha=0.7)
        ax1.plot(df.index, df['EV Demand (kW)'], label='EV Demand', color='black', linestyle='--', linewidth=2)
        ax1.plot(df.index, df['Battery Charge (kW)'], label='To Charge Station Batteries', color='orange', linewidth=2)
        ax1.set_xlabel("Time of Day")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title(f"Energy Schedule and Battery Status for {s}", fontsize=20)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(df.index, [value(model.N_full_batt[s, t]) for t in model.TIME], label='Full Batteries', color='green',
                 marker='o', linewidth=2.5)
        ax2.set_ylabel("Number of Full Batteries")
        ax2.legend(loc='upper right')

        fig.tight_layout()
        filename = f"energy_schedule_{s}.png"
        plt.savefig(os.path.join(config.RESULTS_DIR, filename))
        plt.close()
    print(f"能源调度图已保存到: {config.RESULTS_DIR}/")

def print_assignment_summary(model, data):
    """打印任务分配和车辆边变量等信息, 便于调试模型结果"""
    print("\n[决策变量检查]")
    for t in model.TASKS:
        cust = data['tasks'][t]['delivery_to']
        if hasattr(model, 'is_task_unassigned') and value(model.is_task_unassigned[t]) > 0.5:
            print(f"任务 {t} 被放弃")
            continue
        assigned = [k for k in model.VEHICLES if value(model.y[cust, k]) > 0.5]
        unassigned = value(model.is_task_unassigned[t]) > 0.5
        if assigned:
            print(f"任务 {t} 分配给车辆: {', '.join(assigned)}; is_task_unassigned={unassigned}")
        else:
            print(f"任务 {t} 未分配; is_task_unassigned={unassigned}")

    for k in model.VEHICLES:
        arcs = []
        for i in model.LOCATIONS:
            for j in model.LOCATIONS:
                x_var = model.x[i, j, k]
                if x_var.value is not None and x_var.value > 0.5:
                    arcs.append((i, j))
        if arcs:
            print(f"车辆 {k} 行驶边: {arcs}")
        else:
            print(f"车辆 {k} 无行驶边")


def print_task_demands(data):
    """打印每个配送任务的需求量。"""
    print("\n物流需求一览:")
    for task, info in data['tasks'].items():
        loc = info['delivery_to']
        print(f"  {task}: {loc} - {info['demand']} 吨")

def check_task_feasibility(data):
    """简单评估每个任务在单次往返情况下的时间和能耗需求。"""
    print("\n[任务可达性检查]")
    for task, info in data['tasks'].items():
        cust = info['delivery_to']
        dist = data['dist_matrix'].loc['Depot', cust] + data['dist_matrix'].loc[cust, 'Depot']
        travel_time = data['time_matrix'].loc['Depot', cust] + data['time_matrix'].loc[cust, 'Depot']
        demand = info['demand']
        rate = (config.HDT_BASE_CONSUMPTION_KWH_PER_KM +
                (config.HDT_EMPTY_WEIGHT_TON + demand) * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON)
        energy = dist * rate
        feasible = (energy <= config.HDT_BATTERY_CAPACITY_KWH and travel_time <= info['due_time'])
        flag = "可行" if feasible else "不可行"
        print(f"{task}: 往返距离{dist:.1f}km, 需时{travel_time:.1f}h, 耗电{energy:.1f}kWh -> {flag}")


def _extract_route(model, vehicle):
    """根据到达时间对车辆访问的节点排序, 重建路线"""
    visited = [n for n in model.LOCATIONS if value(model.y[n, vehicle]) > 0.5 or n in model.DEPOT]
    visited_times = {n: value(model.arrival_time[n, vehicle]) for n in visited}
    route = [n for n, _ in sorted(visited_times.items(), key=lambda x: x[1])]
    if route[0] != 'Depot':
        route.insert(0, 'Depot')
    if route[-1] != 'Depot':
        route.append('Depot')
    return route

def print_vehicle_route(model, vehicle):
    """打印车辆的完整行驶路线。若车辆未离开仓库，则进行提示。"""
    route = _extract_route(model, vehicle)
    if len(route) <= 1:
        print(f"车辆 {vehicle} 未执行任何任务，仅停留在仓库。")
    else:
        print(f"车辆 {vehicle} 行驶路线: {' -> '.join(route)}")



def plot_vehicle_metrics(model, data, vehicle):
    """绘制指定车辆的SOC、载重和耗电率变化曲线"""
    route = _extract_route(model, vehicle)
    steps = list(range(len(route)))
    soc = [value(model.soc_arrival[n, vehicle]) for n in route]
    load = [value(model.weight_on_arrival[n, vehicle]) for n in route]
    rate = []
    for idx in range(len(route) - 1):
        w = load[idx]
        for t, info in data['tasks'].items():
            if info['delivery_to'] == route[idx]:
                if value(model.y[info['delivery_to'], vehicle]) > 0.5:
                    w -= info['demand']
        r = config.HDT_BASE_CONSUMPTION_KWH_PER_KM + w * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
        rate.append(r)
    if rate:
        rate.append(rate[-1])
    else:
        rate = [0] * len(route)

        if len(route) <= 1:
            print(f"车辆 {vehicle} 没有行驶记录，仅停留在仓库。生成平坦曲线图。")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, soc, label='电量', color='tab:blue')
    ax1.set_ylabel('电量 (kWh)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.step(steps, load, where='post', label='载重', color='tab:orange')
    ax2.set_ylabel('载重 (t)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(steps, rate, label='耗电率', color='tab:green')
    ax3.set_ylabel('耗电率 (kWh/km)', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.set_xlabel('步骤')
    ax1.set_title(f'{vehicle} 电量/载重/耗电率')

    plt.tight_layout()
    filename = f"metrics_{vehicle}.png"
    plt.savefig(os.path.join(config.RESULTS_DIR, filename))
    plt.close()
    print(f"车辆 {vehicle} 指标图保存到: {os.path.join(config.RESULTS_DIR, filename)}")