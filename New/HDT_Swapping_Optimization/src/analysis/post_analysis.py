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


def safe_value(var):
    """
    【关键修复】: 创建一个安全的取值函数，与旧版Pyomo兼容。
    如果变量未被求解器赋值(其 .value 为 None)，则返回0。
    """
    if not hasattr(var, "value"):
        # 将可能的numpy类型转为Python float
        try:
            return float(var)
        except Exception:
            return 0

    if var.value is None:
        return 0
    return value(var)


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
    others = [n for n in G.nodes if n not in depots + customers + stations]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=others,
        node_color="lightgray",
        node_size=50,
        label="Road Nodes",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=depots,
        node_color="gold",
        node_size=500,
        node_shape="s",
        label="Depots",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=customers,
        node_color="skyblue",
        node_size=300,
        label="Customers",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=stations,
        node_color="lightgreen",
        node_size=400,
        node_shape="p",
        label="Stations",
    )

    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

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
                            style="dashed",
                            alpha=0.8,
                            label=f"Vehicle {k}",
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
    time_index = pd.to_datetime(
        pd.Series(range(config.TOTAL_TIME_STEPS)) * config.TIME_STEP_MINUTES, unit="m"
    )

    for s in model.STATIONS:
        # 使用我们新的安全取值函数
        df_data = {
            "Grid Power (kW)": [safe_value(model.P_grid[s, t]) for t in model.TIME],
            "PV Power (kW)": [
                safe_value(data["pv_generation"][s][t]) for t in model.TIME
            ],
            "To Charge Station Batteries (kW)": [
                safe_value(model.P_charge_batt[s, t]) for t in model.TIME
            ],
            "EV Demand (kW)": [
                safe_value(data["ev_demand_timestep"][s][t] / config.TIME_STEP_HOURS)
                for t in model.TIME
            ],
            "EV Unserved (kW)": [
                safe_value(model.ev_unserved[s, t]) / config.TIME_STEP_HOURS
                for t in model.TIME
            ],
        }
        df = pd.DataFrame(df_data, index=time_index)

        fig, ax1 = plt.subplots(figsize=(18, 9))

        ax1.stackplot(
            df.index,
            df["Grid Power (kW)"],
            df["PV Power (kW)"],
            labels=["From Grid", "From PV"],
            alpha=0.7,
        )
        ax1.plot(
            df.index,
            df["EV Demand (kW)"],
            label="EV Demand",
            color="black",
            linestyle="--",
            linewidth=2,
        )

        ax1.set_xlabel("Time of Day")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title(f"Energy Schedule and Battery Status for {s}", fontsize=20)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(
            time_index,
            [safe_value(model.N_full_batt[s, t]) for t in model.TIME],
            label="Full Batteries",
            color="green",
            marker="o",
            linewidth=2.5,
        )
        ax2.set_ylabel("Number of Full Batteries")
        ax2.legend(loc='upper right')

        fig.tight_layout()
        filename = f"energy_schedule_{s}.png"
        plt.savefig(os.path.join(config.RESULTS_DIR, filename))
        plt.close()
        print(f"能源调度图已保存到: {config.RESULTS_DIR}/")


def check_task_feasibility(data):
    """Calculate round-trip distance, time, and energy for each task."""
    results = []
    for task_id, info in data["tasks"].items():
        depot = info["depot"]
        customer = info["delivery_to"]
        dist_out = data["dist_matrix"].loc[depot, customer]
        dist_back = data["dist_matrix"].loc[customer, depot]
        total_dist = dist_out + dist_back
        time_out = data['time_matrix'].loc[depot, customer]
        time_back = data['time_matrix'].loc[customer, depot]
        total_time = time_out + time_back + config.LOADING_UNLOADING_TIME_HOURS
        energy_out = dist_out * (
                config.HDT_BASE_CONSUMPTION_KWH_PER_KM
                + (config.HDT_EMPTY_WEIGHT_TON + info["demand"])
                * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
        )
        energy_back = dist_back * (
                config.HDT_BASE_CONSUMPTION_KWH_PER_KM
                + config.HDT_EMPTY_WEIGHT_TON * config.HDT_WEIGHT_CONSUMPTION_KWH_PER_KM_TON
        )
        total_energy = energy_out + energy_back
        feasible = (
                total_energy + config.HDT_MIN_SOC_KWH <= config.HDT_BATTERY_CAPACITY_KWH
        )
        results.append(
            {
                "Task": task_id,
                "Roundtrip Distance (km)": round(float(total_dist), 2),
                "Time (h)": round(float(total_time), 2),
                "Energy (kWh)": round(float(total_energy), 2),
                "Feasible": feasible,
            }
        )
    df = pd.DataFrame(results).set_index("Task")
    print("任务可达性检查:")
    print(df.to_string())
    print()
    return df