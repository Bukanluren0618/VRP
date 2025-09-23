# src/analysis/post_analysis.py

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from pyomo.environ import value
import numpy as np

# --- Configuration for plotting with Chinese characters ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_value(var_or_data):
    """Safely get the value of a Pyomo variable, returning 0 if it's None or invalid."""
    if hasattr(var_or_data, 'is_variable_type') and var_or_data.is_variable_type():
        # Check if the variable has a value; if not, return 0 to avoid errors
        return value(var_or_data, exception=False) if var_or_data.value is not None else 0
    return var_or_data


# --- CORE FUNCTION 1: EXTRACT ROUTES (NEWLY ADDED) ---
def extract_routes(model, data, vehicle_ids):
    """
    Extracts the ordered list of visited locations for each vehicle from the solved model.
    This is the missing function that reconstructs the vehicle paths.
    """
    final_routes = {}
    for k in vehicle_ids:
        depot = data['vehicles'][k]['depot_id']
        current_loc = depot
        route = [depot]
        # We loop until we return to the depot or the path breaks
        for _ in range(len(data['locations']) + 1):
            # Find the next location j that the vehicle travels to from current_loc
            next_loc = None
            for j in data['locations']:
                if current_loc != j and safe_value(model.x[current_loc, j, k]) > 0.9:
                    next_loc = j
                    break

            if next_loc and next_loc != depot:
                route.append(next_loc)
                current_loc = next_loc
            else:  # Path ends or returns to depot
                route.append(depot)
                break

        # Clean up route if it's just Depot -> Depot
        if len(route) == 2 and route[0] == route[1]:
            final_routes[k] = []
        else:
            final_routes[k] = route
    return final_routes


# --- CORE FUNCTION 2: ANALYZE SOLUTION (NEWLY ADDED) ---
def analyze_solution(model, data, vehicle_ids, task_ids, config):
    """
    Analyzes the full solution from the model, calculates summaries,
    and triggers all the plotting functions. This is the main missing orchestrator.
    """
    print("\n" + "=" * 20 + " Starting Post-Solution Analysis " + "=" * 20)

    # First, extract the routes using our new function
    routes = extract_routes(model, data, vehicle_ids)

    fleet_summary = {}
    delay_data = []

    for k in vehicle_ids:
        route_nodes = routes.get(k, [])
        if not route_nodes or len(route_nodes) <= 2:
            print(f"Vehicle {k} has no assigned route. Skipping.")
            continue

        total_dist = 0
        for i in range(len(route_nodes) - 1):
            total_dist += data['dist_matrix'].loc[route_nodes[i], route_nodes[i + 1]]

        fleet_summary[k] = {
            'route': route_nodes,
            'distance': total_dist,
            'duration': safe_value(model.tour_duration[k]),
            'tasks': sum(1 for node in route_nodes if node in model.CUSTOMERS)
        }

        # Create a dictionary for detailed plotting that includes the model
        route_info_for_plots = {'route': route_nodes, 'model': model}

        # Generate individual plots for this vehicle
        plot_individual_truck_analysis(k, route_info_for_plots, data, config)
        plot_gantt_chart(k, route_info_for_plots, data, config)

    # Analyze task delays
    for t in task_ids:
        customer_node = data['tasks'][t]['delivery_to']
        arrival = 0
        served_by = "None"
        for k in vehicle_ids:
            if safe_value(model.y[customer_node, k]) > 0.9:
                arrival = safe_value(model.arrival_time[customer_node, k])
                served_by = k
                break

        due = data['tasks'][t]['due_time']
        delay_data.append({
            'task_id': t,
            'customer': customer_node,
            'due_time': due,
            'arrival_time': arrival,
            'delay': max(0, arrival - due),
            'served_by': served_by
        })

    # Calculate cost summary
    cost_summary = {
        'Travel Cost': safe_value(summation(model.tour_duration) * config.MANPOWER_COST_PER_HOUR),
        'Swap Cost': safe_value(summation(model.swap_decision) * config.FIXED_SWAP_COST),
        'Delay Penalty': safe_value(summation(model.delay_hours) * config.DELAY_PENALTY_PER_HOUR)
    }

    # Generate summary plots
    plot_fleet_summary(fleet_summary, cost_summary, config)
    plot_task_delays(delay_data)

    print("=" * 20 + " Post-Solution Analysis Complete " + "=" * 20 + "\n")
    return {'fleet_summary': fleet_summary, 'delay_data': delay_data, 'cost_summary': cost_summary}


# --- All existing plotting functions remain below ---

def plot_road_network_with_routes(final_routes, data, filename="fleet_routing_plan.png"):
    # This function is now OBSOLETE as its functionality is in visualizations.py
    # but we keep it here to prevent breaking old calls if any.
    print("Note: 'plot_road_network_with_routes' in post_analysis.py is deprecated.")
    pass


def plot_individual_truck_analysis(vehicle_id, route_info, data, config):
    filename_prefix = f"truck_analysis_{vehicle_id}"
    print(f"正在为车辆 {vehicle_id} 生成详细分析图...")
    model = route_info['model']
    route_node_names = route_info['route']

    if not model or not route_node_names or len(route_node_names) <= 1:
        print(f"车辆 {vehicle_id} 没有有效路径可供分析。")
        return

    # --- 图1: 单车路径图 ---
    G = data['traffic_graph']
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)
    node_types = {info['node_id']: info['type'] for name, info in data['locations'].items()}
    depots = [data['locations'][name]['node_id'] for name, info in data['locations'].items() if info['type'] == 'Depot']
    customers = [data['locations'][name]['node_id'] for name, info in data['locations'].items() if
                 info['type'] == 'Customer']
    stations = [data['locations'][name]['node_id'] for name, info in data['locations'].items() if
                info['type'] == 'SwapStation']

    nx.draw_networkx_nodes(G, pos, nodelist=depots, node_color='red', node_size=300, node_shape='s', label='仓库')
    nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color='skyblue', node_size=150, label='客户')
    nx.draw_networkx_nodes(G, pos, nodelist=stations, node_color='lightgreen', node_size=250, node_shape='p',
                           label='换电站')

    route_node_ids = [data['locations'][name]['node_id'] for name in route_node_names]
    nx.draw_networkx_nodes(G, pos, nodelist=route_node_ids, node_color='magenta', node_size=100)
    for i in range(len(route_node_ids) - 1):
        path_segment = data['path_matrix'].loc[route_node_names[i], route_node_names[i + 1]]
        if path_segment and len(path_segment) > 1:
            path_edges = list(zip(path_segment[:-1], path_segment[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='magenta', width=2.5, style='solid')
    plt.title(f"车辆 {vehicle_id} 的行驶路径", fontsize=16)
    plt.legend()
    save_path = os.path.join(os.getcwd(), "results", f"{filename_prefix}_route.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # --- 图2: 电量与载重变化图 ---
    timeline, soc_line, weight_line = [], [], []
    for loc_name in route_node_names:
        timeline.append(safe_value(model.arrival_time[loc_name, vehicle_id]))
        soc_line.append(safe_value(model.soc_arrival[loc_name, vehicle_id]))
        weight_line.append(safe_value(model.weight_on_arrival[loc_name, vehicle_id]))
    fig, ax1 = plt.subplots(figsize=(15, 7))
    color = 'tab:blue'
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('电池电量 (kWh)', color=color)
    ax1.plot(timeline, soc_line, color=color, marker='o', label='电池电量')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=config.HDT_MIN_SOC_KWH, color='red', linestyle='--',
                label=f'安全电量阈值 ({config.HDT_MIN_SOC_KWH} kWh)')
    ax1.grid(True)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('车辆载重 (吨)', color=color)
    ax2.step(timeline, weight_line, where='post', color=color, label='车辆载重')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(f'车辆 {vehicle_id} 的电量与载重变化分析', fontsize=16)
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center')
    save_path = os.path.join(os.getcwd(), "results", f"{filename_prefix}_metrics.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"车辆 {vehicle_id} 的详细分析图已保存。")


def plot_gantt_chart(vehicle_id, route_info, data, config):
    print(f"正在为车辆 {vehicle_id} 生成甘特图...")
    model = route_info['model']
    route = route_info['route']
    if len(route) <= 1: return
    fig, ax = plt.subplots(figsize=(18, 5))
    event_y = 0.5
    for i in range(len(route) - 1):
        loc_i, loc_j = route[i], route[i + 1]
        start_time = safe_value(model.departure_time[loc_i, vehicle_id])
        end_time = safe_value(model.arrival_time[loc_j, vehicle_id])
        duration = end_time - start_time
        if duration > 0.01:
            ax.barh(event_y, duration, left=start_time, height=0.5, color='royalblue', edgecolor='black')
            ax.text(start_time + duration / 2, event_y, f'行驶\n{loc_i} -> {loc_j}', ha='center', va='center',
                    color='white', fontsize=8)

        # Plot service/swap time at the destination node loc_j
        service_start = end_time
        service_end = safe_value(model.departure_time[loc_j, vehicle_id])
        service_duration = service_end - service_start
        if service_duration > 0.01:
            loc_type = data['locations'][loc_j]['type']
            color = 'orange' if loc_type == 'Customer' else 'lightgreen' if loc_type == 'SwapStation' else 'gray'
            label = '服务客户' if loc_type == 'Customer' else '换电' if loc_type == 'SwapStation' else '停留'
            ax.barh(event_y, service_duration, left=service_start, height=0.5, color=color, edgecolor='black')
            ax.text(service_start + service_duration / 2, event_y, f'{label}\n@{loc_j}', ha='center', va='center',
                    color='black', fontsize=8)

    ax.set_yticks([])
    ax.set_xlabel('时间 (小时)')
    ax.set_title(f'车辆 {vehicle_id} 调度甘特图', fontsize=16)
    ax.grid(axis='x')
    plt.xlim(0, config.TIME_HORIZON_HOURS)
    save_path = os.path.join(os.getcwd(), "results", f"gantt_chart_{vehicle_id}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_fleet_summary(fleet_summary_data, cost_summary, config):
    print("正在生成车队运营总结图...")
    if not fleet_summary_data:
        print("无车队数据可供分析。")
        return
    df = pd.DataFrame.from_dict(fleet_summary_data, orient='index')
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('车队运营性能总结', fontsize=24)
    df['duration'].sort_values().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('各车辆总行驶时长')
    axes[0, 0].set_ylabel('时长 (小时)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    df['distance'].sort_values().plot(kind='bar', ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('各车辆总行驶距离')
    axes[0, 1].set_ylabel('距离 (公里)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    df['tasks'].sort_values().plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('各车辆服务任务数')
    axes[1, 0].set_ylabel('任务数')
    axes[1, 0].tick_params(axis='x', rotation=45)
    cost_series = pd.Series(cost_summary)
    if not cost_series.empty and cost_series.sum() > 0:
        axes[1, 1].pie(cost_series, labels=cost_series.index, autopct='%1.1f%%', startangle=90,
                       colors=['gold', 'lightcoral', 'lightskyblue'])
        axes[1, 1].set_title('总运输成本构成')
    else:
        axes[1, 1].text(0.5, 0.5, '无成本数据', horizontalalignment='center', verticalalignment='center')
    axes[1, 1].axis('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(os.getcwd(), "results", "summary_fleet_performance.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"车队运营总结图已保存到: {save_path}")


def plot_task_delays(delay_data):
    print("正在生成任务准时性分析图...")
    if not delay_data:
        print("无任务延误数据可供分析。")
        return
    df = pd.DataFrame(delay_data)
    plt.figure(figsize=(12, 8))
    ontime = df[df['delay'] <= 0.01]
    late = df[df['delay'] > 0.01]
    plt.scatter(ontime['due_time'], ontime['arrival_time'], color='green', alpha=0.7,
                label=f'准时/早达 ({len(ontime)})')
    plt.scatter(late['due_time'], late['arrival_time'], color='red', alpha=0.7, label=f'迟到 ({len(late)})')
    max_time = max(df['due_time'].max(), df['arrival_time'].max()) if not df.empty else 24
    plt.plot([0, max_time], [0, max_time], 'k--', label='准时线 (到达时间 = 截止时间)')
    plt.title('任务送达准时性分析', fontsize=18)
    plt.xlabel('任务要求截止时间 (小时)')
    plt.ylabel('车辆实际到达时间 (小时)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    save_path = os.path.join(os.getcwd(), "results", "summary_task_delays.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"任务准时性图已保存到: {save_path}")


# These station-related plots are likely for a different model (two-stage)
# They are kept here but might not be used by the current main.py
def plot_station_energy_flows(station_id, model, station_data, config, filename_prefix="station_analysis"):
    pass


def plot_station_summary(station_summary_data, config):
    pass