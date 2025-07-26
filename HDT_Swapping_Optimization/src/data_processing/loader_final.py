# HDT_Swapping_Optimization/src/data_processing/loader_final.py

import pandas as pd
import numpy as np
import os
import random
import networkx as nx
from src.common import config_final as config
from src.simulation import road_network


def get_ieee_118_bus_data():
    load_buses = [
        1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 31, 32, 35, 36, 39, 41, 43, 44, 45, 46, 47, 48, 50,
        51, 52, 53, 57, 58, 63, 64, 67, 72, 73, 74, 75, 76, 82, 83, 84,
        85, 86, 91, 93, 94, 95, 96, 97, 98, 101, 102, 108, 109, 110, 111,
        112, 114, 115, 117
    ]
    return [f"Bus_{i}" for i in load_buses]


def create_urban_delivery_scenario():
    print("=" * 30)
    print("开始创建全新的【城市配送】场景...")
    random.seed(42)
    np.random.seed(42)

    G, pos = road_network.generate_connected_graph(config.CITY_NODE_COUNT, config.CITY_GRAPH_RADIUS)
    all_nodes = list(G.nodes())
    random.shuffle(all_nodes)

    depot_nodes = all_nodes[:config.NUM_DEPOTS]
    station_nodes = all_nodes[config.NUM_DEPOTS: config.NUM_DEPOTS + config.NUM_STATIONS]
    customer_nodes = all_nodes[
                     config.NUM_DEPOTS + config.NUM_STATIONS: config.NUM_DEPOTS + config.NUM_STATIONS + config.NUM_CUSTOMERS]

    locations_data = {}
    for i, node in enumerate(depot_nodes): locations_data[f"Depot_{i + 1}"] = {'type': 'Depot', 'pos': pos[node],
                                                                               'node_id': node}
    for i, node in enumerate(station_nodes): locations_data[f"Station_{i + 1}"] = {'type': 'SwapStation',
                                                                                   'pos': pos[node], 'node_id': node}
    for i, node in enumerate(customer_nodes): locations_data[f"Customer_{i + 1}"] = {'type': 'Customer',
                                                                                     'pos': pos[node], 'node_id': node}
    print(f"场景设施点分配完毕: {len(depot_nodes)}仓库, {len(station_nodes)}换电站, {len(customer_nodes)}客户。")

    all_location_node_ids = [d['node_id'] for d in locations_data.values()]
    dist_matrix_raw, path_matrix_raw = road_network.get_path_and_distance_matrices(G, all_location_node_ids)
    key_locations_list = list(locations_data.keys())
    node_id_to_name_map = {d['node_id']: name for name, d in locations_data.items()}
    dist_matrix = dist_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)
    path_matrix = path_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)
    dist_matrix = dist_matrix / config.DISTANCE_DIVISOR
    print(f"路网距离已除以 {config.DISTANCE_DIVISOR} 以模拟城市环境。")
    avg_speed_kmh = 30.0
    time_df = dist_matrix / avg_speed_kmh

    vehicles = {}
    tasks = {}
    task_id_counter = 1
    depot_names = [name for name, info in locations_data.items() if info['type'] == 'Depot']
    customer_names = [name for name, info in locations_data.items() if info['type'] == 'Customer']

    for i in range(config.NUM_TRUCKS):
        truck_id = f"HDT_{i + 1}"
        assigned_depot = random.choice(depot_names)
        vehicles[truck_id] = {'initial_soc': config.HDT_BATTERY_CAPACITY_KWH, 'depot_id': assigned_depot}

        num_tasks_for_truck = random.randint(config.MIN_TASKS_PER_TRUCK, config.MAX_TASKS_PER_TRUCK)

        if len(customer_names) < num_tasks_for_truck:
            assigned_customers = customer_names
        else:
            assigned_customers = random.sample(customer_names, num_tasks_for_truck)

        for cust_name in assigned_customers:
            task_id = f"Task_{task_id_counter}"

            # --- [最终修正] 确保due_time是合理的 ---
            one_way_time = time_df.loc[assigned_depot, cust_name]
            earliest_possible_due_time = one_way_time + config.LOADING_UNLOADING_TIME_HOURS + 1.0
            latest_possible_due_time = earliest_possible_due_time + 8.0
            due_time = round(random.uniform(earliest_possible_due_time, latest_possible_due_time), 1)

            tasks[task_id] = {
                'delivery_to': cust_name,
                'demand': round(random.uniform(1.0, 2.5), 2),
                'due_time': due_time,
                'depot': assigned_depot
            }
            task_id_counter += 1

    print(f"已生成 {len(vehicles)} 辆卡车和 {len(tasks)} 个配送任务。")

    ieee_118_buses = get_ieee_118_bus_data()
    random.shuffle(ieee_118_buses)
    stations = {}
    station_to_bus_map = {}
    station_names = [name for name, info in locations_data.items() if info['type'] == 'SwapStation']
    for i, station_name in enumerate(station_names):
        bus_id = ieee_118_buses[i % len(ieee_118_buses)]
        stations[station_name] = {'initial_full': 10, 'initial_empty': 5, 'bus_id': bus_id}
        station_to_bus_map[station_name] = bus_id

    power_grid = nx.DiGraph()
    power_buses = ['Bus_Substation_Slack'] + ieee_118_buses
    power_grid.add_nodes_from(power_buses)
    for bus in ieee_118_buses: power_grid.add_edge('Bus_Substation_Slack', bus)

    time_steps = range(config.TOTAL_TIME_STEPS)
    electricity_prices = pd.Series(
        [0.4 if 23 <= t * config.TIME_STEP_HOURS or t * config.TIME_STEP_HOURS < 7 else 1.0 for t in time_steps],
        index=time_steps)
    ev_demand_timestep = {s: pd.Series(np.random.uniform(50, 200, len(time_steps)), index=time_steps) for s in stations}
    pv_generation = {s: pd.Series(np.sin(np.linspace(0, np.pi, len(time_steps))) ** 2 * 300, index=time_steps) for s in
                     stations}

    model_data = {
        'traffic_graph': G, 'locations': locations_data, 'tasks': tasks,
        'vehicles': vehicles, 'stations': stations,
        'dist_matrix': dist_matrix, 'time_matrix': time_df, 'path_matrix': path_matrix,
        'power_grid': power_grid, 'power_buses': power_buses, 'power_lines': list(power_grid.edges()),
        'line_params': {line: {'r_pu': 0.01, 'x_pu': 0.05} for line in power_grid.edges()},
        'station_to_bus_map': station_to_bus_map,
        'substation_bus': 'Bus_Substation_Slack',
        'time_steps': list(time_steps), 'electricity_prices': electricity_prices,
        'pv_generation': pv_generation, 'ev_demand_timestep': ev_demand_timestep
    }
    print("=" * 30)
    print("全新的【城市配送】场景数据创建完成！")
    return model_data