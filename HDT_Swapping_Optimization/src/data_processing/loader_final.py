# src/data_processing/loader_final.py

import pandas as pd
import numpy as np
import os
import random
import networkx as nx
from src.simulation import road_network
import pandapower as pp
from scipy.stats import norm


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.model_data = {}
        self.road_network = None

    def get_stochastic_pv_generation(self, time_steps, time_step_hours, peak_power, noise_level):
        """生成带有随机云层遮挡效应的光伏出力曲线"""
        hours = np.array([t * time_step_hours for t in time_steps])
        sin_wave = np.sin(np.maximum(0, hours - 6) / 12 * np.pi)
        base_pv = np.maximum(0, sin_wave) ** 1.5 * peak_power
        cloud_noise = pd.Series(np.random.normal(0, noise_level, len(time_steps))).rolling(window=4,
                                                                                           min_periods=1).mean()
        cloud_factor = np.clip(1 - cloud_noise, 1 - noise_level, 1.0)
        stochastic_pv = np.maximum(0, base_pv * cloud_factor)
        return pd.Series(stochastic_pv, index=time_steps)

    def get_stochastic_ev_demand(self, time_steps, time_step_hours, peak_kw, noise_level):
        """生成带有随机性的双高峰EV充电需求曲线"""
        if peak_kw == 0:  # If EV demand is turned off, return zeros
            return pd.Series(0, index=time_steps)

        hours = np.array([t * time_step_hours for t in time_steps])
        peak1 = norm.pdf(hours, loc=8.5, scale=1.5)
        peak2 = norm.pdf(hours, loc=18, scale=2.5)
        base_demand = (peak1 * 0.6 + peak2) / np.max(peak1 * 0.6 + peak2) * peak_kw
        noise = np.random.normal(0, peak_kw * noise_level, len(time_steps))
        stochastic_demand = np.maximum(0, base_demand + noise)
        return pd.Series(stochastic_demand, index=time_steps)

    def load_all(self):
        """This method now orchestrates the entire data creation process."""
        print("=" * 30)
        print("开始创建全新的【城市配送+配网交互】场景...")
        config = self.config
        random.seed(42)
        np.random.seed(42)

        G, pos = road_network.generate_connected_graph(config.CITY_NODE_COUNT, config.CITY_GRAPH_RADIUS)

        for node, p in pos.items():
            G.nodes[node]['pos'] = p
            G.nodes[node]['info'] = {'name': str(node)}

        self.road_network = G

        try:
            net = pp.networks.case118()
            load_buses = net.load.bus.to_list()
            print(f"成功加载IEEE 118节点系统，找到 {len(load_buses)} 个可用的负荷节点。")
        except Exception as e:
            print(f"加载pandapower case118失败: {e}。将使用备用节点列表。")
            load_buses = list(range(1, 119))

        all_nodes = list(G.nodes())
        random.shuffle(all_nodes)

        depot_nodes = all_nodes[:config.NUM_DEPOTS]
        station_nodes = all_nodes[config.NUM_DEPOTS: config.NUM_DEPOTS + config.NUM_STATIONS]
        customer_nodes = all_nodes[
                         config.NUM_DEPOTS + config.NUM_STATIONS: config.NUM_DEPOTS + config.NUM_STATIONS + config.NUM_CUSTOMERS]

        locations_data = {}
        for i, node in enumerate(depot_nodes):
            name = f"Depot_{i + 1}"
            locations_data[name] = {'type': 'Depot', 'pos': pos[node], 'node_id': node}
            G.nodes[node]['info'] = {'name': name, 'type': 'Depot'}

        for i, node in enumerate(customer_nodes):
            name = f"Customer_{i + 1}"
            locations_data[name] = {'type': 'Customer', 'pos': pos[node], 'node_id': node}
            G.nodes[node]['info'] = {'name': name, 'type': 'Customer'}

        stations_info = {}
        station_to_bus_map = {}
        random.shuffle(load_buses)
        for i, node in enumerate(station_nodes):
            station_name = f"Station_{i + 1}"
            bus_id = load_buses[i % len(load_buses)]
            locations_data[station_name] = {'type': 'SwapStation', 'pos': pos[node], 'node_id': node}
            G.nodes[node]['info'] = {'name': station_name, 'type': 'SwapStation'}
            stations_info[station_name] = {'initial_full': 10, 'initial_empty': 5, 'bus_id': bus_id}
            station_to_bus_map[station_name] = bus_id
        print(f"场景设施点分配完毕: {len(depot_nodes)}仓库, {len(stations_info)}换电站, {len(customer_nodes)}客户。")

        # --- MODIFIED: Corrected logic to calculate distance matrices ---
        all_location_node_ids = [d['node_id'] for d in locations_data.values()]
        # 1. Call the correct, existing function with node IDs
        dist_matrix_raw, path_matrix_raw = road_network.get_path_and_distance_matrices(G, all_location_node_ids)
        # 2. Create a map from the node ID back to the location name
        node_id_to_name_map = {d['node_id']: name for name, d in locations_data.items()}
        # 3. Rename the matrix indices and columns to use the location names
        dist_matrix_renamed = dist_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)
        path_matrix = path_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)

        dist_matrix = dist_matrix_renamed * config.CITY_SCALE_KM
        print(f"路网已缩放以模拟真实的城市环境，城市尺度约为 {config.CITY_SCALE_KM} 公里。")
        avg_speed_kmh = 30.0
        time_df = dist_matrix / avg_speed_kmh

        vehicles = {}
        depot_names = [name for name, info in locations_data.items() if info['type'] == 'Depot']
        for i in range(config.NUM_TRUCKS):
            truck_id = f"HDT_{i + 1}"
            vehicles[truck_id] = {'initial_soc': config.HDT_BATTERY_CAPACITY_KWH,
                                  'depot_id': random.choice(depot_names)}

        tasks = {}
        task_id_counter = 1
        customer_names = [name for name, info in locations_data.items() if info['type'] == 'Customer']
        for cust_name in customer_names:
            task_id = f"Task_{task_id_counter}"
            closest_depot = min(depot_names, key=lambda d: dist_matrix.loc[d, cust_name])

            one_way_time = time_df.loc[closest_depot, cust_name]
            earliest_possible_due_time = one_way_time + config.LOADING_UNLOADING_TIME_HOURS + 1.0
            latest_possible_due_time = earliest_possible_due_time + 8.0
            due_time = round(random.uniform(earliest_possible_due_time, latest_possible_due_time), 1)

            tasks[task_id] = {
                'delivery_to': cust_name,
                'demand': round(random.uniform(1.0, 2.5), 2),
                'due_time': due_time,
                'depot': closest_depot
            }
            task_id_counter += 1
        print(f"已生成 {len(vehicles)} 辆卡车和 {len(tasks)} 个任务的中央任务池。")

        time_steps = range(config.TOTAL_TIME_STEPS)
        electricity_prices = pd.Series(
            [0.4 if 7 <= t * config.TIME_STEP_HOURS < 11 or 14 <= t * config.TIME_STEP_HOURS < 19
             else (1.2 if 11 <= t * config.TIME_STEP_HOURS < 14 or 19 <= t * config.TIME_STEP_HOURS < 21 else 0.8)
             for t in time_steps], index=time_steps)

        pv_generation = {
            s: self.get_stochastic_pv_generation(time_steps, config.TIME_STEP_HOURS, config.PV_PEAK_POWER_KW,
                                                 config.PV_CLOUD_NOISE_LEVEL) for s in stations_info}
        ev_demand_timestep = {
            s: self.get_stochastic_ev_demand(time_steps, config.TIME_STEP_HOURS, config.EV_DEMAND_PEAK_KW,
                                             config.EV_DEMAND_NOISE_LEVEL) for s in stations_info}

        self.model_data = {
            'traffic_graph': G, 'locations': locations_data, 'tasks': tasks,
            'vehicles': vehicles, 'stations': stations_info,
            'dist_matrix': dist_matrix, 'time_matrix': time_df, 'path_matrix': path_matrix,
            'power_grid_net': net,
            'station_to_bus_map': station_to_bus_map,
            'time_steps': list(time_steps), 'electricity_prices': electricity_prices,
            'pv_generation': pv_generation, 'ev_demand_timestep': ev_demand_timestep
        }
        print("=" * 30)
        print("全新的【城市配送+配网交互】场景数据创建完成！")
        return self.model_data