# src/data_processing/loader_final.py

import pandas as pd
import numpy as np
import os
import random
import networkx as nx
from src.simulation import road_network
import pandapower as pp
from scipy.stats import norm
import osmnx as ox


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.model_data = {}
        self.road_network = None

    def get_stochastic_pv_generation(self, time_steps, time_step_hours, peak_power, noise_level):
        hours = np.array([t * time_step_hours for t in time_steps])
        sin_wave = np.sin(np.maximum(0, hours - 6) / 12 * np.pi)
        base_pv = np.maximum(0, sin_wave) ** 1.5 * peak_power
        cloud_noise = pd.Series(np.random.normal(0, noise_level, len(time_steps))).rolling(window=4,
                                                                                           min_periods=1).mean()
        cloud_factor = np.clip(1 - cloud_noise, 1 - noise_level, 1.0)
        stochastic_pv = np.maximum(0, base_pv * cloud_factor)
        return pd.Series(stochastic_pv, index=time_steps)

    def get_stochastic_ev_demand(self, time_steps, time_step_hours, peak_kw, noise_level):
        if peak_kw == 0: return pd.Series(0, index=time_steps)
        hours = np.array([t * time_step_hours for t in time_steps])
        peak1 = norm.pdf(hours, loc=8.5, scale=1.5)
        peak2 = norm.pdf(hours, loc=18, scale=2.5)
        base_demand = (peak1 * 0.6 + peak2) / np.max(peak1 * 0.6 + peak2) * peak_kw
        noise = np.random.normal(0, peak_kw * noise_level, len(time_steps))
        stochastic_demand = np.maximum(0, base_demand + noise)
        return pd.Series(stochastic_demand, index=time_steps)

    def load_all(self):
        print("=" * 30);
        print("开始创建场景...")
        config = self.config
        random.seed(42);
        np.random.seed(42)

        # --- MODIFIED: 根据config选择路网模型 ---
        if config.ROAD_MODEL == 'waxman':
            G, pos = road_network.generate_city_like_graph(config.CITY_NODE_COUNT, alpha=config.WAXMAN_ALPHA,
                                                           beta=config.WAXMAN_BETA)
            scaling_factor = config.CITY_SCALE_KM
        elif config.ROAD_MODEL == 'real_world':
            G = road_network.generate_real_road_network(city_name=config.CITY_NAME)
            pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
            scaling_factor = 1  # OSMnx距离单位是米，会在get_path_and_distance_matrices中转换为km
        else:
            raise ValueError(f"未知的路网模型: {config.ROAD_MODEL}")

        self.road_network = G

        try:
            net = pp.networks.case118()
            load_buses = net.load.bus.to_list();
            print(f"成功加载IEEE 118节点系统，找到 {len(load_buses)} 个负荷节点。")
        except Exception as e:
            print(f"加载pandapower case118失败: {e}。将使用备用节点列表。");
            load_buses = list(range(1, 119))

        all_nodes = list(G.nodes())
        random.shuffle(all_nodes)

        num_facilities = config.NUM_DEPOTS + config.NUM_STATIONS + config.NUM_CUSTOMERS
        if len(all_nodes) < num_facilities:
            raise ValueError(f"路网节点数 ({len(all_nodes)}) 不足以容纳所有设施点 ({num_facilities})")

        depot_nodes = all_nodes[:config.NUM_DEPOTS]
        station_nodes = all_nodes[config.NUM_DEPOTS: config.NUM_DEPOTS + config.NUM_STATIONS]
        customer_nodes = all_nodes[
                         config.NUM_DEPOTS + config.NUM_STATIONS: config.NUM_DEPOTS + config.NUM_STATIONS + config.NUM_CUSTOMERS]

        locations_data = {};
        node_id_to_name_map = {}
        for i, node_id in enumerate(depot_nodes):
            name = f"Depot_{i + 1}";
            locations_data[name] = {'type': 'Depot', 'node_id': node_id};
            node_id_to_name_map[node_id] = name
        for i, node_id in enumerate(customer_nodes):
            name = f"Customer_{i + 1}";
            locations_data[name] = {'type': 'Customer', 'node_id': node_id};
            node_id_to_name_map[node_id] = name

        stations_info = {};
        station_to_bus_map = {}
        random.shuffle(load_buses)
        for i, node_id in enumerate(station_nodes):
            name = f"Station_{i + 1}";
            locations_data[name] = {'type': 'SwapStation', 'node_id': node_id};
            node_id_to_name_map[node_id] = name
            bus_id = load_buses[i % len(load_buses)];
            stations_info[name] = {'initial_full': 10, 'initial_empty': 5, 'bus_id': bus_id};
            station_to_bus_map[name] = bus_id

        print(f"场景设施点已放置: {len(depot_nodes)}仓库, {len(station_nodes)}换电站, {len(customer_nodes)}客户。")

        all_location_node_ids = [d['node_id'] for d in locations_data.values()]
        dist_matrix_raw, path_matrix_nodes = road_network.get_path_and_distance_matrices(G, all_location_node_ids)

        dist_matrix_renamed = dist_matrix_raw.rename(index=node_id_to_name_map, columns=node_id_to_name_map)
        dist_matrix = dist_matrix_renamed * scaling_factor

        avg_speed_kmh = 40.0
        time_df = dist_matrix / avg_speed_kmh

        vehicles = {};
        depot_names = [name for name, info in locations_data.items() if info['type'] == 'Depot']
        for i in range(config.NUM_TRUCKS):
            truck_id = f"HDT_{i + 1}";
            vehicles[truck_id] = {'initial_soc': config.HDT_BATTERY_CAPACITY_KWH,
                                  'depot_id': random.choice(depot_names)}
        tasks = {};
        task_id_counter = 1;
        customer_names = [name for name, info in locations_data.items() if info['type'] == 'Customer']
        for cust_name in customer_names:
            task_id = f"Task_{task_id_counter}";
            closest_depot = min(depot_names, key=lambda d: dist_matrix.loc[d, cust_name])
            one_way_time = time_df.loc[closest_depot, cust_name];
            earliest_due = one_way_time + config.LOADING_UNLOADING_TIME_HOURS + 1.0
            latest_due = earliest_due + 8.0;
            due_time = round(random.uniform(earliest_due, latest_due), 1)
            tasks[task_id] = {'delivery_to': cust_name, 'demand': round(random.uniform(1.0, 2.5), 2),
                              'due_time': due_time, 'depot': closest_depot}
            task_id_counter += 1
        print(f"已生成 {len(vehicles)} 辆卡车和 {len(tasks)} 个任务的中央任务池。")

        time_steps = range(config.TOTAL_TIME_STEPS)
        electricity_prices = pd.Series([
                                           0.4 if 7 <= t * config.TIME_STEP_HOURS < 11 or 14 <= t * config.TIME_STEP_HOURS < 19 else (
                                               1.2 if 11 <= t * config.TIME_STEP_HOURS < 14 or 19 <= t * config.TIME_STEP_HOURS < 21 else 0.8)
                                           for t in time_steps], index=time_steps)
        pv_generation = {
            s: self.get_stochastic_pv_generation(time_steps, config.TIME_STEP_HOURS, config.PV_PEAK_POWER_KW,
                                                 config.PV_CLOUD_NOISE_LEVEL) for s in stations_info}
        ev_demand_timestep = {
            s: self.get_stochastic_ev_demand(time_steps, config.TIME_STEP_HOURS, config.EV_DEMAND_PEAK_KW,
                                             config.EV_DEMAND_NOISE_LEVEL) for s in stations_info}

        self.model_data = {
            'traffic_graph': G, 'locations': locations_data, 'tasks': tasks, 'vehicles': vehicles,
            'stations': stations_info,
            'dist_matrix': dist_matrix, 'time_matrix': time_df, 'path_matrix': None,
            'power_grid_net': net, 'station_to_bus_map': station_to_bus_map, 'time_steps': list(time_steps),
            'electricity_prices': electricity_prices, 'pv_generation': pv_generation,
            'ev_demand_timestep': ev_demand_timestep
        }
        print("=" * 30);
        print("场景数据创建完成！");
        return self.model_data