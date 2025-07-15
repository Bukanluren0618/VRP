# HDT_Swapping_Optimization/src/data_processing/loader_final.py

import pandas as pd
import numpy as np
import os
import networkx as nx
from src.common import config_final as config
from src.simulation import road_network


def create_final_scenario():
    """
    【最终耦合版】: 创建包含交通路网和配电网的耦合场景，并设定为城市多点配送(CVRP)模式。
    """
    print("开始创建最终耦合版综合场景...")

    # --- 1. 定义交通网络与客户点 (CVRP场景) ---
    # 仓库位于城外，客户点和换电站位于城内
    key_locations = {
        'Depot': (-120, -120),
        'Customer1': (80, 60),
        'Customer2': (40, -30),
        'Customer3': (-50, -80),
        'Customer4': (20, 90),
        'Customer5': (-80, 20),
        'SwapStation1': (10, 80),
        'SwapStation2': (-20, -60),
        # 'Depot': (0, 0),
        # 'Customer1': (20, 10),
        # 'Customer2': (15, -15),
        # 'Customer3': (-25, -20),
        # 'Customer4': (10, 25),
        # 'Customer5': (-30, 5),
        # 'SwapStation1': (5, 20),
        # 'SwapStation2': (-10, -15),
    }

    # --- 2. 创建交通路网并计算各关键点之间的最短路径 ---
    traffic_graph = road_network.create_road_network(key_locations)
    dist_matrix, path_matrix = road_network.get_path_and_distance_matrices(
        traffic_graph, list(key_locations.keys())
    )
    # 将整张路网距离缩小 10 倍，以保证行驶里程更合理
    dist_matrix = dist_matrix / 1.0
    # 假设平均速度受路况影响，降低20%
    time_df = dist_matrix / (config.HDT_AVERAGE_SPEED_KMPH * 0.8)

    # --- 3. 定义多点配送任务 ---
    # 每个任务代表一个需要从Depot配送货物的客户
    tasks = {
        # 'Task_C1': {'delivery_to': 'Customer1', 'demand': 5.0, 'due_time': 18.0},
        # 'Task_C2': {'delivery_to': 'Customer2', 'demand': 4.0, 'due_time': 18.0},
        # 'Task_C3': {'delivery_to': 'Customer3', 'demand': 6.0, 'due_time': 18.0},
        # 'Task_C4': {'delivery_to': 'Customer4', 'demand': 3.0, 'due_time': 18.0},
        # 'Task_C5': {'delivery_to': 'Customer5', 'demand': 7.0, 'due_time': 18.0},
        'Task_C1': {'delivery_to': 'Customer1', 'demand': 5.0, 'due_time': 24.0},
        'Task_C2': {'delivery_to': 'Customer2', 'demand': 4.0, 'due_time': 24.0},
        'Task_C3': {'delivery_to': 'Customer3', 'demand': 6.0, 'due_time': 24.0},
        'Task_C4': {'delivery_to': 'Customer4', 'demand': 3.0, 'due_time': 24.0},
        'Task_C5': {'delivery_to': 'Customer5', 'demand': 7.0, 'due_time': 24.0},
    }

    # --- 4. 定义HDT车辆与能源站初始状态 ---
    vehicles = {
        'HDT1': {'initial_soc': config.HDT_BATTERY_CAPACITY_KWH},
        'HDT2': {'initial_soc': config.HDT_BATTERY_CAPACITY_KWH}
    }
    stations = {'SwapStation1': {'initial_full': 15, 'initial_empty': 5},
                'SwapStation2': {'initial_full': 15, 'initial_empty': 5}}

    # --- 5. 【核心新增】创建配电网络 ---
    print("正在创建配电网络拓扑...")
    power_grid = nx.DiGraph()  # 配电网是有向图
    # 添加节点 (母线)，一个变电站(Substation)和多个负荷/换电站母线
    power_buses = ['Bus_Sub', 'Bus_1', 'Bus_2', 'Bus_3', 'Bus_4', 'Bus_5']
    power_grid.add_nodes_from(power_buses)
    # 添加线路 (有向边)，构成一个典型的辐射状网络
    power_lines = [('Bus_Sub', 'Bus_1'), ('Bus_1', 'Bus_2'), ('Bus_1', 'Bus_3'), ('Bus_3', 'Bus_4'), ('Bus_3', 'Bus_5')]
    power_grid.add_edges_from(power_lines)

    # 将换电站连接到电网的特定母线
    station_to_bus_map = {
        'SwapStation1': 'Bus_2',
        'SwapStation2': 'Bus_4'
    }

    # 计算线路的标幺化阻抗值
    base_impedance = (config.BASE_VOLTAGE_KV ** 2) / config.BASE_POWER_MVA
    line_params = {}
    # 假设每段线路长度为2公里
    for u, v in power_lines:
        line_length_km = 2.0
        r_pu = (line_length_km * config.LINE_RESISTANCE_OHM_PER_KM) / base_impedance
        x_pu = (line_length_km * config.LINE_REACTANCE_OHM_PER_KM) / base_impedance
        line_params[(u, v)] = {'r_pu': r_pu, 'x_pu': x_pu}
    print("配电网络创建完成。")

    # --- 6. 生成时间序列数据 (EV, 电价, 光伏) ---
    time_steps = range(config.TOTAL_TIME_STEPS)
    ev_demand_timestep = {s: pd.Series(0.0, index=time_steps) for s in stations}
    charge_load_filepath = os.path.join(config.RAW_DATA_DIR, config.CHARGE_LOAD_FILENAME)
    try:
        print(f"尝试从 '{charge_load_filepath}' 读取真实充电负荷数据...")
        charge_df = pd.read_csv(charge_load_filepath, encoding='gbk')
        if '开始时间' not in charge_df.columns or '电量' not in charge_df.columns:
            raise ValueError("CSV文件中缺少'开始时间'或'电量'列。")
        # charge_df['开始时间'] = pd.to_datetime(charge_df['开始时间'])
        charge_df['开始时间'] = pd.to_datetime(
            charge_df['开始时间'], format='%H:%M', errors='coerce')
        charge_df.dropna(subset=['开始时间'], inplace=True)
        charge_df['hour'] = charge_df['开始时间'].dt.hour
        hourly_load = charge_df.groupby('hour')['电量'].sum()
        total_real_load = hourly_load.sum()
        hourly_load_profile = hourly_load / total_real_load if total_real_load > 0 else hourly_load
        total_simulated_demand = 25 * config.EV_AVG_CHARGE_DEMAND_KWH
        for t in time_steps:
            hour_of_day = int(t * config.TIME_STEP_HOURS)
            ev_demand_timestep['SwapStation1'][t] = total_simulated_demand * hourly_load_profile.get(hour_of_day, 0)
        print("成功！已根据您的真实数据文件生成EV充电需求曲线。")
    except Exception as e:
        print(f"警告: 处理真实数据文件时发生错误 ({e})。将回退到随机生成EV需求。")
        ev_demand_timestep = {s: pd.Series(0.0, index=time_steps) for s in stations}
        num_evs = 25
        for i in range(num_evs):
            arrival_hour = np.random.uniform(7, 21)
            station_id = np.random.choice(list(stations.keys()))
            demand_kwh = config.EV_AVG_CHARGE_DEMAND_KWH * np.random.uniform(0.8, 1.2)
            arrival_step = int(arrival_hour / config.TIME_STEP_HOURS)
            if arrival_step < config.TOTAL_TIME_STEPS:
                ev_demand_timestep[station_id][arrival_step] += demand_kwh

    prices = [0.4 if 23 <= t * config.TIME_STEP_HOURS or t * config.TIME_STEP_HOURS < 7 else (
        1.2 if (7 <= t * config.TIME_STEP_HOURS < 11 or 18 <= t * config.TIME_STEP_HOURS < 21) else 0.8) for t in
              time_steps]
    electricity_prices = pd.Series(prices, index=time_steps)
    pv_profile = np.sin(np.linspace(0, np.pi, config.TOTAL_TIME_STEPS)) ** 2
    pv_generation = {s: pd.Series(pv_profile * p, index=time_steps) for s, p in config.PV_PEAK_POWER_KW.items()}

    # --- 7. 组装最终数据字典 ---
    model_data = {
        'traffic_graph': traffic_graph, 'locations': key_locations, 'tasks': tasks,
        'vehicles': vehicles, 'stations': stations,
        'dist_matrix': dist_matrix, 'time_matrix': time_df, 'path_matrix': path_matrix,
        'power_grid': power_grid, 'power_buses': power_buses, 'power_lines': power_lines,
        'line_params': line_params, 'station_to_bus_map': station_to_bus_map,
        'substation_bus': 'Bus_Sub',
        'time_steps': list(time_steps), 'electricity_prices': electricity_prices,
        'pv_generation': pv_generation, 'ev_demand_timestep': ev_demand_timestep
    }

    print("最终耦合版场景数据创建完成！")
    return model_data