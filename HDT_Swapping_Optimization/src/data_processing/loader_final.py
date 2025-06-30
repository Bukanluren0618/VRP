# HDT_Swapping_Optimization/src/data_processing/loader_final.py

import pandas as pd
import numpy as np
import os
from src.common import config_final as config
from src.simulation import road_network


def create_final_scenario():
    """
    创建最终版的、基于真实路网的、包含所有细节的综合场景数据。
    """
    print("开始创建最终版综合场景...")

    # --- 1. 定义关键位置节点 ---
    key_locations = {
        'Depot': (0, 0),
        'Task1_Pickup': (80, 60), 'Task1_Delivery': (-20, 70),
        'Task2_Pickup': (40, -30), 'Task2_Delivery': (90, -50),
        'Task3_Pickup': (-50, -80), 'Task3_Delivery': (-80, 20),
        'SwapStation1': (10, 80),
        'SwapStation2': (-20, -60),
    }

    # --- 2. 创建路网并计算路径 ---
    G = road_network.create_road_network(key_locations)
    dist_matrix, path_matrix = road_network.get_path_and_distance_matrices(G, list(key_locations.keys()))
    time_df = dist_matrix / (config.HDT_BASE_CONSUMPTION_KWH_PER_KM * 0.8)

    # --- 3. 定义一个固定的、有挑战性的任务集 ---
    tasks = {
        'Task1': {'pickup': 'Task1_Pickup', 'delivery': 'Task1_Delivery', 'weight': 15.0, 'ready_time': 8.0,
                  'due_time': 15.0},
        'Task2': {'pickup': 'Task2_Pickup', 'delivery': 'Task2_Delivery', 'weight': 10.0, 'ready_time': 9.0,
                  'due_time': 18.0},
        'Task3': {'pickup': 'Task3_Pickup', 'delivery': 'Task3_Delivery', 'weight': 12.0, 'ready_time': 10.0,
                  'due_time': 20.0}
    }

    # --- 4. 定义HDT车辆 ---
    vehicles = {'HDT1': {'initial_soc': 350.0}, 'HDT2': {'initial_soc': 315.0}, 'HDT3': {'initial_soc': 350.0}}

    # --- 5. 定义能源站初始状态 ---
    stations = {
        'SwapStation1': {'initial_full': 8, 'initial_empty': 2},
        'SwapStation2': {'initial_full': 6, 'initial_empty': 4}
    }

    # --- 6. 处理EV充电需求 ---
    time_steps = range(config.TOTAL_TIME_STEPS)
    ev_demand_timestep = {s: pd.Series(0.0, index=time_steps) for s in stations}
    charge_load_filepath = os.path.join(config.RAW_DATA_DIR, config.CHARGE_LOAD_FILENAME)
    try:
        print(f"尝试从 '{charge_load_filepath}' 读取真实充电负荷数据...")
        charge_df = pd.read_csv(charge_load_filepath, encoding='gbk')
        if '开始时间' not in charge_df.columns or '电量' not in charge_df.columns:
            raise ValueError("CSV文件中缺少'开始时间'或'电量'列。")
        charge_df['开始时间'] = pd.to_datetime(charge_df['开始时间'])
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

    # --- 7. 生成其他时间序列数据 ---
    prices = [0.4 if 23 <= t * config.TIME_STEP_HOURS or t * config.TIME_STEP_HOURS < 7 else (
        1.2 if (7 <= t * config.TIME_STEP_HOURS < 11 or 18 <= t * config.TIME_STEP_HOURS < 21) else 0.8) for t in
              time_steps]
    electricity_prices = pd.Series(prices, index=time_steps)
    pv_profile = np.sin(np.linspace(0, np.pi, config.TOTAL_TIME_STEPS)) ** 2
    pv_generation = {s: pd.Series(pv_profile * p, index=time_steps) for s, p in config.PV_PEAK_POWER_KW.items()}

    # --- 8. 组装最终数据字典 ---
    model_data = {
        'graph': G, 'locations': key_locations, 'tasks': tasks, 'vehicles': vehicles,
        'stations': stations, 'time_steps': list(time_steps),
        'dist_matrix': dist_matrix, 'time_matrix': time_df, 'path_matrix': path_matrix,
        'electricity_prices': electricity_prices, 'pv_generation': pv_generation,
        'ev_demand_timestep': ev_demand_timestep
    }
    print("最终版场景数据创建完成！")
    return model_data